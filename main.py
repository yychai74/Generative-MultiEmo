import argparse
import os
import logging
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import AdamW, T5Tokenizer
from model import modeling_t5
import transformers

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data_utils import EmotionDataset
from eval_utils import evaluate, extract_GoEmotion
from contrastive_loss import SupConLoss
from label_co_existence import get_co_existence
from sklearn.metrics import classification_report, jaccard_score

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='GoEmotions', type=str, help='select from GoEmotions and SemEvalEc')
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path of pre-trained model")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_eval", action='store_true', default=True)
    parser.add_argument("--CLP", action='store_true', default=False)
    parser.add_argument("--send_CLP", action="store_true", default=False)

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=[0])
    parser.add_argument("--num_beams", default=3, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=4e-5, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    if not os.path.exists('./output'):
        os.mkdir('./output')

    output_dir = f"./output/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    args.output_dir = output_dir
    args.device = args.n_gpu[0]

    return args


def get_dataset(tokenizer, data_type, args):
    if args.CLP:
        max_len = args.max_seq_length - label_size
    else:
        max_len = args.max_seq_length
    return EmotionDataset(tokenizer=tokenizer, data_dir=args.dataset, data_type=data_type,
                          max_len=max_len)


def get_label_size():
    return label_size


def send_CLP():
    return args.send_CLP


class T5EmotionGeneration(pl.LightningModule):
    def __init__(self, hparams):
        super(T5EmotionGeneration, self).__init__()
        self.hyparams = hparams
        if args.CLP:
            self.model = modeling_t5.T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        else:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)

        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
        self.alpha = torch.tensor(0.9)
        self.temperature = 0.3 if args.dataset == 'GoEmotions' else 0.07
        self.loss_scl = SupConLoss(temperature=self.temperature)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None,
                past_key_values=None):
        if args.CLP:
            label_mask = torch.ones(input_ids.shape[0], label_size, dtype=torch.int).to(attention_mask.device)
            attention_mask = torch.cat((label_mask, attention_mask), dim=1)
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch, validate=False):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask'],
        )

        loss1 = outputs[0]

        if validate or not args.CLP:
            return loss1
        else:
            normed_prompts = F.normalize(self.model.encoder.label, dim=1)
            loss2 = self.loss_scl(normed_prompts, labels=batch["labels_idx"], weight=weight)

            return self.alpha * loss1 + (1 - self.alpha) * loss2

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, validate=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        print("avg_val_loss", avg_loss)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hyparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyparams.learning_rate, eps=self.hyparams.adam_epsilon)
        self.opt = optimizer

        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        if self.trainer.global_step < self.hyparams.warmup_steps:
            lr_scale = float(self.trainer.global_step + 1) / self.hyparams.warmup_steps

        elif self.trainer.global_step < self.total_step:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.total_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * self.hyparams.learning_rate

        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        train_dataset = get_dataset(self.tokenizer, data_type='train', args=self.hyparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hyparams.train_batch_size, drop_last=True,
                                num_workers=4, shuffle=True)
        t_total = (
                (len(dataloader.dataset) // (self.hyparams.train_batch_size * max(1, len(self.hyparams.n_gpu))))
                // self.hyparams.gradient_accumulation_steps
                * float(self.hyparams.num_train_epochs)
        )

        self.total_step = t_total
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(self.tokenizer, data_type='dev', args=self.hyparams)
        return DataLoader(val_dataset, batch_size=self.hyparams.eval_batch_size, num_workers=4)


args = init_args()
weight = torch.from_numpy(get_co_existence(args.dataset))

if args.dataset == 'GoEmotions':
    label_size = 28
    label_list = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                  'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                  'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                  'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
elif args.dataset == 'SemEvalEc':
    label_size = 11
    label_list = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism',
                  'sadness', 'surprise', 'trust']
else:
    raise NameError("No such dataset")

if __name__ == '__main__':
    # print(weight)

    if args.do_train:
        print("\n", "=" * 30, f"NEW EXP: train on {args.dataset}", "=" * 30, "\n")
        seed_everything(args.seed)
        model = T5EmotionGeneration(hparams=args)
        tokenizer = model.tokenizer

        print(f"Here are examples from `{args.dataset}`")

        dataset = get_dataset(tokenizer=tokenizer, data_type='dev', args=args)
        data_sample = dataset[0]
        data_sample1 = dataset[109]
        print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
        print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
        print(data_sample["labels_idx"])
        print('Input :', tokenizer.decode(data_sample1['source_ids'], skip_special_tokens=True))
        print('Output:', tokenizer.decode(data_sample1['target_ids'], skip_special_tokens=True))
        print(data_sample1["labels_idx"])
        print("\n****** Conduct Training ******")

        callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir, monitor='val_loss', mode='min', save_top_k=3
        )
        log = TensorBoardLogger('logs', name=args.dataset)

        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            # strategy="dp",
            gradient_clip_val=1.0,
            # amp_level='O1',
            max_epochs=args.num_train_epochs,
            callbacks=[callback, EarlyStopping(monitor="val_loss", patience=3, mode="min")],
            logger=log,
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

    if args.do_eval:
        all_checkpoints = []
        best_micro_f1, best_macro_f1, best_jaccord = 0, 0, 0

        for f in os.listdir(args.output_dir):
            file_name = os.path.join(args.output_dir, f)
            if 'ckpt' in file_name:
                all_checkpoints.append(file_name)
        print(f'Test model on following checkpoints: {all_checkpoints}')

        for ckpt in all_checkpoints:
            model = T5EmotionGeneration.load_from_checkpoint(f'{ckpt}', hparams=args)
            tokenizer = model.tokenizer
            test_dataset = get_dataset(tokenizer=tokenizer, data_type='test', args=args)
            device = torch.device(f'cuda:{args.device}')
            model.model.to(device)
            model.model.eval()

            test_dataloader = DataLoader(test_dataset, batch_size=16)
            outputs, targets = [], []
            print(f'results on {ckpt}')

            for batch in tqdm(test_dataloader):
                if args.CLP:
                    label_mask = torch.ones(batch['source_ids'].shape[0], label_size, dtype=torch.int).to(
                        f'cuda:{args.device}')
                    attention_mask = torch.cat((label_mask, batch['source_mask'].to(device)), dim=1)
                else:
                    attention_mask = batch['source_mask'].to(device)
                outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                            attention_mask=attention_mask, max_length=128, num_beams=args.num_beams)
                # **model_kwargs)
                outputs.append([tokenizer.decode(ids, skip_special_tokens=True) for ids in outs])
                targets.append([tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]])
            results = evaluate(outputs, targets, args.dataset)
            if results['classification']['micro avg']['f1-score'] > best_micro_f1:
                best_micro_f1, best_macro_f1, best_jaccord = results['classification']['micro avg']['f1-score'], \
                                                             results['classification']['macro avg']['f1-score'], \
                                                             results['jaccard_score']
                best_out, labels = outputs, targets

            print("/*******************************/")

        print(f'best_micro_f1: {best_micro_f1}, macro_f1: {best_macro_f1}, jaccord: {best_jaccord}')
