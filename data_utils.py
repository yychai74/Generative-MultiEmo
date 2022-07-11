from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import random


def get_sem_data(path, data_type):
    sents, labels, label_idx = [], [], []
    df = pd.read_csv(path, sep="	")
    processed_sen_df = pd.read_csv(f'data/SemEvalEc/processed_{data_type}.csv')
    lab = df.iloc[:, 2:]
    all_label = [emotion for emotion in lab.columns]
    sents_array = processed_sen_df['0'].values.tolist()  # (6838,)
    label_array = df.iloc[:, 2:].values.tolist()  # （6838,11)

    for temp in label_array:

        if np.asarray(temp).sum() == 0:
            labels.append('The emotion neutral is expressed in this sentence.')
            label_idx.append(torch.tensor(temp, dtype=torch.float32))
            continue
        # temp.append(0)
        label_idx.append(torch.tensor(temp, dtype=torch.float32))

        label_list = []
        index = [inx for inx, num in enumerate(temp) if num == 1]
        i_label_strs = [all_label[a] for a in index]
        for emo in i_label_strs:
            label_list.append(f'The emotion {emo} is expressed in this sentence.')
        labels.append(' [SSEP] '.join(label_list))

    for sen in sents_array:
        sen = sen.split()
        if sen != '':
            sents.append(sen)

    return sents, labels, label_idx


def get_GoEmotions_data(path, data_dir):
    sents, targets, label_idx = [], [], []
    df = pd.read_csv(path, sep="\t", names=['sentence', 'label', 'null'])
    emotion_label = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                     'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                     'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                     'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

    emotion_label_dict = {idx: emo for idx, emo in enumerate(emotion_label)}
    df_sents = df.iloc[:, 0]
    df_labels = df.iloc[:, 1]
    sents_list = df_sents.values.tolist()
    labels_idx = df_labels.values.tolist()

    for index in labels_idx:
        idx_list = index.split(',')
        emotion_list = []
        label = torch.zeros(len(emotion_label), dtype=torch.float32)

        for num in idx_list:
            emo_idx = int(num)
            emo = emotion_label_dict[emo_idx]
            emotion_list.append(f'The emotion {emo} is expressed in this sentence.')
            # emotion_list.append(f'{emo}')
            # emotion_list.append(f'It expressed emotion {emo}.')
            label[emo_idx] = 1
        targets.append(' [SSEP] '.join(emotion_list))
        label_idx.append(label)

    for sen in sents_list:
        sen = sen.strip()
        temp = sen.split()
        if temp != '':
            sents.append(temp)

    return sents, targets, label_idx


def get_transformed_data(data_path, data_dir, data_type):
    if data_dir == 'SemEvalEc':
        sents, label, labels_idx = get_sem_data(data_path, data_type)
    else:
        sents, label, labels_idx = get_GoEmotions_data(data_path, data_dir)


    return sents, label, labels_idx


class EmotionDataset(Dataset):
    def __init__(self, tokenizer, data_type, data_dir, max_len=128):
        self.tokenizer = tokenizer
        if data_dir == 'SemEvalEc':
            self.data_path = f'data/{data_dir}/{data_type}.txt'
        elif data_dir == 'GoEmotions':
            self.data_path = f'data/{data_dir}/{data_type}.tsv'
        self.max_len = max_len
        self.data_dir = data_dir
        self.data_type = data_type
        self.target_length = max_len if data_dir == 'GoEmotions' else 128

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        label_idx = self.label_idx[index]

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "labels_idx": label_idx}

    def _build_examples(self):
        inputs, targets, labels_idx = get_transformed_data(self.data_path, self.data_dir, self.data_type)

        for i in range(len(inputs)):
            input = ' '.join(inputs[i])
            # print(input)
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding='max_length', truncation=True,
                return_tensors="pt",
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.target_length, padding='max_length', truncation=True,
                return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
            self.label_idx = labels_idx

