# Prompt-Based Generative Multi-label Emotion Prediction with Label Contrastive Learning
This is the source code for NLPCC 2022 paper: Prompt-Based Generative Multi-label Emotion Prediction with Label Contrastive Learning
## 1. Environments

```
- python (3.9.5)
- cuda (11.4)
```

## 2. Dependencies

```
- torch (1.10.0)
- pytorch-lightning (1.5.1)
- transformers (4.12.3)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- tensorboard (2.7.0)
- ekphrasis (0.5.1)
```

## 3. Dataset

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [SemEval 2018 E-c](https://competitions.codalab.org/competitions/17751#learn_the_details)

For SemEval 2018 dataset, we utilized a tweet pre-process tool named [ekphrasis](https://github.com/cbaziotis/ekphrasis) to convert words to lower case, normalise user mentions, urls and repeated-characters. We provided the processed data additionally.

## 4. Training

For GoEmotions dataset, run:

```bash
>> python main.py --CLP
```

For SemEval 2018 dataset, run:

```bash
>> python main.py --dataset 'SemEvalEc' --num_beams 2 --CLP
```

For ablation study, run:

```bash
>> python main.py
>> python main.py --dataset 'SemEvalEc' --num_beams 2
```

## 5. Credits

We thank for the following open sourced projects and their contributions:
- [https://github.com/IsakZhang/ABSA-QUAD]
- [https://github.com/HobbitLong/SupContrast]
- [https://github.com/Lightning-AI/lightning]
