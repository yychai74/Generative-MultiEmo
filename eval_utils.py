import pandas as pd
from sklearn.metrics import classification_report, jaccard_score, hamming_loss
import numpy as np


def jaccard(y_gold, y_pred):
    y_gold = np.asarray(y_gold).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    assert len(y_gold) == len(y_pred)
    tmp_sum = 0
    num_sample = len(y_gold)
    for i in range(num_sample):
        if y_pred[i][-1] == 1 and y_pred[i].sum() > 1 and y_gold[i][-1] != 1:
            # if y_gold[i][-1] != 1:
            y_gold_i = y_gold[i][:-1]
            y_pred_i = np.zeros((len(y_gold_i),), dtype=np.int)
        else:
            y_gold_i = y_gold[i][:-1]
            y_pred_i = y_pred[i][:-1]
        if sum(np.logical_or(y_gold_i, y_pred_i)) == 0:
            tmp_sum += 1
        else:
            tmp_sum += sum(y_gold_i & y_pred_i) / sum(np.logical_or(y_gold_i, y_pred_i))
    return tmp_sum / num_sample


def compute_classification_eval_metrics(metrics, predictions, labels, dataset):
    report = classification_report(labels, predictions, digits=4, zero_division=0, output_dict=True)
    metrics['classification'] = report

    if dataset == 'GoEmotions':
        metrics['jaccard_score'] = jaccard_score(labels, predictions, average='samples')

    return metrics


def extract_SemEvalEc(seq_list, emotion_dict):
    extractions, neutral_extractions = [], []
    for seqs in seq_list:
        for seq in seqs:
            num1 = np.zeros((11,), dtype=int).tolist()
            num2 = np.zeros((len(emotion_dict),), dtype=int).tolist()
            # if "neutral" in seq:
            #     extractions.append(num)
            #     break
            targets = seq.split(' [SSEP] ')
            # targets = seq.split()
            for target in targets:
                words = target.split()
                try:
                    emo = words[2].strip(".")
                    # emo = target
                except IndexError:
                    print(target)
                    emo = ''

                if emo != '' and emo in emotion_dict.keys():
                    if emo == "neutral":
                        idx = emotion_dict[emo]
                        num2[idx] = 1
                    else:
                        idx = emotion_dict[emo]
                        num1[idx] = 1
                        num2[idx] = 1
                else:
                    print(targets)
                    num1 = np.zeros((11,), dtype=int).tolist()
                    num2 = np.zeros((len(emotion_dict),), dtype=int).tolist()
                    break
            if num2[11] == 1:
                num1 = np.zeros((11,), dtype=int).tolist()
            extractions.append(num1)
            neutral_extractions.append(num2)

    return extractions, neutral_extractions


def evaluate_SemEvalEc(outputs, targets, dataset):
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness",
                "surprise", "trust", "neutral"]
    emotion_dict = {emo: idx for idx, emo in enumerate(emotions)}
    pred_pt, jaccord_pred = extract_SemEvalEc(outputs, emotion_dict)
    gold_pt, jaccord_gold = extract_SemEvalEc(targets, emotion_dict)

    pred_pt = np.array(pred_pt)
    gold_pt = np.array(gold_pt)
    metrics = {}
    metrics = compute_classification_eval_metrics(metrics, pred_pt, gold_pt, dataset=dataset)
    metrics['jaccard_score'] = jaccard(jaccord_gold, jaccord_pred)
    # print(metrics)

    return metrics


def extract_ISEAR(seq_list, emotion_dict):
    extractions = []
    for seq in seq_list:
        target = seq.split()
        target = target[6]
        # target = ''.join(target)
        # temp = target.split(',')
        # print(temp)
        # for e in temp:
        if target in emotion_dict:
            extractions.append(emotion_dict[target])
        else:
            extractions.append(0)

    return extractions


def evaluate_ISEAR(outputs, targets):
    outputs = sum(outputs, [])
    targets = sum(targets, [])
    emotions = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']
    emotion_dict = {emo: idx for idx, emo in enumerate(emotions)}
    pred = extract_ISEAR(outputs, emotion_dict)
    gold = extract_ISEAR(targets, emotion_dict)

    results = classification_report(gold, pred)
    print(results)
    return results


def evaluate_GoEmotion(outputs, targets, dataset):
    emotion_label = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                     'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                     'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                     'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    emotion_dict = {emo: idx for idx, emo in enumerate(emotion_label)}

    pred = extract_GoEmotion(outputs, emotion_dict)
    gold = extract_GoEmotion(targets, emotion_dict)

    metrics = {}
    metrics = compute_classification_eval_metrics(metrics, pred, gold, dataset=dataset)
    print(metrics)
    # print(metrics.keys())
    return metrics


def extract_GoEmotion(seq_list, emotion_dict):
    extractions = []
    for seq in seq_list:
        for target in seq:
            num = np.zeros((len(emotion_dict),), dtype=int).tolist()
            # target = target.strip('.')
            targets = target.split(' [SSEP] ')
            # targets = target.split()
            # targets = target.split(', ')
            for sen in targets:
                words = sen.split()
                try:
                    emo = words[2].strip('.')
                    # emo = sen
                except IndexError:
                    print(sen)
                    emo = ''
            # print(temp)
            # for e in temp:
            # for emo in targets:
                if emo != '' and emo in emotion_dict.keys():
                    idx = emotion_dict[emo]
                    num[idx] = 1
                else:
                    print(targets)
                    num = np.zeros((len(emotion_dict),), dtype=np.int).tolist()
                    break
            extractions.append(num)

    return extractions


def evaluate(outputs, targets, dataset):
    if dataset == 'SemEvalEc':
        results = evaluate_SemEvalEc(outputs, targets, dataset)
    elif dataset == 'GoEmotions':
        results = evaluate_GoEmotion(outputs, targets, dataset)
    return results


if __name__ == '__main__':
    df = pd.read_csv('sem42_out.csv')
    pred = df['0'].values.tolist()
    target = df['1'].values.tolist()

    l1, l2, l3, l4 = [], [], [], []
    tl1, tl2, tl3, tl4 = [], [], [], []

    l5, tl5 = [], []

    for i in range(len(target)):
        if len(target[i].split(' [SSEP] ')) == 1:
            tl1.append(target[i])
            l1.append(pred[i])

        elif len(target[i].split(' [SSEP] ')) == 2:
            tl2.append(target[i])
            l2.append(pred[i])

        elif len(target[i].split(' [SSEP] ')) == 3:
            tl3.append(target[i])
            l3.append(pred[i])

        # elif len(target[i].split(' [SSEP] ')) == 4:
        #     tl4.append(target[i])
        #     l4.append(pred[i])

        else:
            tl5.append(target[i])
            l5.append(pred[i])

    # aa = evaluate_GoEmotion(outputs=[l1], targets=[tl1], dataset='GoEmotions')
    aa = evaluate_SemEvalEc(outputs=[l5], targets=[tl5], dataset='SemEvalEc')
    # print(len(tl4))
