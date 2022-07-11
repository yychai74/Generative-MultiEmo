import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt


def get_co_existence(dataset):
    if dataset == 'GoEmotions':
        df = pd.read_csv(f'data/{dataset}/train.tsv', sep="\t", names=['sentence', 'label', 'null'])
        label_size = 28
        labels_list = df.iloc[:, 1].values
    else:
        df = pd.read_csv(f'data/{dataset}/train.txt', sep="	")
        label_size = 11

    relation_matrix = np.zeros((label_size, label_size), dtype=int)

    def permutation(tup, k):
        lst = list(tup)
        result = []
        tmp = [0] * k

        def next_num(a, ni=0):
            if ni == k:
                result.append(copy.copy(tmp))
                return
            for lj in a:
                tmp[ni] = lj
                b = a[:]
                b.pop(a.index(lj))
                next_num(b, ni + 1)

        c = lst[:]
        next_num(c, 0)
        return result

    if dataset == 'GoEmotions':
        for label in labels_list:
            if isinstance(eval(label), tuple):
                all_relation = permutation(eval(label), 2)

                for relation in all_relation:
                    relation_matrix[relation[0]][relation[1]] += 1

    elif dataset == 'SemEvalEc':
        label_array = df.iloc[:, 2:].values.tolist()
        for temp in label_array:
            indexes = [inx for inx, num in enumerate(temp) if num == 1]
            if len(indexes) > 1:
                all_relation = permutation(indexes, 2)
                for relation in all_relation:
                    relation_matrix[relation[0]][relation[1]] += 1

    return relation_matrix

