import numpy as np
import pandas as pd
import config
from sklearn.metrics import accuracy_score#, precision_score, recall_score, f1_score, confusion_matrix

original_table = pd.read_csv("output/gold_file.tsv", delimiter='\t', names=["text", "paragraph", "pos", "tag"], header=None)
predict_table = pd.read_csv("output/pred_output.tsv", delimiter='\t', names=["text", "paragraph", "pos", "tag"], header=None)

def get_tags(datapath):
    df = pd.read_csv(datapath, delimiter='\t', names=["text", "paragraph", "pos", "tag"], header=None)
    tags = df["tag"]
    return np.array(tags)

def precision_score(y_test, pred):
    M = 0
    C = 0
    cnt = 0
    for index in range(len(y_test)):
        if pred[index] == 'B':
            M += 1
        if y_test[index] == 'O':
            if cnt != 0:
                C += 1
            cnt = 0
        if y_test[index] == 'B' and cnt == 0:
            if pred[index] == 'B':
                cnt += 1
        elif y_test[index] == 'B' and cnt != 0:
            C += 1
            if pred[index] == 'B':
                cnt = 1
        elif y_test[index] == 'I':
            if pred[index] == y_test[index]:
                cnt += 1
            else:
                cnt = 0 
    if y_test[-1] == 'B' and pred[-1] == 'B':
        C += 1
    print(f"C: {C} M: {M}")
    return C / M

def recall_score(y_test, pred):
    N = 0
    C = 0
    cnt = 0
    for index in range(len(y_test)):
        if y_test[index] == 'B':
            N += 1
        if y_test[index] == 'O':
            if cnt != 0:
                C += 1
            cnt = 0
        if y_test[index] == 'B' and cnt == 0:
            if pred[index] == 'B':
                cnt += 1
        elif y_test[index] == 'B' and cnt != 0:
            C += 1
            if pred[index] == 'B':
                cnt = 1
        elif y_test[index] == 'I':
            if pred[index] == y_test[index]:
                cnt += 1
            else:
                cnt = 0 
    if y_test[-1] == 'B' and pred[-1] == 'B':
        C += 1
    print(f"C: {C} N: {N}")
    return C / N

def f1_score(p, r):
    return (2 * p * r) / (p + r)


def get_metrics_score(y_test, pred):
    average = 'micro'
    own_precision = precision_score(y_test, pred)
    own_recall = recall_score(y_test, pred)
    own_f1 = f1_score(own_precision, own_recall)
    print(f"Model:{config.BASE_MODEL_PATH} MAX LENGTH: {config.MAX_LEN}")
    print(f"{average} precision:{own_precision}\n{average} recall:{own_recall}\n{average} f1:{own_f1}")

if __name__ == "__main__":
    y = get_tags("output/gold_file.tsv")
    pred = get_tags("output/pred_output.tsv")
    get_metrics_score(y, pred)
