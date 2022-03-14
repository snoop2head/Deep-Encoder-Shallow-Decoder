from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import numpy as np


class AverageMetrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_bleu(df_inferenced: pd.DataFrame):
    bleu_list = []
    for index, row in df_inferenced.iterrows():
        pred = row["pred"].strip()
        label = row["en"]
        pred_word_list = pred.split(" ")
        label_word_list = label.split(" ")

        # fill empty strings with list to match the length
        if len(pred_word_list) < len(label_word_list):
            pred_word_list = pred_word_list + [" "] * (
                len(label_word_list) - len(pred_word_list)
            )
        elif len(pred_word_list) > len(label_word_list):
            label_word_list = label_word_list + [" "] * (
                len(pred_word_list) - len(label_word_list)
            )
        bleu_list.append(
            corpus_bleu(pred_word_list, label_word_list, weights=(1.0, 0, 0, 0))
        )
    inference_size = len(bleu_list)
    print(f"{inference_size} are number of test samples for bleu score")
    return np.mean(bleu_list)
