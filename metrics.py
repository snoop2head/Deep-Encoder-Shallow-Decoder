from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pandas as pd
import numpy as np
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])


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


def calculate_bleu(df_inferenced: pd.DataFrame, bleu_type: str = "corpus"):
    bleu_list = []
    for index, row in df_inferenced.iterrows():
        pred = row["prediction"].strip()
        label = row["label"]
        if bleu_type == "corpus":
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
        elif bleu_type == "sentence":
            bleu_list.append(
                sentence_bleu([label.split()], pred.split(), weights=(1.0, 0, 0, 0))
            )
    inference_size = len(bleu_list)
    print(f"{inference_size} are number of test samples for {bleu_type} bleu score")

    bleu_score = np.mean(bleu_list)
    print(f"{bleu_type.upper()} BLEU score: {bleu_score}")
    return


if __name__ == "__main__":
    df_inferenced = pd.read_csv(
        f"./result/inference_{CFG.num_inference_sample}_samples_{CFG.src_language}_to_{CFG.tgt_language}.csv",
    )
    calculate_bleu(df_inferenced, bleu_type="corpus")
