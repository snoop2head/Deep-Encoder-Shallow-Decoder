from preprocess import use_dataset
from model import DeepShallowConfig, DeepShallowModel
from trainer import generate_square_subsequent_mask
from dataset import TargetDataset, custom_collate_inference_fn

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

import torch
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and CFG.DEBUG == False else "cpu"
)

korean_tokenizer = PreTrainedTokenizerFast.from_pretrained("snoop2head/Deep-Shallow-Ko")
english_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "snoop2head/Deep-Shallow-En"
)

# function to generate output sequence using greedy algorithm
# Greedy Search: Greedy search simply selects the word with the highest probability as its next word
# https://huggingface.co/blog/how-to-generate
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == korean_tokenizer.eos_token_id:
            break
    return ys


def inference(transformer, src_input_id):
    # find the index where first pad_token_id appears in src_input_id
    num_tokens = src_input_id.shape[0]
    pad_token_id = korean_tokenizer.pad_token_id

    # get index of the last True
    src_mask = (src_input_id != pad_token_id).squeeze()
    num_tokens_without_pad = src_mask.sum().item() - 1

    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    transformer.eval()
    tgt_tokens = greedy_decode(
        transformer,
        src_input_id,
        src_mask,
        max_len=num_tokens_without_pad,
        start_symbol=korean_tokenizer.cls_token_id,
    ).flatten()
    target_tokenizer = (
        english_tokenizer if CFG.tgt_language == "en" else korean_tokenizer
    )
    result = " ".join(target_tokenizer.convert_ids_to_tokens(tgt_tokens))
    result = result.replace(" ##", "")
    result = result.replace("[UNK]", "")
    result = result.replace("[CLS]", "")
    return result


if __name__ == "__main__":
    _, _, df_test = use_dataset()
    print(df_test.head(2))

    list_inferenced = []
    slice_index = CFG.num_inference_sample - 1

    list_test = df_test[CFG.src_language].tolist()[:slice_index]
    list_answer = df_test[CFG.tgt_language].tolist()[:slice_index]

    test_dataset = TargetDataset(list_test)

    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=CFG.valid_batch_size,
    #     collate_fn=custom_collate_inference_fn,
    # )

    config = DeepShallowConfig.from_pretrained(CFG.load_model_name)
    transformer = DeepShallowModel.from_pretrained(CFG.load_model_name, config=config)
    transformer = transformer.to(DEVICE)

    for index_num, item in enumerate(tqdm(test_dataset)):
        src_input_id = item.view(-1, 1)
        result = inference(transformer, src_input_id)
        list_inferenced.append(result)
        if index_num == slice_index:
            break

    df_result = pd.DataFrame(
        {
            "input": list_test,
            "prediction": list_inferenced,
            "label": list_answer,
        }
    )
    df_result.to_csv(
        f"./result/inference_{CFG.num_inference_sample}_samples_{CFG.src_language}_to_{CFG.tgt_language}.csv",
        index=False,
    )
