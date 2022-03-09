from config import CFG
from preprocess import df_train, df_valid, df_test
from tokenizer import korean_tokenizer, english_tokenizer

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(
        self,
        df_input: pd.DataFrame,
        source_language: str = CFG.src_language,
        target_language: str = CFG.tgt_language,
        max_length: int = CFG.max_token_length,
    ):

        self.df_input = df_input
        self.max_length = max_length
        self.source_language = source_language
        self.target_language = target_language

    def __len__(self):
        return len(self.df_input)

    def __getitem__(self, idx):
        # get sentences
        source_sent = self.df_input.iloc[idx][self.source_language]
        target_sent = self.df_input.iloc[idx][self.target_language]

        # tokenize
        source_input_ids = korean_tokenizer(
            korean_tokenizer.cls_token + source_sent + korean_tokenizer.eos_token,
            add_special_tokens=False,
            padding="max_length",
            max_length=CFG.max_token_length,
            truncation=True,
            return_token_type_ids=False,
        )["input_ids"]

        target_input_ids = english_tokenizer(
            english_tokenizer.cls_token + target_sent + english_tokenizer.eos_token,
            add_special_tokens=False,
            padding="max_length",
            max_length=CFG.max_token_length,
            truncation=True,
            return_token_type_ids=False,
        )["input_ids"]

        return tuple(
            (
                source_input_ids,
                target_input_ids,
            )
        )


class TargetDataset(Dataset):
    def __init__(
        self,
        list_data: list,
        source_language=CFG.src_language,
        max_length=CFG.max_token_length,
    ):

        self.list_data = list_data
        self.max_length = max_length
        self.source_language = source_language

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        # get sentences
        source_sent = self.list_data[idx]

        source_input_ids = korean_tokenizer(
            korean_tokenizer.cls_token + source_sent + korean_tokenizer.eos_token,
            add_special_tokens=False,
            padding="max_length",
            max_length=CFG.max_token_length,
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]

        return source_input_ids


# function to collate data samples into batch tesors
# TODO: Make Random Masked Collator
def custom_collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = torch.transpose(torch.tensor(src_batch), 0, 1)
    tgt_batch = torch.transpose(torch.tensor(tgt_batch), 0, 1)
    return src_batch, tgt_batch


# function to collate data samples into batch tesors
def custom_collate_inference_fn(batch):
    print(batch)
    src_batch = []
    for src_sample in batch:
        src_batch.append(src_sample)
    print(torch.tensor(src_batch).size())
    src_batch = torch.transpose(torch.tensor(src_batch), 0, 1)
    return src_batch


if __name__ == "__main__":
    train_iter = TranslationDataset(df_train)
    train_dataloader = DataLoader(
        train_iter, batch_size=CFG.train_batch_size, collate_fn=custom_collate_fn
    )

    val_iter = TranslationDataset(df_valid)
    val_dataloader = DataLoader(
        val_iter, batch_size=CFG.valid_batch_size, collate_fn=custom_collate_fn
    )

    test_iter = TargetDataset(df_test)
    test_dataloader = DataLoader(
        test_iter,
        batch_size=CFG.test_batch_size,
        collate_fn=custom_collate_inference_fn,
    )
