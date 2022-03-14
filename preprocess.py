import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, DatasetDict, load_dataset
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

korean_tokenizer = PreTrainedTokenizerFast.from_pretrained("snoop2head/Deep-Shallow-Ko")
english_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "snoop2head/Deep-Shallow-En"
)


def preprocess(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    1) Remove characters that are irrelevant to both source and target language
    2) Notate length of each sentence pair
    """

    df_return = df_input.copy()

    # removing characters except for english, korean, numerics and whitespaces
    df_return["ko"] = df_return["ko"].str.replace("[^A-Z a-z 0-9 가-힣]", "")
    df_return["en"] = df_return["en"].str.replace("[^A-Z a-z 0-9]", "")

    # apply tokenizer for ko
    df_return["ko_len"] = df_return["ko"].apply(
        lambda x: len(korean_tokenizer.encode(x)) + 2
    )
    df_return["en_len"] = df_return["en"].apply(
        lambda x: len(english_tokenizer.encode(x)) + 2
    )

    print(df_return.shape)
    print(df_return.head(3))
    print(df_return.tail(3))
    return df_return


def cleanse_broken(df_input: pd.DataFrame) -> pd.DataFrame:
    """removing rows with empty values"""
    df_input.replace("", np.nan, inplace=True)
    df_input.replace(" ", np.nan, inplace=True)
    df_input.dropna(inplace=True)
    return df_input


def fetch_raw_dataset():
    """preprocess and upload to huggingface hub"""
    # https://huggingface.co/datasets/PoolC/AIHUB-parallel-ko-en
    # fetch original parallel corpus
    dataset = load_dataset("PoolC/AIHUB-parallel-ko-en", use_auth_token=True)
    df_parallel = pd.concat(
        [dataset["train"].to_pandas(), dataset["valid"].to_pandas()]
    )

    # split train, valid and test set
    df_valid_and_test = df_parallel.sample(frac=0.04, random_state=42)
    train = df_parallel.drop(df_valid_and_test.index)
    valid = df_valid_and_test.sample(frac=0.5, random_state=42)
    test = df_valid_and_test.drop(valid.index)
    return train, valid, test


def make_dataset():
    train, valid, test = fetch_raw_dataset()
    # apply preprocessing function
    df_train = preprocess(train)
    df_valid = preprocess(valid)
    df_test = preprocess(test)

    translation_train_dataset = Dataset.from_pandas(df_train)  # 4.90 M pairs
    translation_valid_dataset = Dataset.from_pandas(df_valid)  # 0.10 M pairs
    translation_test_dataset = Dataset.from_pandas(df_test)  # 0.10 M pairs

    # https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090
    translation_train_valid_dataset = DatasetDict(
        {
            "train": translation_train_dataset,
            "valid": translation_valid_dataset,
            "test": translation_test_dataset,
        }
    )

    translation_train_valid_dataset.push_to_hub("PoolC/AIHUB-parallel-ko-en-cleansed")
    return


def use_dataset():
    """download from huggingface hub and make dataframes"""
    dataset = load_dataset("PoolC/AIHUB-parallel-ko-en-cleansed", use_auth_token=True)

    df_train = dataset["train"].to_pandas()
    df_valid = dataset["valid"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # set up maximum token length according to dataset percentile
    RATIO = 0.98  # using 98 percentile of total parallel corpus
    MAX_TOKEN_LENGTH = int(
        max(
            np.quantile(df_train["ko_len"].values, RATIO),
            np.quantile(df_train["en_len"].values, RATIO),
        )
    )
    # make max_token_length even number
    if MAX_TOKEN_LENGTH % 2 != 0:
        MAX_TOKEN_LENGTH += 1

    df_train = df_train[
        (df_train["ko_len"] <= MAX_TOKEN_LENGTH)
        & (df_train["en_len"] <= MAX_TOKEN_LENGTH)
    ]
    df_valid = df_valid[
        (df_valid["ko_len"] <= MAX_TOKEN_LENGTH)
        & (df_valid["en_len"] <= MAX_TOKEN_LENGTH)
    ]
    df_test = df_test[
        (df_test["ko_len"] <= MAX_TOKEN_LENGTH)
        & (df_test["en_len"] <= MAX_TOKEN_LENGTH)
    ]

    # remove rows with empty values
    df_train = cleanse_broken(df_train)
    df_valid = cleanse_broken(df_valid)
    df_test = cleanse_broken(df_test)
    return df_train, df_valid, df_test
