import os

from preprocess import fetch_raw_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

os.environ["TOKENIZERS_PARALLELISM"] = "true"
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

if CFG.TRAIN_TOKENIZER:
    train, _, _ = fetch_raw_dataset()
    """Train tokenizer from scratch and upload to huggingface hub"""
    korean_sentences = list(train["ko"].values)
    english_sentences = list(train["en"].values)

    # https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#build-a-tokenizer-from-scratch
    # train korean_tokenizer based on given dataset
    korean_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    korean_tokenizer.pre_tokenizer = Whitespace()
    korean_trainer = WordPieceTrainer(vocab_size=10000, special_tokens=SPECIAL_TOKENS)
    korean_tokenizer.train_from_iterator(korean_sentences, korean_trainer)
    korean_tokenizer.save("korean-WPE.json")

    # train english_tokenizer based on given dataset
    english_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    english_tokenizer.pre_tokenizer = Whitespace()
    english_trainer = WordPieceTrainer(vocab_size=10000, special_tokens=SPECIAL_TOKENS)
    english_tokenizer.train_from_iterator(english_sentences, english_trainer)
    english_tokenizer.save("english-WPE.json")

    UNK_IDX, BOS_IDX, EOS_IDX, PAD_IDX, MASK_IDX = (
        korean_tokenizer.unk_token_id,
        korean_tokenizer.bos_token_id,
        korean_tokenizer.eos_token_id,
        korean_tokenizer.pad_token_id,
        korean_tokenizer.mask_token_id,
    )

    # switch from Tokenizer class to PreTrainedTokenizerFast class to utilize more methods
    korean_tokenizer = PreTrainedTokenizerFast(tokenizer_file="korean-WPE.json")
    english_tokenizer = PreTrainedTokenizerFast(tokenizer_file="english-WPE.json")

    # Designate special token map for the PreTrainedTokenizerFast class (doesn't increase token number)
    korean_tokenizer.add_special_tokens(
        {
            "unk_token": "[UNK]",
            "bos_token": "[CLS]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "eos_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        }
    )

    english_tokenizer.add_special_tokens(
        {
            "unk_token": "[UNK]",
            "bos_token": "[CLS]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "eos_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        }
    )

    # save tokenizer on huggingface hub
    korean_tokenizer.push_to_hub("snoop2head/Deep-Shallow-Ko", use_temp_dir=True)
    english_tokenizer.push_to_hub("snoop2head/Deep-Shallow-En", use_temp_dir=True)
