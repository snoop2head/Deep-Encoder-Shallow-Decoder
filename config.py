import os
import wandb
from datetime import datetime
import torch
from tokenizer import korean_tokenizer, english_tokenizer
from preprocess import MAX_TOKEN_LENGTH
from easydict import EasyDict

# login wandb and get today's date until hour and minute
today = datetime.now().strftime("%m%d_%H:%M")

# Debug set to true in order to debug high-layer code.
# CFG Configuration
CFG = EasyDict()  # wandb.config provides functionality of easydict.EasyDict
CFG.DEBUG = False
CFG.num_workers = 2
CFG.train_batch_size = 256
CFG.valid_batch_size = 128
CFG.DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and CFG.DEBUG == False else "cpu"
)


# Train configuration
CFG.TRAIN_TOKENIZER = False
CFG.PREPROCESS = False
CFG.user_name = "snoop2head"
CFG.num_epochs = 10  # validation loss is increasing after 5 epochs
CFG.max_token_length = MAX_TOKEN_LENGTH  # refer to tokenizer file
CFG.stopwords = []
CFG.learning_rate = 5e-4
CFG.weight_decay = 1e-2  # https://paperswithcode.com/method/weight-decay

# Translation settings
CFG.src_vocab_size = korean_tokenizer.vocab_size
CFG.tgt_vocab_size = english_tokenizer.vocab_size
CFG.src_language = "ko"
CFG.tgt_language = "en"

# root path
ROOT_PATH = os.path.abspath(".")

# Transformer Configs
CFG.EMB_SIZE = 512
CFG.NHEAD = 8
CFG.FFN_HID_DIM = 2048
CFG.BATCH_SIZE = CFG.train_batch_size
CFG.NUM_ENCODER_LAYERS = 12
CFG.NUM_DECODER_LAYERS = 1
CFG.DROPOUT_RATE = 0.1
CFG.ADAM_BETAS = (0.9, 0.98)
CFG.EPSILON = 1e-9

# set best model path
CFG.model_path = "./best-eval-loss-model-9-epochs.pt"
