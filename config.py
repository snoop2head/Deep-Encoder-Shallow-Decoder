from tokenizer import korean_tokenizer, english_tokenizer
from preprocess import MAX_TOKEN_LENGTH

import os
from datetime import datetime
import torch
from transformers import PretrainedConfig
from easydict import EasyDict

# login wandb and get today's date until hour and minute
today = datetime.now().strftime("%m%d_%H:%M")


class DeepShallowConfig(PretrainedConfig):
    """
    - Reference: MarianConfig | https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/marian/configuration_marian.py#L35
    - Reference: ResnetConfig | https://huggingface.co/docs/transformers/custom_models
    """

    model_type = "transformer"

    def __init__(
        self,
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        emb_size=512,
        attention_heads=8,
        ffn_hid_dim=2048,
        max_position_embeddings=64,
        encoder_layers=12,
        decoder_layers=1,
        is_encoder_decoder=True,
        activation_function="gelu",
        dropout=0.1,
        **kwargs,  # the __init__ of your PretrainedConfig must accept any kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.emb_size = emb_size
        self.attention_heads = attention_heads
        self.ffn_hid_dim = ffn_hid_dim
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.is_encoder_decoder = is_encoder_decoder
        self.activation_function = activation_function
        self.dropout = dropout

        super().__init__(
            **kwargs
        )  # those kwargs need to be passed to the superclass __init__.


if __name__ == "__main__":
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
    CFG.num_epochs = 8  # validation loss is increasing after 5 epochs
    CFG.max_token_length = MAX_TOKEN_LENGTH  # refer to EDA
    CFG.stopwords = []
    CFG.learning_rate = 5e-4
    CFG.weight_decay = 1e-2  # https://paperswithcode.com/method/weight-decay
    CFG.adam_betas = (0.9, 0.98)
    CFG.epsilon = 1e-9

    # Translation settings
    CFG.src_vocab_size = korean_tokenizer.vocab_size
    CFG.tgt_vocab_size = english_tokenizer.vocab_size
    CFG.src_language = "ko"
    CFG.tgt_language = "en"

    # root path
    CFG.ROOT_PATH = os.path.abspath(".")

    # set best model path
    CFG.model_path = "./best-eval-loss-model-9-epochs.pt"
