import math
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

korean_tokenizer = PreTrainedTokenizerFast.from_pretrained("snoop2head/Deep-Shallow-Ko")
english_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "snoop2head/Deep-Shallow-En"
)


class DeepShallowConfig(PretrainedConfig):
    """
    - Reference: MarianConfig | https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/marian/configuration_marian.py#L35
    - Reference: ResnetConfig | https://huggingface.co/docs/transformers/custom_models
    """

    model_type = "transformer"

    def __init__(
        self,
        emb_size=CFG.emb_size,
        attention_heads=CFG.attention_heads,
        ffn_hid_dim=CFG.ffn_hid_dim,
        encoder_layers=CFG.encoder_layers,
        decoder_layers=CFG.decoder_layers,
        is_encoder_decoder=CFG.is_encoder_decoder,
        activation_function=CFG.activation_function,
        dropout=CFG.dropout,
        src_vocab_size=CFG.src_vocab_size,
        tgt_vocab_size=CFG.tgt_vocab_size,
        max_position_embeddings=CFG.max_token_length,
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


class PositionalEncoding(nn.Module):
    """Reference
    MarianSinusoidalPositionalEmbedding:https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/src/transformers/models/marian/modeling_marian.py#L109
    """

    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int,
    ):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(
            -torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )  # embedding_size = dimension of model
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)  # sine function
        pos_embedding[:, 1::2] = torch.cos(pos * den)  # cosine function
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    """Reference
    MarianEncoderLayer: https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/src/transformers/models/marian/modeling_marian.py#L289
    """

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class DeepShallowModel(PreTrainedModel):
    """Reference
    MarianModel: https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/marian/modeling_marian.py#L1081
    MarianMTModel: https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/marian/modeling_marian.py#L1210
    ResnetModel: https://huggingface.co/docs/transformers/custom_models
    """

    def __init__(self, config: DeepShallowConfig):
        super().__init__(config)

        if not config.activation_function.lower() in ["relu", "gelu"]:
            raise ValueError("activation_function must be in ['relu', 'gelu']")
        elif config.activation_function == "relu":
            self.activation_function = nn.ReLU()  # original paper implementation
        elif config.activation_function == "gelu":
            self.activation_function = nn.GELU()  # deviation from paper implementation

        self.transformer = nn.Transformer(
            d_model=config.emb_size,
            nhead=config.attention_heads,
            num_encoder_layers=config.encoder_layers,
            num_decoder_layers=config.decoder_layers,
            dim_feedforward=config.ffn_hid_dim,
            dropout=config.dropout,
            activation=self.activation_function,
        )

        self.generator = nn.Linear(config.emb_size, config.tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(config.src_vocab_size, config.emb_size)
        self.tgt_tok_emb = TokenEmbedding(config.tgt_vocab_size, config.emb_size)
        self.positional_encoding = PositionalEncoding(
            config.emb_size,
            dropout=config.dropout,
            maxlen=config.max_position_embeddings,
        )

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,  # tgt_mask: masking for limiting the autoregressive prediction
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)
