from transformers import PretrainedConfig
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])


class DeepShallowConfig(PretrainedConfig):
    """
    - Reference: MarianConfig | https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/marian/configuration_marian.py#L35
    - Reference: ResnetConfig | https://huggingface.co/docs/transformers/custom_models
    """

    model_type = "transformer"

    def __init__(
        self,
        emb_size=512,
        attention_heads=8,
        ffn_hid_dim=2048,
        encoder_layers=12,
        decoder_layers=1,
        is_encoder_decoder=True,
        activation_function="gelu",
        dropout=0.1,
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
