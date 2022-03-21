import os
import torch
import transformers
from easydict import EasyDict
import yaml
from model import DeepShallowConfig, DeepShallowModel # fetch from model.py file

# load pretrained config and model path
run_path = "/home/ubuntu/_clones/Deep-Encoder-Shallow-Decoder/runs/En2Ko/run1"
config_path = os.path.join(run_path, "config.yaml")
model_path = os.path.join(run_path, "best-eval-loss-model-10-epochs.pt")

# Read config.yaml file
with open(config_path) as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])
    
    # check whether run config is loaded properly
    print(CFG)

# load config according to run config
config = DeepShallowConfig(
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
)

# setup model and load
transformer = DeepShallowModel(config=config)
transformer.load_state_dict(torch.load(model_path))

config.push_to_hub(f"{CFG.user_name}/Deep-Shallow-{CFG.src_language.capitalize()}2{CFG.tgt_language.capitalize()}", use_auth_token=True, use_temp_dir=True)
transformer.push_to_hub(f"{CFG.user_name}/Deep-Shallow-{CFG.src_language.capitalize()}2{CFG.tgt_language.capitalize()}", use_auth_token=True, use_temp_dir=True)