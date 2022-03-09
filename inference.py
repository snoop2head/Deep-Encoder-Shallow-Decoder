from config import CFG
from tokenizer import korean_tokenizer, english_tokenizer
from model import Seq2SeqTransformer
from trainer import generate_square_subsequent_mask
from dataset import TargetDataset, test_iter, test_dataloader

import torch

# function to generate output sequence using greedy algorithm
# Greedy Search: Greedy search simply selects the word with the highest probability as its next word
# https://huggingface.co/blog/how-to-generate
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(CFG.Device)
    src_mask = src_mask.to(CFG.Device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(CFG.Device)
    for i in range(max_len - 1):
        memory = memory.to(CFG.Device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            CFG.Device
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


if __name__ == "__main__":
    transformer = Seq2SeqTransformer(
        num_encoder_layers=CFG.NUM_ENCODER_LAYERS,
        num_decoder_layers=CFG.NUM_DECODER_LAYERS,
        emb_size=CFG.EMB_SIZE,
        nhead=CFG.NHEAD,
        src_vocab_size=CFG.SRC_VOCAB_SIZE,
        tgt_vocab_size=CFG.TGT_VOCAB_SIZE,
        dim_feedforward=CFG.FFN_HID_DIM,
        dropout=CFG.dropout_rate,
    )

    sample_id = 1000
    transformer.load_state_dict(torch.load(CFG.model_path))
    transformer.eval()
    src_input_id = TargetDataset(test_iter)[sample_id].view(-1, 1)
    num_tokens = src_input_id.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    print(src_input_id)
    print(num_tokens)
    print(src_mask)

    tgt_tokens = greedy_decode(
        transformer,
        src_input_id,
        src_mask,
        max_len=CFG.max_token_length,
        start_symbol=korean_tokenizer.cls_token_id,
    ).flatten()

    translated_sentence = " ".join(
        english_tokenizer.convert_ids_to_tokens(tgt_tokens.tolist())
    )

    print(korean_tokenizer.decode(test_iter[0][sample_id]))
    print(translated_sentence)
