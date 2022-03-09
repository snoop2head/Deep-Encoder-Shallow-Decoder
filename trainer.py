from config import CFG
from tokenizer import korean_tokenizer, english_tokenizer
from model import PositionalEncoding, TokenEmbedding, Seq2SeqTransformer
from dataset import train_dataloader, val_dataloader
from metrics import AverageMetrics

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.notebook import tqdm
from transformers import get_cosine_schedule_with_warmup


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=CFG.DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=CFG.DEVICE).type(
        torch.bool
    )
    # sequence length x sequence length matrix
    # intialized all to False -> just to match target mask which is to prevent decoder to cheat in autoregressive training

    src_padding_mask = (src == korean_tokenizer.pad_token_id).transpose(0, 1)
    tgt_padding_mask = (tgt == english_tokenizer.pad_token_id).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == "__main__":
    # prevent possible OOM error
    try:
        if transformer:
            transformer.cpu()
            del transformer
            torch.cuda.empty_cache()
    except:
        pass

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

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(CFG.DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=english_tokenizer.pad_token_id)
    optimizer = AdamW(
        transformer.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        betas=CFG.adam_betas,
        eps=CFG.epsilon,
    )

    train_loss = AverageMetrics()
    eval_loss = AverageMetrics()

    CFG.logging_steps = (
        len(train_dataloader) // 4
    )  # set logging steps according to the length of train_loader
    CFG.warmup_steps = CFG.logging_steps  # warmup steps as 1/3 of first epoch

    # https://huggingface.co/transformers/main_classes/optimizer_schedules.html
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CFG.warmup_steps,
        num_training_steps=len(train_dataloader) * CFG.num_epochs,
    )

    best_eval_loss = 5.0

    # Train and Validation iteration
    for epoch in range(1, CFG.num_epochs + 1):
        for num_steps, (src, tgt) in enumerate(tqdm(train_dataloader)):
            src = src.to(CFG.DEVICE)
            tgt = tgt.to(CFG.DEVICE)

            transformer.train()

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )

            logits = transformer(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.update(loss.item(), len(src[0]))

            if num_steps % CFG.logging_steps == 0 and num_steps != 0:  # batch
                print(
                    "Epoch: {}/{}".format(epoch, CFG.num_epochs),
                    "Step: {}".format(num_steps),
                    "Train Loss: {:.4f}".format(train_loss.avg),
                )
                transformer.eval()

                for src, tgt in tqdm(val_dataloader):
                    src = src.to(CFG.DEVICE)
                    tgt = tgt.to(CFG.DEVICE)

                    tgt_input = tgt[:-1, :]

                    (
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                    ) = create_mask(src, tgt_input)
                    logits = transformer(
                        src,
                        tgt_input,
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                        src_padding_mask,
                    )
                    tgt_out = tgt[1:, :]
                    dev_loss = loss_fn(
                        logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                    )
                    eval_loss.update(dev_loss.item(), len(src[0]))

                print(
                    "Epoch: {}/{}".format(epoch, CFG.num_epochs),
                    "Step: {}".format(num_steps),
                    "Dev Loss: {:.4f}".format(eval_loss.avg),
                )

                # wandb.log(
                #     {
                #         'train/loss':train_loss.avg,
                #         'train/learning_rate':optimizer.param_groups[0]['lr'],
                #         'eval/loss':eval_loss.avg,
                #         'Step':num_steps
                #     }
                # )

                if best_eval_loss > eval_loss.avg:
                    best_eval_loss = eval_loss.avg
                    torch.save(
                        transformer.state_dict(),
                        f"./best-eval-loss-model-{epoch}-epochs.pt",
                    )
                    print(
                        "Saved model with lowest validation loss: {:.4f}".format(
                            best_eval_loss
                        )
                    )
                    # wandb.log({'best_eval_loss':best_eval_loss})

                # reset metrics
                eval_loss.reset()
                train_loss.reset()
