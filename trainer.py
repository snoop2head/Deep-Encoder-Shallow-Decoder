from preprocess import use_dataset
from config import DeepShallowConfig
from model import DeepShallowModel
from metrics import AverageMetrics
from dataset import TranslationDataset, custom_collate_fn

import torch

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and CFG.DEBUG == False else "cpu"
)

korean_tokenizer = PreTrainedTokenizerFast.from_pretrained("snoop2head/Deep-Shallow-Ko")
english_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "snoop2head/Deep-Shallow-En"
)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
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
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
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

    df_train, df_valid, _ = use_dataset()
    train_iter = TranslationDataset(df_train)
    train_dataloader = DataLoader(
        train_iter, batch_size=CFG.train_batch_size, collate_fn=custom_collate_fn
    )

    val_iter = TranslationDataset(df_valid)
    val_dataloader = DataLoader(
        val_iter, batch_size=CFG.valid_batch_size, collate_fn=custom_collate_fn
    )

    transformer = DeepShallowModel(config=DeepShallowConfig())

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=english_tokenizer.pad_token_id)
    optimizer = AdamW(
        transformer.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        betas=(CFG.adam_beta_1, CFG.adam_beta_2),
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
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

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
                    src = src.to(DEVICE)
                    tgt = tgt.to(DEVICE)

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
