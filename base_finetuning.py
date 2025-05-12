import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

# Set CUDA environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,4'

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from llama.generation import Llama

# Prepare Alpaca dataset

def get_alpaca_dataset(data_split, tokenizer, max_length=512):
    class AlpacaDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    samples = []
    for example in data_split:
        instr = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        out = example.get("output", "").strip()
        if inp:
            prompt = f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instr}\n### Response:\n"
        ptokens = tokenizer.encode(prompt, bos=True, eos=False)
        rtokens = tokenizer.encode(out, bos=False, eos=True)
        toks = (ptokens + rtokens)[:max_length]
        prompt_len = min(len(ptokens), len(toks))
        samples.append((toks, prompt_len))
    return AlpacaDataset(samples)

# Training loop with gradient accumulation

def train(model, loader, optimizer, scheduler, device, num_epochs=3, grad_accum_steps=4):
    model.train()
    for epoch in range(num_epochs):
        start = time.time()
        print(f"Epoch {epoch+1} start: {time.ctime(start)}")
        total_loss = 0.0
        optimizer.zero_grad()
        for step, (input_ids, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(tokens=input_ids, start_pos=0)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            loss = loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        end = time.time()
        print(f"Epoch {epoch+1} end: {time.ctime(end)}, duration: {end-start:.2f}s, avg loss: {avg_loss:.4f}")

# Main entry point

def main():
    ckpt_dir = "/home/mangyeong/.llama/checkpoints/Llama3.2-1B"
    tok_path = "/home/mangyeong/.llama/checkpoints/Llama3.2-1B/tokenizer.model"
    max_seq_len = 512
    batch_size = 16
    grad_accum_steps = 4
    num_epochs = 3
    lr = 3e-5
    wd = 0.0

    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tok_path,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size * grad_accum_steps
    )
    # Reset default tensor type to CPU to avoid CUDA default for new tensors (for DataLoader internals)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float32)

    model = llama.model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0,1])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    tokenizer = llama.tokenizer

    ds_raw = load_dataset("tatsu-lab/alpaca", split="train")
    ds = get_alpaca_dataset(ds_raw, tokenizer, max_length=max_seq_len)

    pad = tokenizer.pad_id
    def collate_fn(batch):
        toks, pr_lens = zip(*batch)
        max_len = max(len(x) for x in toks)
        input_ids = torch.full((len(toks), max_len), pad, dtype=torch.long)
        labels = torch.full((len(toks), max_len), -100, dtype=torch.long)
        for i, (t, pl) in enumerate(zip(toks, pr_lens)):
            L = len(t)
            input_ids[i, :L] = torch.tensor(t, dtype=torch.long)
            labels[i, pl:L] = input_ids[i, pl:L]
        return input_ids, labels

    generator = torch.Generator(device='cpu')
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4
    )

    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total = (len(loader) // grad_accum_steps) * num_epochs
    warm = int(total * 0.03)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warm, num_training_steps=total)

    device = next(model.parameters()).device
    train(model, loader, opt, sched, device, num_epochs=num_epochs, grad_accum_steps=grad_accum_steps)

    out_path = "llama3_8B_alpaca.pth"
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), out_path)
    else:
        torch.save(model.state_dict(), out_path)
    print(f"Finetuning complete. Saved to {out_path}")

if __name__ == '__main__':
    main()
