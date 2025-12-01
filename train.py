import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from utils import get_batch, decode, DEVICE

torch.manual_seed(0)

model = LanguageModel()
model = model.to(DEVICE)

learning_rate = 1e-3
eval_iters = 200
eval_interval = 300
max_iters = 10000

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Training on {DEVICE}: ")


    for iter in range(max_iters):
    
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training completed")

# train model
train()

# generate a sequence
seq = torch.zeros(1, 1, dtype=torch.long)
seq = seq.to(device)
print(decode(model.generate(seq, max_new_tokens=500)[0].tolist()))
