import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from utils import get_batch, decode
from hyperparams import util_hyperparams, model_hyperparams, training_hyperparams

# unpack util hyperparams
SEED = util_hyperparams["seed"]
DEVICE = util_hyperparams["device"]
# -----

# unpack training hyperparams
LEARNING_RATE = training_hyperparams["learning_rate"]
MAX_ITERS = training_hyperparams["max_iters"]
EVAL_ITERS = training_hyperparams["eval_iters"]
EVAL_INTERVAL = training_hyperparams["eval_interval"]
# -----

torch.manual_seed(SEED)

model = LanguageModel()
model = model.to(DEVICE)


# calculate avg loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# -----

def generate(max_new_tokens):

    print("-----")
    print(f"Generating a sequence of {max_new_tokens} new tokens:")
    print("-----")

    seq = torch.zeros(1, 1, dtype=torch.long)
    seq = seq.to(DEVICE)
    print(decode(model.generate(seq, max_new_tokens=max_new_tokens)[0].tolist()))


# training loop
def train():

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("-----")
    print(f"Training on {DEVICE}: ")
    print("-----")


    for iter in range(MAX_ITERS):
    
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("-----")
    print("Training completed")
    print("-----")

    torch.save({
        "model_state": model.state_dict(),
        "optimizer state": optimizer.state_dict(),
        "util_hyperparams": util_hyperparams,
        "model_hyperparams": model_hyperparams,
        "training_hyperparams": training_hyperparams
    }, "checkpoint.pt")

    print("-----")
    print(f"Saved model to 'checkpoint.pt")
    print("-----")
    

# train model
train()

# generate a sequence
generate(max_new_tokens=500)
