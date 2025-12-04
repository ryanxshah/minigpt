import torch
from hyperparams import util_hyperparams, model_hyperparams

# unpack util hparams
seed = util_hyperparams["seed"]
device = util_hyperparams["device"]
# -----

# unpack model hyperparams
batch_size = model_hyperparams["batch_size"]
seq_len = model_hyperparams["seq_len"]
d_model = model_hyperparams["d_model"]
num_heads = model_hyperparams["num_heads"]


torch.manual_seed(seed)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
chars = sorted(list(set(text)))

stoi = {char:i for i, char in enumerate(chars)}
itos = {i:char for i, char in enumerate(chars)}

def encode(string):
    return [stoi[char] for char in string]

def decode(list):
    return "".join([itos[i] for i in list])


DATA = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(DATA))
TRAIN_DATA = DATA[:n]
VAL_DATA = DATA[n:]
VOCAB_SIZE = len(chars)
# -----

# get a batch of data
def get_batch(split):
    data = TRAIN_DATA if split == "train" else VAL_DATA
    indices = torch.randint(0, len(data) - seq_len, (batch_size,)) # create batch_size number of random start indices
    x = torch.stack([data[i:i+seq_len] for i in indices], dim=-2)
    y = torch.stack([data[i+1:i+seq_len+1] for i in indices], dim=-2)

    x, y = x.to(device), y.to(device)
    
    # return inputs, targets -> both are (batch_size, seq_len)
    # these are both tensors containing indices 
    return x, y
