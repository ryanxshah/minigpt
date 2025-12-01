import torch


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


# hyperparameters
BATCH_SIZE = 32
SEQ_LEN = 8
VOCAB_SIZE = len(chars)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
SEED = 0
# -----

# get a batch of data
def get_batch(split):
    data = TRAIN_DATA if split == "train" else VAL_DATA
    indices = torch.randint(0, len(data) - SEQ_LEN, (BATCH_SIZE,)) # create batch_size number of random start indices
    x = torch.stack([data[i:i+SEQ_LEN] for i in indices], dim=-2)
    y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in indices], dim=-2)

    x, y = x.to(DEVICE), y.to(DEVICE)
    
    # return inputs, targets -> both are (batch_size, seq_len)
    # these are both tensors containing indices 
    return x, y
