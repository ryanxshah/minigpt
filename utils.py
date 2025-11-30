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


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# hyperparameters
batch_size = 32
seq_len = 8
vocab_size = len(chars)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# -----

# get a batch of data
def get_batch(split, device):
    data = train_data if split == "train" else val_data
    indices = torch.randint(0, len(data) - seq_len, (batch_size,)) # create batch_size number of random start indices
    x = torch.stack([data[i:i+seq_len] for i in indices], dim=-2)
    y = torch.stack([data[i+1:i+seq_len+1] for i in indices], dim=-2)

    x, y = x.to(device), y.to(device)
    
    # return inputs, targets -> both are (batch_size, seq_len)
    # these are both tensors containing indices 
    return x, y
