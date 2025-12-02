import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from utils import VOCAB_SIZE, SEQ_LEN, DEVICE, SEED

torch.manual_seed(SEED)

EMB_DIM = 32
NUM_HEADS = 4
D_K = EMB_DIM // NUM_HEADS


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.masked_attention = Attention(EMB_DIM, NUM_HEADS, D_K, D_K, SEQ_LEN)
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM, EMB_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMB_DIM * 4, EMB_DIM)
        )

        self.masked_attention_norm = nn.LayerNorm(EMB_DIM)
        self.mlp_norm = nn.LayerNorm(EMB_DIM)


    def forward(self, x):
        x = x + self.masked_attention(self.masked_attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMB_DIM)
        self.position_embedding_table = nn.Embedding(num_embeddings=SEQ_LEN, embedding_dim=EMB_DIM)

        self.block_sequence = nn.Sequential(
            Block(),
            Block()
        )

        self.linearOut = nn.Linear(EMB_DIM, VOCAB_SIZE)
    

    def forward(self, x, targets=None):

        # x: (batch_size, seq_len) of indices in the vocab
        # targets: (batch_size, seq_len) of indices in the vocab
        B, T = x.shape

        token_embeddings = self.token_embedding_table(x) # (batch_size, seq_len, emb_dim)
        position_embeddings = self.position_embedding_table(torch.arange(0, T, device=DEVICE)) # (seq_len, emb_dim)
        x = token_embeddings + position_embeddings # (batch_size, seq_len, emb_dim)

        x = self.block_sequence(x)
        logits = self.linearOut(x)


        if targets == None:
            loss = None
        else:
            BS, SL, VS = logits.shape
            logits = logits.view(BS*SL, VS)
            targets = targets.view(BS*SL)

            # F.cross_entropy includes the softmax
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    
    def generate(self, curr_seq, max_new_tokens):
        for _ in range(max_new_tokens):
            curr_seq_cropped = curr_seq[:, -SEQ_LEN:]
            logits, loss = self(curr_seq_cropped)
            logits = logits[:, -1, :]
            next_token_probs = F.softmax(logits, dim=-1)
            next_token_idxs = torch.multinomial(next_token_probs, num_samples=1)
            curr_seq = torch.cat((curr_seq, next_token_idxs), dim=-1)
        return curr_seq
    