import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from utils import vocab_size, seq_len, device


emb_dim = 32

torch.manual_seed(0)

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.position_embedding_table = nn.Embedding(num_embeddings=seq_len, embedding_dim=emb_dim)
        self.projection = nn.Linear(emb_dim, vocab_size)
    

    def forward(self, x, targets=None):

        # x: (batch_size, seq_len) of indices in the vocab
        # targets: (batch_size, seq_len) of indices in the vocab

        token_embeddings = self.token_embedding_table(x) # (batch_size, seq_len, emb_dim)
        position_embeddings = self.position_embedding_table(torch.arange(0, seq_len, device=device)) # (seq_len, emb_dim)
        x = token_embeddings + position_embeddings # (batch_size, seq_len, emb_dim)
        logits = self.projection(x) # (batch_size, seq_len, vocab_size)


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
        for i in range(max_new_tokens):
            logits, loss = self(curr_seq) # logits is (BS, SL, VS)
            logits = logits[:, -1, :] # Keep only the last token in each seq
            next_token_probs = F.softmax(logits, dim=-1)
            next_token_idxs = torch.multinomial(next_token_probs, num_samples=1)

            curr_seq = torch.cat((curr_seq, next_token_idxs), dim=-1)
        return curr_seq



