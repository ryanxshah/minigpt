import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention
from utils import vocab_size


torch.manual_seed(0)

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        # x = (batch_size, seq_len) of indices in the vocab
        # targets = (batch_size, seq_len)

        logits = self.token_embedding_table(x)

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



