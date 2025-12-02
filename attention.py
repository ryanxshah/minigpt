import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from utils import DEVICE


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v, seq_len):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.seq_len = seq_len

        self.linearK = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.linearQ = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.linearV = nn.Linear(d_model, num_heads * d_v, bias=False)

        self.linearOut = nn.Linear(num_heads * d_v, d_model)


    def forward(self, XKV, XQ=None):
        
        K = self.linearK(XKV)
        Q = self.linearQ(XKV) if XQ is None else self.linearQ(XQ)
        V = self.linearV(XKV)

        K = rearrange(K, "batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        Q = rearrange(Q, "batch_size seq_lenQ (num_heads d_k) -> batch_size num_heads seq_lenQ d_k", num_heads=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "batch_size seq_len (num_heads d_v) -> batch_size num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)

        BS, SL, ED = XKV.shape
        tril = torch.tril(torch.ones(self.seq_len, self.seq_len))
        zeros = torch.zeros(SL, SL)
        mask = zeros.masked_fill(tril[:SL, :SL]==0, float("-inf")).to(DEVICE)


        attn_filter = F.softmax(((Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)) + mask, dim=-1)
        fv = attn_filter @ V
        concatenated = rearrange(fv, "batch_size num_heads seq_lenQ d_v -> batch_size seq_lenQ (num_heads d_v)")
        output = self.linearOut(concatenated)

        return output
        
        