# add all  your Encoder and Decoder code here

import torch.nn
import torch.nn as nn
from torch.nn import functional as F
import constants as c


class Decoder(nn.Module) :

    def __init__(self, vocab_size) :
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, c.n_embd)
        self.position_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.sa_head = Head(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, vocab_size)

    def forward(self, idx, targets=None): 
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=c.device)) 
        x = token_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -c.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Head(nn.Module) :
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(c.n_embd, head_size, bias=False)
        self.query = nn.Linear(c.n_embd, head_size, bias=False)
        self.value = nn.Linear(c.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(c.block_size, c.block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.value(x)

        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x)
        out = weights @ v
        return out 