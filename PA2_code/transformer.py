# add all  your Encoder and Decoder code here

import torch.nn
import torch.nn as nn
from torch.nn import functional as F
import constants as c
import numpy as np


class Encoder(nn.Module):
    def __init__(self, vocab_size) :
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, c.n_embd)
        self.position_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.Sequential(
            Block(num_head=4, is_decoder=False),
            Block(num_head=4, is_decoder=False),
            Block(num_head=4, is_decoder=False),
            Block(num_head=4, is_decoder=False),
            nn.LayerNorm(c.n_embd)
        )
        # self.lm_head = nn.Linear(c.n_embd, 3)
        self.classifier = FeedForward(c.n_input, c.n_output, c.n_hidden)

    def forward(self, idx): 
        B, T = idx.shape
        # print("called")
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=c.device))
        # print(np.shape(x))
        x = token_emb + pos_emb
        x = self.blocks(x)
        # B, T, C = x.shape
        # x = x.view(B*T, C)
        # print(np.shape(x))
        x = x.mean(dim=-2)
        x = self.classifier(x) 
        return x


class Decoder(nn.Module) :
    def __init__(self, vocab_size) :
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, c.n_embd)
        self.position_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.Sequential(
            Block(num_head=4),
            Block(num_head=4),
            Block(num_head=4),
            Block(num_head=4),
            nn.LayerNorm(c.n_embd)
        )
        self.lm_head = nn.Linear(c.n_embd, vocab_size)

    def forward(self, idx, targets=None): 
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=c.device)) 
        x = token_emb + pos_emb
        x = self.blocks(x)
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
    def __init__(self, head_size, is_decoder=True):
        super().__init__()
        self.is_decoder = is_decoder
        self.key = nn.Linear(c.n_embd, head_size, bias=False)
        self.query = nn.Linear(c.n_embd, head_size, bias=False)
        self.value = nn.Linear(c.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(c.block_size, c.block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.value(x)

        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        if self.is_decoder:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x)
        out = weights @ v
        return out 


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, is_decoder=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, is_decoder) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, c.n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, input_size=c.n_embd, output_size=c.n_embd, hidden_size=4*c.n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_head, is_decoder=True):
        super().__init__()
        head_size = c.n_embd // num_head
        self.sa = MultiHeadAttention(num_head, head_size, is_decoder)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
