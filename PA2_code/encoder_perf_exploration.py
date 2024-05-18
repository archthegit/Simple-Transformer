import torch.nn
import torch.nn as nn
from torch.nn import functional as F
import constants as c


class ImprovedPerfEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, c.n_embd)
        self.position_embedding_table = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList([Block(num_head=4, is_decoder=False) for _ in range(4)])
        self.ln = nn.LayerNorm(c.n_embd)
        self.classifier = FeedForward(c.n_input, c.n_output, c.n_hidden)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=c.device))
        x = token_emb + pos_emb

        all_attn_maps = []
        for block in self.blocks:
            x, attn_maps = block(x)
            all_attn_maps.extend(attn_maps)
        
        x = self.ln(x)
        x = x.mean(dim=-2)
        logits = self.classifier(x)
        attn_maps_tensor = torch.stack(all_attn_maps)

        return logits, attn_maps_tensor

class Head(nn.Module):
    def __init__(self, head_size, is_decoder=True):
        super().__init__()
        self.is_decoder = is_decoder
        self.key = nn.Linear(c.n_embd, head_size, bias=False)
        self.query = nn.Linear(c.n_embd, head_size, bias=False)
        self.value = nn.Linear(c.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(c.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(c.block_size, c.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        if self.is_decoder:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v

        return out, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, is_decoder=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, is_decoder) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, c.n_embd)
        self.dropout = nn.Dropout(c.dropout)
    
    def forward(self, x):
        outputs = []
        attn_maps = []
        for head in self.heads:
            out, weights = head(x)
            outputs.append(out)
            attn_maps.append(weights)
        
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, attn_maps


class FeedForward(nn.Module):
    def __init__(self, input_size=c.n_embd, output_size=c.n_embd, hidden_size=4*c.n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),
            nn.Dropout(c.dropout),
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
        sa_out, attn_maps = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, attn_maps

