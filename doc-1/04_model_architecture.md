# 04 — Model Architecture

## Overview

Our model is a transformer-based encoder stack inspired by BERT and GPT architectures. It includes:

* Token + position embeddings
* Multi-head self-attention with fused QKV projection
* Pre-layer normalization
* Feed-forward networks
* Tied output and input embeddings

We define the model in modular components:

---

## Multi-Head Self-Attention (Fused QKV)

```python
class FusedQKVAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.nh = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.wo(context)
```

---

## Feedforward Layer

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
```

---

## Transformer Block (Pre-Norm)

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FusedQKVAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout1(self.attn(self.ln1(x), mask))
        x = x + self.dropout2(self.ff(self.ln2(x)))
        return x
```

---

## SnowflakeCore: Complete Architecture

```python
class SnowflakeCore(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Parameter(self._build_pos_enc(max_len, d_model), requires_grad=False)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.emb.weight

    def _build_pos_enc(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids) + self.pe[:, :input_ids.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        return self.out(x)
```

---

## Summary

* The architecture uses a transformer encoder stack with fused QKV attention and pre-layer normalization
* Positional encodings are sinusoidal and fixed
* Input and output embeddings are tied
* Model supports standard attention masking for padding

In the next section, we’ll configure the optimizer, gradient scaling, and learning rate scheduler for training.

