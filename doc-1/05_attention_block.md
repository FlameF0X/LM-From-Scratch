# 05 — Attention Block Deep Dive

## Motivation

Self-attention is the core of transformer models. It enables each token to attend to others in the input sequence, modeling long-range dependencies efficiently. We use a **fused QKV projection** and a **pre-norm residual layout**, which improves stability and performance.

---

## Structure Recap

Each attention block in our architecture includes:

* LayerNorm (pre-normalization)
* Multi-head self-attention with fused QKV projection
* Residual connection
* Dropout

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

## Why Fused QKV?

Traditionally, queries (Q), keys (K), and values (V) are projected using separate linear layers. Fusing them improves performance:

* Reduces memory overhead
* Improves compute efficiency by minimizing kernel launches
* Common in production-ready implementations (e.g., HuggingFace, NVIDIA APEX)

---

## Attention Masking

Masking allows the model to ignore padded tokens during attention score computation. In our implementation:

* The attention mask is expanded to shape `(B, 1, 1, T)`
* Scores are masked using `masked_fill`
* The model assigns `-inf` to masked positions before softmax

This ensures invalid positions (like padding) do not influence output representations.

---

## Pre-Normalization Layout

We apply `LayerNorm` before the attention sub-layer:

```python
x = x + dropout(attn(layernorm(x), mask))
```

Advantages of pre-norm:

* Better gradient flow, especially in deeper networks
* Allows for more stable training dynamics
* Used in most modern encoder designs (e.g., T5, GPT-NeoX)

---

## Final Output

The output of attention is passed through a projection layer `wo` to transform it back to the model’s dimensionality. This enables interaction with subsequent blocks (FFN and beyond).

---

## Summary

* Self-attention captures token interactions across the sequence
* Fused QKV projection improves performance
* Attention masks ensure proper handling of padding
* Pre-norm residual layout supports deeper, more stable training

Next, we’ll explore the feed-forward sublayer in detail.
