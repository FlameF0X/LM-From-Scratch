# 07 — Transformer Block Composition

## Purpose

The **Transformer Block** combines two critical components:

1. Multi-head self-attention
2. Position-wise feedforward network (FFN)

These are stacked with residual connections and layer normalization to form a complete layer in the transformer encoder.

---

## Architectural Overview

Each block performs the following operations in sequence:

```python
x = x + Dropout(SelfAttention(LayerNorm(x)))
x = x + Dropout(FeedForward(LayerNorm(x)))
```

This **pre-normalization** structure is used for improved stability in deeper models.

---

## Implementation

Here’s the complete implementation:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = FusedMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Pre-Norm Self-Attention
        att_input = self.norm1(x)
        att_out = self.att(att_input, attention_mask)
        x = x + self.drop1(att_out)

        # Pre-Norm Feedforward
        ff_input = self.norm2(x)
        ff_out = self.ffn(ff_input)
        x = x + self.drop2(ff_out)

        return x
```

---

## Layer Normalization (Pre-Norm)

* Applied **before** each sublayer (attention and FFN)
* Stabilizes the network by normalizing activations
* Prevents vanishing gradients in deep stacks

---

## Residual Connections

Each sublayer (attention and feedforward) uses a residual (skip) connection:

* Adds the output back to the input
* Helps preserve gradients
* Allows deeper models to train more efficiently

---

## Dropout Regularization

Dropout is applied **after** each subcomponent:

* Controls overfitting
* Adds noise for robustness
* Keeps training stable

---

## Summary

* Each Transformer block = Self-Attention + FFN
* Wrapped with pre-layer normalization and residuals
* Dropout is used after each component
* This block forms the foundation of the encoder stack

In the next section, we’ll explore how to assemble these blocks into a full model capable of language modeling from scratch.
