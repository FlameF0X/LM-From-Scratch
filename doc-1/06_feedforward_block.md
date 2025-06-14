# 06 — Feedforward Block Deep Dive

## Purpose

After the self-attention operation models relationships between tokens, each token's representation is passed through a **position-wise feedforward network (FFN)**. This module enables nonlinear transformation and local feature projection at the token level.

---

## Structure

Our FFN block is composed of:

* Linear projection from `d_model` to `ff_dim`
* Activation function (GELU)
* Dropout
* Linear projection back to `d_model`

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_model, ff_dim)
        self.a = nn.GELU()
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(ff_dim, d_model)
        self.d2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.d2(self.l2(self.d1(self.a(self.l1(x)))))
```

---

## Why GELU?

GELU (Gaussian Error Linear Unit) is a smooth, non-linear activation function often used in transformer models. Compared to ReLU:

* It better approximates the expected behavior of stochastic neurons
* Produces smoother gradients
* Empirically leads to improved performance in NLP tasks

---

## Position-wise Application

The FFN is applied independently to each token position:

* Input: `(batch, sequence_length, d_model)`
* Each token vector is transformed identically

This design allows the model to learn **local token-specific transformations** while attention handles global interactions.

---

## Dropout Regularization

Dropout is applied after both linear layers to prevent overfitting:

* Helps stabilize training
* Encourages sparsity in intermediate representations

---

## Integration with Residual & Norm

As with attention, the FFN block is wrapped with a pre-norm and residual connection:

```python
x = x + dropout(ffn(layernorm(x)))
```

This structure improves gradient stability and training convergence.

---

## Summary

* FFN is a 2-layer MLP applied independently per token
* GELU is used for activation, offering smoother behavior than ReLU
* Dropout adds regularization
* The block is wrapped in a pre-norm residual structure for training stability

In the next section, we’ll examine how multiple blocks are stacked to form the full transformer encoder.
