# 08 — Model Wrapper and Full Architecture

## Purpose

The model wrapper combines multiple transformer blocks, embeddings, positional encodings, and output layers into a complete language model architecture.

---

## Core Components

1. **Token Embeddings**: Converts token IDs into dense vectors.
2. **Positional Encodings**: Adds position information to token embeddings.
3. **Stack of Transformer Blocks**: Multiple layers of attention and feedforward.
4. **Final LayerNorm**: Normalizes the output of the transformer stack.
5. **Output Projection**: Maps transformer output to vocabulary logits.
6. **Weight Tying**: Shares weights between input embeddings and output projection for parameter efficiency.

---

## Implementation

```python
class SnowflakeCore(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Initialize sinusoidal positional encodings
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe.data = pe.data
        
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Weight tying
        self.fc.weight = self.emb.weight
        
        # Weight initialization
        nn.init.normal_(self.emb.weight, mean=0, std=0.02)
        
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        x = self.emb(input_ids) + self.pe[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        logits = self.fc(x)
        return logits
    
    def get_input_embeddings(self):
        return self.emb
    
    def set_input_embeddings(self, embedding):
        self.emb = embedding
        self.fc.weight = embedding.weight
    
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        # Additional save code (e.g., config, tokenizer) can be added here
```

---

## Notes

* **Positional Encoding**: Fixed sinusoidal encodings injected as learned parameters for flexibility.
* **Weight Tying**: Reduces parameters and improves generalization.
* **LayerNorm and Dropout**: Stabilize training and regularize.
* **Extensibility**: This wrapper makes it straightforward to add features like custom tokenizers or advanced positional embeddings.

---

## Summary

This class encapsulates the entire transformer architecture, managing input embedding, positional information, stacked transformer blocks, normalization, and output projection — forming the backbone of a language model capable of text generation.
