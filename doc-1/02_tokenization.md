
# 02 — Tokenization

## Choosing a Tokenizer

Tokenization is the process of converting raw text into a sequence of numerical tokens that can be processed by neural networks. In this guide, we use the **BERT tokenizer** from HuggingFace's `transformers` library:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
```

The BERT tokenizer is a WordPiece-based tokenizer that offers:

* A fixed-size vocabulary
* Efficient handling of out-of-vocabulary words
* Lower memory footprint than byte-pair encoding (BPE) alternatives

## Sequence Padding and Truncation

Language models generally require fixed-length inputs. We apply **max-length truncation and padding** to ensure every sample is the same length:

```python
max_seq_len = 384

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt"
    )
```

This function returns a dictionary with:

* `input_ids`: Token indices
* `attention_mask`: Binary mask indicating valid (non-padding) tokens

## Padding Token Handling

During training, we ignore loss from padded tokens using the `ignore_index` parameter in the loss function:

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
```

This ensures the model is not penalized for predicting padding tokens.

## Reusing the Tokenizer

Once initialized, the tokenizer can be reused for:

* Data preprocessing
* Model input formatting
* Evaluation and inference

The tokenizer can be saved with the model for future compatibility:

```python
tokenizer.save_pretrained("path/to/save")
```

## Summary

* A WordPiece tokenizer from BERT is used for tokenization
* Inputs are padded and truncated to a maximum sequence length
* Padding tokens are ignored during training
* Tokenizer is fully reusable and saveable

Next, we will implement a data loading strategy using PyTorch `DataLoader` to batch our tokenized inputs efficiently.
