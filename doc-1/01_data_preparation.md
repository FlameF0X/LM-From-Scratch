
# 01 — Data Preparation

## Dataset: DialogMLM-50K

For this project, we use the **DialogMLM-50K** dataset—a corpus of multi-turn dialogue examples suitable for masked language modeling. While you can use any text dataset, DialogMLM-50K provides a clean and compact benchmark for fast iteration and experimentation.

We load the dataset using the 🤗 HuggingFace Datasets library:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("FlameF0X/DialogMLM-50K")
```

## Understanding the Structure

Each sample in the dataset contains one key field:

* `text`: A string containing one or more utterances.

Example:

```json
{
  "text": "Hello, how can I help you today? I'm looking for a nearby restaurant."
}
```

## Preprocessing: Tokenization + Truncation

We prepare each text sample by applying a tokenizer with truncation and padding:

```python
from transformers import BertTokenizer

max_seq_len = 384

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt"
    )

# Apply tokenizer to the entire dataset
tokenized_ds = dataset.map(tokenize, batched=True, batch_size=24, remove_columns=["text"])
tokenized_ds.set_format("torch")
```

## Train/Validation Split

If a validation set is not provided, we manually split the training set:

```python
if "validation" in tokenized_ds:
    train_ds = tokenized_ds["train"]
    val_ds = tokenized_ds["validation"]
else:
    split = tokenized_ds["train"].train_test_split(test_size=0.1)
    train_ds = split["train"]
    val_ds = split["test"]
```

## Summary

* **DialogMLM-50K** is used as the base dataset
* Tokenization uses `bert-base-uncased`
* Padding/truncation ensures fixed-length input
* A validation split ensures robust training

In the next section, we will build a DataLoader to feed this tokenized data efficiently into our model.
