# 03 — Data Loading

## Efficient Batching with PyTorch

Once the dataset is tokenized, we need to prepare it for training using PyTorch's `DataLoader`. This enables efficient mini-batching, shuffling, and prefetching.

We begin by defining a custom **collate function** to assemble individual samples into batched tensors:

```python
import torch

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
```

This function ensures that the tensors are correctly batched and formatted for model input.

## Creating the DataLoaders

We create separate `DataLoader` instances for training and validation:

```python
from torch.utils.data import DataLoader

batch_size = 4  # Can be increased with gradient accumulation

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=1
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    collate_fn=collate_fn,
    pin_memory=True
)
```

* `shuffle=True` ensures randomness during training.
* `pin_memory=True` enables faster data transfer to GPU.
* `num_workers=1` can be tuned based on hardware for parallel data loading.

## Optimization Tip: Tokenized Batching

For larger datasets, you may consider dynamically padding or bucketing sequences of similar lengths to improve memory and speed. For now, we use static padding for simplicity and reproducibility.

## Summary

* Defined a custom `collate_fn` to batch `input_ids` and `attention_mask`
* Created separate `DataLoader`s for training and validation
* Enabled shuffling, pinning, and multi-threaded loading

With data loading in place, we’re ready to construct the model architecture itself in the next section.
