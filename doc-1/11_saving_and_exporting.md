
# 11 — Saving and Exporting the Model

## Purpose

Properly saving and exporting your trained model ensures reproducibility, easy deployment, and interoperability with popular frameworks like HuggingFace Transformers.

---

## What to Save

* **Model state dictionary (`state_dict`)**: Contains all learned parameters.
* **Tokenizer**: To maintain consistent tokenization during inference.
* **Model configuration**: Captures architecture hyperparameters for reloading.

---

## Saving with PyTorch

```python
import os

save_dir = "path/to/save_dir"
os.makedirs(save_dir, exist_ok=True)

# Save model weights
torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

# Save tokenizer
tokenizer.save_pretrained(save_dir)

# Save configuration
model.config.save_pretrained(save_dir)
```

---

## Saving with safetensors

* `safetensors` is a safer and faster alternative for storing model weights.
* Supports memory-mapped loading and mitigates security risks.

```python
from safetensors.torch import save_model

save_model(model.half(), os.path.join(save_dir, "model.safetensors"), metadata={"format": "pt"})
```

---

## Loading the Model Later

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(save_dir)
config = AutoConfig.from_pretrained(save_dir)
model = AutoModel.from_config(config)
model.load_state_dict(torch.load(os.path.join(save_dir, "pytorch_model.bin")))
model.to(device)
```

---

## Tips

* Save the model in half precision (`float16`) for reduced disk usage.
* Maintain consistent naming conventions for easy loading.
* Regularly checkpoint during training to prevent data loss.

---

## Summary

Saving your model, tokenizer, and configuration properly enables seamless future use, fine-tuning, or deployment. Using formats like `safetensors` improves security and efficiency.
