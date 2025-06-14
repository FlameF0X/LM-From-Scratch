
# 10 — Evaluation and Validation

## Purpose

Evaluate model performance on validation data to monitor training progress and prevent overfitting.

---

## Evaluation Metrics

* **Loss**: Use the same cross-entropy loss as training to assess how well the model predicts tokens.
* **Perplexity**: Exponential of the loss, interpretable as the model’s uncertainty.

---

## Evaluation Process

* Switch model to evaluation mode (`model.eval()`) to disable dropout and other training-only layers.
* No gradient computation (`torch.no_grad()`) for memory efficiency.
* Iterate over validation dataset, compute loss for each batch.
* Average losses across batches to get overall validation loss.

---

## Example Evaluation Loop

```python
model.eval()
val_loss = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        
        val_loss += loss.item()

avg_val_loss = val_loss / len(val_loader)
perplexity = torch.exp(torch.tensor(avg_val_loss))

print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
```

---

## Tips

* Evaluate periodically (e.g., after each epoch or every few steps).
* Use evaluation results to adjust hyperparameters or implement early stopping.
* Save best model checkpoints based on validation loss.

---

## Summary

Systematic evaluation during training helps ensure the model generalizes and provides a meaningful stopping point.

