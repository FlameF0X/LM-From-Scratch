
# 09 — Training Loop and Optimization

## Purpose

This section covers the process of training the language model from scratch, including data loading, loss computation, backpropagation, optimizer setup, and learning rate scheduling.

---

## Data Preparation

* Use a tokenizer to convert raw text into token IDs.
* Pad or truncate sequences to a fixed maximum length.
* Create datasets and dataloaders with batching and shuffling.

---

## Loss Function

* Use `CrossEntropyLoss` with `ignore_index` set to the padding token ID.
* This allows the model to ignore padded tokens during loss calculation.

---

## Optimizer and Scheduler

* **AdamW** optimizer is preferred for transformer training, with weight decay for regularization.
* Use a **CosineAnnealingLR** scheduler for smooth learning rate decay over training steps.
* Optionally, use gradient clipping to stabilize training.

---

## Mixed Precision and Gradient Accumulation

* Mixed precision training (`torch.cuda.amp`) reduces memory usage and speeds up training.
* Gradient accumulation allows effective batch size increase by accumulating gradients over multiple mini-batches before optimizer step.

---

## Example Training Loop

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SnowflakeCore(...).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
scaler = torch.cuda.amp.GradScaler()

model.train()
global_step = 0
accumulation_steps = 4

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        epoch_loss += loss.item() * accumulation_steps

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
```

---

## Key Points

* Keep track of global steps for scheduler.
* Use gradient clipping to prevent exploding gradients.
* Mixed precision training requires scaling gradients.
* Accumulate gradients to handle large batch sizes beyond GPU memory limits.

---

## Summary

This training loop implements essential best practices to efficiently and stably train a transformer language model from scratch using modern techniques.
