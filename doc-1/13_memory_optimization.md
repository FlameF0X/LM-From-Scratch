
# 13 — Memory Optimization Techniques

## Purpose

Efficient memory management is crucial when training large language models to fit within hardware constraints and accelerate training.

---

## Techniques Covered

### 1. Mixed Precision Training

* Use `torch.cuda.amp` to train in **half precision (float16)** with automatic loss scaling.
* Benefits:

  * Reduces memory footprint.
  * Increases throughput on compatible GPUs.
* Code snippet:

  ```python
  scaler = torch.cuda.amp.GradScaler()

  with torch.cuda.amp.autocast():
      outputs = model(inputs)
      loss = criterion(outputs, targets)

  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

---

### 2. Gradient Accumulation

* Accumulate gradients over multiple mini-batches before stepping optimizer.
* Allows effective larger batch sizes without increasing memory use.
* Implemented by dividing loss and calling `optimizer.step()` less frequently.

---

### 3. Layer Normalization Before Attention and FFN (Pre-Norm)

* Pre-norm architecture stabilizes gradients, allowing deeper models and larger batch sizes.
* Reduces memory spikes during backpropagation.

---

### 4. Weight Tying

* Share weights between the embedding layer and output projection.
* Saves memory by reducing parameter count.

---

### 5. Efficient Attention Implementation

* Fuse QKV linear projection to one matrix multiplication.
* Minimize intermediate memory allocations.
* Use attention masks to avoid unnecessary computation.

---

### 6. Half-Precision Model Storage

* Store model weights in `float16` to reduce disk space and improve loading times.
* Use libraries like `safetensors` for safe and efficient storage.

---

### 7. Garbage Collection and Cache Clearing

* Explicitly call `torch.cuda.empty_cache()` and Python `gc.collect()` periodically during training.
* Helps prevent memory fragmentation and leaks.

---

## Summary

Combining these memory optimization strategies allows training larger models with limited resources, speeds up training, and improves stability.
