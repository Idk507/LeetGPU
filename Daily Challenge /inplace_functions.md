copy_()
```python
import torch

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
B = torch.tensor([[5.0], [6.0]], device='cuda')
C = torch.empty((2, 1), device='cuda')

C.copy_(A @ B)  # Now C contains the result of A @ B
```

---

### 🔁 `add_()` — In-place Addition
```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x.add_(5)  # adds 5 to each element
# x becomes tensor([6., 7., 8.])
```

---

### ✖️ `mul_()` — In-place Multiplication
```python
x = torch.tensor([2, 4, 6], dtype=torch.float32)
x.mul_(2)
# x becomes tensor([4., 8., 12.])
```

---

### 🧯 `zero_()` — Set All Elements to Zero
```python
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x.zero_()
# x becomes tensor([[0., 0.], [0., 0.]])
```

---

### 🎯 `fill_()` — Fill with Scalar Value
```python
x = torch.empty(3)
x.fill_(7)
# x becomes tensor([7., 7., 7.])
```

---

### 🔁 `resize_()` — Reshape the Tensor (Caution⚠️)
```python
x = torch.tensor([[1, 2], [3, 4]])
x.resize_(4)  # dangerous: breaks shape and possibly content
# x might become tensor([1, 2, 3, 4])
```

Use this only if you understand what you're doing! Prefer `reshape()` or `view()` for safer reshaping.

---

### 🎲 `normal_()` — Fill with Normal Distribution
```python
x = torch.empty(5)
x.normal_(mean=0.0, std=1.0)
# x will be filled with random values from N(0,1)
```

---


