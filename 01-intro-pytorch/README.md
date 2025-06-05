# Deep Learning with Pytorch

# PyTorch Learning Series

Welcome to the PyTorch Learning Series! This guide covers everything from the basics to advanced topics like transformers.

---

## 📚 Table of Contents

1. [Introduction to PyTorch](./01-intro-pytorch/README.md)
2. [Intermediate PyTorch](./02-intermediate-pytorch/README.md)
3. [Image Processing with PyTorch](./03-image-pytorch/README.md)
4. [Text Processing with PyTorch](./04-text-pytorch/README.md)
5. [Transformer Models with PyTorch](./05-transformers-pytorch/README.md)

## ✅ 1. Introduction to PyTorch, a Deep Learning Library

### 📌 1.1 Introduction to Deep Learning with PyTorch
```python
import torch
print(torch.__version__)  # Print PyTorch version
```

### 📌 1.2 Getting Started with PyTorch Tensors
```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1, 2], [3, 4]])
print(a, a.shape)
print(b, b.shape)
```

### 📌 1.3 Checking and Adding Tensors
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("Sum:", a + b)
print("Elementwise multiply:", a * b)
```

### 📌 1.4 Neural Networks and Layers
```python
from torch import nn
layer = nn.Linear(3, 2)
print(layer)
```

### 📌 1.5 Linear Layer Network
```python
x = torch.randn(1, 3)
layer = nn.Linear(3, 2)
output = layer(x)
print("Output:", output)
```

### 📌 1.6 Understanding Weights
```python
print("Weights:", layer.weight)
print("Bias:", layer.bias)
```

### 📌 1.7 Hidden Layers and Parameters
```python
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
print(model)
```

### 📌 1.8 Your First Neural Network
```python
x = torch.randn(10, 3)
y = torch.randn(10, 1)

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

pred = model(x)
print(pred.shape)
```

### 📌 1.9 Stacking Linear Layers
```python
stacked_model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)
print(stacked_model)
```

### 📌 1.10 Counting the Number of Parameters
```python
def count_params(model):
    return sum(p.numel() for p in model.parameters())

print("Total parameters:", count_params(stacked_model))
```

---

## ✅ 2. Neural Network Architecture and Hyperparameters

### 📌 2.1 Discovering Activation Functions
```python
import torch.nn.functional as F
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("ReLU:", F.relu(x))
print("Tanh:", torch.tanh(x))
```

### 📌 2.2 The Sigmoid and Softmax Functions
```python
x = torch.tensor([[1.0, 2.0, 3.0]])
print("Sigmoid:", torch.sigmoid(x))
print("Softmax:", F.softmax(x, dim=1))
```

### 📌 2.3 Running a Forward Pass
```python
import torch.nn as nn
model = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
input_tensor = torch.tensor([[0.2, 0.4, 0.6]])
output = model(input_tensor)
print("Forward pass output:", output)
```

### 📌 2.4 Building a Binary Classifier
```python
model = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
x = torch.randn(5, 3)
y = torch.tensor([[0.], [1.], [0.], [1.], [1.]])
loss_fn = nn.BCELoss()
output = model(x)
loss = loss_fn(output, y)
print("Binary classification loss:", loss.item())
```

### 📌 2.5 Regression to Multi-Class Classification
```python
x = torch.randn(4, 5)
y = torch.tensor([0, 1, 2, 3])
model = nn.Linear(5, 4)
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, y)
print("Multi-class loss:", loss.item())
```

### 📌 2.6 One-Hot Encoding
```python
labels = torch.tensor([0, 2, 1])
one_hot = F.one_hot(labels, num_classes=3)
print("One-hot labels:", one_hot)
```

### 📌 2.7 Cross Entropy Loss
```python
logits = torch.tensor([[1.5, 0.3, 0.2]], requires_grad=True)
target = torch.tensor([0])
loss = nn.CrossEntropyLoss()(logits, target)
print("Cross entropy loss:", loss.item())
```

### 📌 2.8 Derivatives for Parameter Updates
```python
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
w = torch.randn(3, 1, requires_grad=True)
output = x @ w
output.sum().backward()
print("Gradients of w:", w.grad)
```

### 📌 2.9 Accessing Model Parameters
```python
model = nn.Linear(3, 2)
for name, param in model.named_parameters():
    print(name, param.shape)
```

### 📌 2.10 Manual Weight Update
```python
with torch.no_grad():
    for param in model.parameters():
        param -= 0.01 * param.grad
```

### 📌 2.11 Using PyTorch Optimizer
```python
model = nn.Linear(3, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(10, 3)
y = torch.randint(0, 2, (10,))
criterion = nn.CrossEntropyLoss()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

---

## ✅ 3. Training a Neural Network with PyTorch

```python
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(100, 3)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(3):
    for xb, yb in loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## ✅ 4. Evaluating and Improving Models

```python
from torchvision import models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean()

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
```
