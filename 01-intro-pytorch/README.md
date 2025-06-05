# Introduction to PyTorch

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

---

### 📌 3.1 A Deeper Dive into Loading Data

You can load custom data using PyTorch’s `TensorDataset`.

```python
from torch.utils.data import TensorDataset, DataLoader
import torch

X = torch.randn(100, 3)
y = torch.randint(0, 2, (100,))

dataset = TensorDataset(X, y)
```

---

### 📌 3.2 Using TensorDataset

```python
print("First item in dataset:", dataset[0])
print("Length of dataset:", len(dataset))
```

---

### 📌 3.3 Using DataLoader

`DataLoader` helps in batching and shuffling data.

```python
loader = DataLoader(dataset, batch_size=10, shuffle=True)
for batch_x, batch_y in loader:
    print("Batch X:", batch_x.shape)
    print("Batch Y:", batch_y.shape)
    break
```

---

### 📌 3.4 Writing Our First Training Loop

A simple loop that goes through the dataset.

```python
model = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2)
)

loss_fn = torch.nn.CrossEntropyLoss()
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

### 📌 3.5 Using the MSELoss

```python
mse_loss_fn = torch.nn.MSELoss()

preds = torch.tensor([[0.5], [0.8]])
targets = torch.tensor([[1.0], [0.0]])
loss = mse_loss_fn(preds, targets)
print("MSE Loss:", loss.item())
```

---

### 📌 3.6 ReLU Activation Functions

```python
x = torch.tensor([[-1.0, 0.0, 1.0]])
relu_output = torch.nn.ReLU()(x)
print("ReLU:", relu_output)
```

---

### 📌 3.7 Implementing ReLU

You can define it manually too:

```python
def relu(x):
    return torch.maximum(torch.zeros_like(x), x)

print("Manual ReLU:", relu(x))
```

---

### 📌 3.8 Implementing Leaky ReLU

```python
leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
print("Leaky ReLU:", leaky_relu(x))
```

---

### 📌 3.9 Understanding Activation Functions

Try with both ReLU and Tanh in the same model to compare.

```python
model_relu = torch.nn.Sequential(
    torch.nn.Linear(3, 3),
    torch.nn.ReLU()
)

model_tanh = torch.nn.Sequential(
    torch.nn.Linear(3, 3),
    torch.nn.Tanh()
)
```

---

### 📌 3.10 Learning Rate and Momentum

Momentum helps accelerate learning in the relevant direction.

```python
optimizer_momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

### 📌 3.11 Experimenting with Learning Rate

```python
# Try small vs large learning rate
optim_small = torch.optim.SGD(model.parameters(), lr=0.0001)
optim_large = torch.optim.SGD(model.parameters(), lr=1.0)
```

---

### 📌 3.12 Experimenting with Momentum

```python
optim_no_momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
optim_high_momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
```


## ✅ 4. Evaluating and Improving Models

---

### 📌 4.1 Layer Initialization and Transfer Learning

Use pre-trained models and modify the final layers.

```python
from torchvision import models

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Replace final layer for 2-class classification
import torch.nn as nn
model.fc = nn.Linear(model.fc.in_features, 2)
```

---

### 📌 4.2 Fine-Tuning Process

Unfreeze some layers and train again.

```python
for param in model.layer4.parameters():
    param.requires_grad = True  # Fine-tune deeper layers
```

---

### 📌 4.3 Freeze Layers of a Model

```python
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False
```

---

### 📌 4.4 Layer Initialization

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

simple_model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

simple_model.apply(init_weights)
```

---

### 📌 4.5 Evaluating Model Performance

Calculate accuracy after each epoch.

```python
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean()

x = torch.randn(8, 4)
y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
model = nn.Linear(4, 4)
with torch.no_grad():
    preds = model(x)
print("Accuracy:", accuracy(preds, y).item())
```

---

### 📌 4.6 Writing the Evaluation Loop

```python
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

---

### 📌 4.7 Calculating Accuracy Using `torchmetrics`

```bash
pip install torchmetrics
```

```python
from torchmetrics.classification import Accuracy
metric = Accuracy(task="multiclass", num_classes=4)

preds = torch.tensor([[0.8, 0.1, 0.05, 0.05], [0.1, 0.6, 0.1, 0.2]])
labels = torch.tensor([0, 1])
print("Torchmetrics Accuracy:", metric(preds, labels).item())
```

---

### 📌 4.8 Fighting Overfitting

Overfitting occurs when your model performs well on training but poorly on validation data.

Use dropout to prevent this.

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(20, 2)
)
```

---

### 📌 4.9 Experimenting with Dropout

```python
model.train()  # dropout active
out_train = model(torch.randn(5, 10))

model.eval()  # dropout inactive
out_eval = model(torch.randn(5, 10))
```

---

### 📌 4.10 Understanding Overfitting

Monitor training/validation accuracy or loss difference over epochs. Use early stopping or regularization.

---

### 📌 4.11 Improving Model Performance

* Add more data
* Reduce complexity
* Try different architectures
* Tune hyperparameters

---

### 📌 4.12 Implementing Random Search

```python
import random

learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [16, 32, 64]

random_config = {
    "lr": random.choice(learning_rates),
    "batch_size": random.choice(batch_sizes)
}

print("Random hyperparameters:", random_config)
```



