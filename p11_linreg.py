# Program 11: Linear Regression using PyTorch (gradients + optimization)
import torch
import torch.nn as nn

torch.manual_seed(0)

# Data following y = 2x + 1
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = 2 * x + 1

model = nn.Linear(1, 1)                       # y = w*x + b
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()                     # clear old gradients
    loss.backward()                           # compute dLoss/dw and dLoss/db
    optimizer.step()                          # w = w - lr * grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  Loss = {loss.item():.4f}  "
              f"dL/dw = {model.weight.grad.item():.4f}  dL/db = {model.bias.grad.item():.4f}")

print(f"\nLearned: w = {model.weight.item():.3f}, b = {model.bias.item():.3f}  (true: w = 2, b = 1)")
