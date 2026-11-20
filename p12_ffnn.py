# Program 12: Basic feedforward neural network on random data using PyTorch
import torch
import torch.nn as nn

torch.manual_seed(0)

# Random data: 100 samples, 10 input features, 1 output
X = torch.randn(100, 10)
y = torch.randn(100, 1)

model = nn.Sequential(
    nn.Linear(10, 16),     # input layer -> hidden layer
    nn.ReLU(),             # activation
    nn.Linear(16, 1)       # hidden layer -> output
)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    pred = model(X)                 # forward pass
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()                 # backward pass
    optimizer.step()                # update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  Loss = {loss.item():.4f}")

print(f"Final Loss = {loss.item():.4f}")
