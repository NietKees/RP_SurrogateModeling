import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# simple surrogate model (PhysicsNeMo-style ML surrogate)
model = nn.Sequential(
    nn.Linear(2, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 1)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(200):
    x = torch.randn(2048, 2, device=device)
    y = torch.sin(x[:, 0:1]) + x[:, 1:2]**2

    pred = model(x)
    loss = ((pred - y)**2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 20 == 0:
        print(f"step {i} loss {loss.item():.6f}")


