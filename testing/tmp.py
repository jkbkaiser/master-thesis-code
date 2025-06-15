import torch
from geoopt import PoincareBallExact

ball = PoincareBallExact(c=1.0)

x = torch.randn(1000, 128)
x = x / x.norm(dim=1, keepdim=True) * 0.55  # Norm > 1
x_proj = ball.projx(x)

print("Original max norm:", x.norm(dim=1).max().item())
print("Projected max norm:", x_proj.norm(dim=1).max().item())
print("Projected min norm:", x_proj.norm(dim=1).min().item())

