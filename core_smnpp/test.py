import torch

x = torch.linspace(1,6,6)
y = torch.linspace(1,5,5)
d = torch.linspace(1,4,4)

print(x.shape)
print(y.shape)
print(d.shape)

dg, xg, yg = torch.meshgrid(d, x, y)
g = torch.cat((dg.unsqueeze(-1), xg.unsqueeze(-1), yg.unsqueeze(-1)), -1).reshape(4,-1,3)

print(xg.shape)
print(yg.shape)
print(dg.shape)
print(g.shape)

