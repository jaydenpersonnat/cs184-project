import torch
import numpy as np

# A = 10
# S = 30
# H = 5

# x = (1.0 / A) * torch.ones(S, A, H, dtype = torch.float32)

# x[0, :, 0]

# x = torch.zeros(14,2)
# x[0,0], x[0,1] = 2, 3
# print(x)

P = torch.zeros(3, 3, 3, 3, 5, 2)
thing = torch.ones(3, 3, 3, 3)
P[:, :, :, :, 0, 0] = thing

print(np.max(P, thing))