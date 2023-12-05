import torch
import numpy as np

A = 10
S = 30
H = 5

x = (1.0 / A) * torch.ones(S, A, H, dtype = torch.float32)

x[0, :, 0]