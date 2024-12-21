import meshio
import numpy as np
import torch.nn as nn
import torch

mesh = meshio.read("results/Stokes/case16_GreenONet.vtu")

u_pred = mesh.point_data['u_pred']
u_exac = mesh.point_data['u_exac']

criterion = nn.MSELoss()
loss = criterion(torch.tensor(u_exac, dtype=torch.float32), torch.tensor(u_pred, dtype=torch.float32))
print(loss.item())