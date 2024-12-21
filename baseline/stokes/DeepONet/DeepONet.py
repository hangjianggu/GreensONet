import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import meshio
from scipy.interpolate import griddata
import numpy as np


def F(x, case_index):
    f_file = os.path.join('../../data/stokes/train_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(f_file)
    points = mesh.points
    fx = mesh.point_data.get("fx", None)
    interpolated_fx = griddata(points=points, values=fx, xi=x, method='nearest')
    fy = mesh.point_data.get("fy", None)
    interpolated_fy = griddata(points=points, values=fy, xi=x, method='nearest')
    fz = mesh.point_data.get("fz", None)
    interpolated_fz = griddata(points=points, values=fz, xi=x, method='nearest')
    res = np.column_stack((interpolated_fx, interpolated_fy, interpolated_fz))
    return res.flatten()


def boundary_v_mag(x):
    result = np.zeros_like(x[:, 0])
    mask = x[:, 2] == 1
    result[mask] = 1
    return result


def U_exact(x, case_index):
    u_exact_file = os.path.join('../../data/stokes/train_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)
    v_exact = mesh.point_data.get("v", None)
    w_exact = mesh.point_data.get("w", None)

    U = (u_exact**2 + v_exact**2 + w_exact**2)**0.5
    interpolated_values = griddata(points=points, values=U, xi=x, method='nearest')
    return interpolated_values


def read_mesh(meshfile):
    with open(meshfile, 'r') as file:
        line1 = next(file).strip()
        line2 = next(file).strip()
        n_str, e_str, _, _ = line2.split(',')
        n = int(n_str.split('=')[1])
        e = int(e_str.split('=')[1])

        vertices = []
        for _ in range(n):
            line = next(file).strip()
            values = line.split()
            x, y, z = map(float, values)
            vertices.append((x, y, z))

        tetrahedrons = []
        for _ in range(e):
            line = next(file).strip()
            values = line.split()
            ele = list(map(int, values))
            tetrahedrons.append(ele)

    vertices = np.array(vertices)
    tetrahedrons = np.array(tetrahedrons)
    return vertices, tetrahedrons


def save_to_vtk(vertices, elements, u_pred, u_exact):
    mesh = meshio.Mesh(
        points = vertices,
        cells = [("tetra", elements)],
        point_data={"u_pred": u_pred, "u_exac": u_exact, "u_error": np.abs(u_pred - u_exact)}
    )
    mesh.write('case1_deeponet.vtu')


class CustomNet(nn.Module):
    def __init__(self, layers):
        super(CustomNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x


class DeepONet(nn.Module):
    def __init__(self, branch_layers1, branch_layers2, trunk_layers):
        super(DeepONet, self).__init__()
        self.branch_net1 = CustomNet(branch_layers1)
        self.branch_net2 = CustomNet(branch_layers2)
        self.trunk_net = CustomNet(trunk_layers)

    def forward(self, input_data1, input_data2, spatial_coords):
        branch_output1 = self.branch_net1(input_data1)
        branch_output2 = self.branch_net2(input_data2)
        trunk_output = self.trunk_net(spatial_coords)
        combined_branch_output = branch_output1 + branch_output2  # 或者使用其他组合方式
        u_pred = combined_branch_output * trunk_output
        # return torch.sum(combined_branch_output * trunk_output, dim=2, keepdim=True)
        return u_pred


########################################### main函数 ###################################################################

# 读入网格点
vertices, elements = read_mesh('../../mesh/stokes/domain.mphtxt')
npoints = vertices.shape[0]

spatial_coords = []
f_data = []
BC_data = []
targets = []
for case_index in range(15):
    spatial_coords.append(vertices)
    f_data.append(np.tile(F(vertices, case_index), (npoints, 1)))
    BC_data.append(np.tile(boundary_v_mag(vertices), (npoints, 1)))
    targets.append(U_exact(vertices, case_index)[..., np.newaxis])

# 定义用户自定义的网络层数和每层的维度
branch_layers1 = [npoints*3, 12, 24, 12, 4]  # 第一个分支网络 ——> f
branch_layers2 = [npoints, 12, 24, 12, 4]  # 第二个分支网络 ——> BC
trunk_layers = [3, 12, 24, 12, 4]    # 主干网络 ——>  spatial_coords

# 创建DeepONet
deeponet = DeepONet(branch_layers1, branch_layers2, trunk_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(deeponet.parameters(), lr=0.001)

spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32)
f_data = torch.tensor(f_data, dtype=torch.float32)
BC_data = torch.tensor(BC_data, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# 训练模型
for epoch in range(2000):
    start_time = time.time()
    optimizer.zero_grad()
    output = deeponet(f_data, BC_data, spatial_coords)
    output_mag = torch.sqrt(output[:, :, [0]]**2+output[:, :, [1]]**2+output[:, :, [2]]**2)
    #     data_loss = criterion(output_mag, targets)
    loss = criterion(output_mag, targets)
    loss.backward()
    optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time
    print(f'Epoch {epoch}, Loss: {loss.item()}, time: {training_time:.2f} seconds')

    if epoch % 100 == 0:
        torch.save(deeponet.state_dict(), 'deeponet_model_uvw.pth')

# model_path = 'deeponet_model.pth'
# deeponet.load_state_dict(torch.load(model_path))
#
# # 使用模型进行预测
# with torch.no_grad():
#     output = deeponet(f_data, BC_data, spatial_coords)
#     loss = criterion(output, targets)
#     print(loss.item())
#     save_to_vtk(vertices, elements, output[0, ...].detach().cpu().numpy(), targets[0, ...].detach().cpu().numpy())