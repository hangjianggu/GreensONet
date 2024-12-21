import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import meshio
from scipy.interpolate import griddata
import numpy as np

mu = 1/100


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

    def forward(self, f, bc_data, spatial_coords):
        f.requires_grad_(True)
        bc_data.requires_grad_(True)
        spatial_coords.requires_grad_(True)

        branch_output1 = self.branch_net1(f)
        branch_output2 = self.branch_net2(bc_data)
        trunk_output = self.trunk_net(spatial_coords)
        combined_branch_output = branch_output1 + branch_output2  # 或者使用其他组合方式
        u_pred = combined_branch_output * trunk_output
        return u_pred


def stoke_loss(u_pred, f_flatten, spatial_coords):
    f = f_flatten.reshape(15, -1, 3)

    u_pred.requires_grad_(True)

    # 提取 ux, uy, uz, p
    ux = u_pred[:, :, 0]
    uy = u_pred[:, :, 1]
    uz = u_pred[:, :, 2]
    p = u_pred[:, :, 3]

    # 计算梯度和拉普拉斯算子
    ux_grad = torch.autograd.grad(ux.sum(), spatial_coords, create_graph=True)[0]
    uy_grad = torch.autograd.grad(uy.sum(), spatial_coords, create_graph=True)[0]
    uz_grad = torch.autograd.grad(uz.sum(), spatial_coords, create_graph=True)[0]
    p_grad = torch.autograd.grad(p.sum(), spatial_coords, create_graph=True)[0]

    ux_laplacian = torch.autograd.grad(ux_grad[:, :, 0].sum(), spatial_coords, create_graph=True)[0][:, :, 0] + \
                   torch.autograd.grad(ux_grad[:, :, 1].sum(), spatial_coords, create_graph=True)[0][:, :, 1] + \
                   torch.autograd.grad(ux_grad[:, :, 2].sum(), spatial_coords, create_graph=True)[0][:, :, 2]

    uy_laplacian = torch.autograd.grad(uy_grad[:, :, 0].sum(), spatial_coords, create_graph=True)[0][:, :, 0] + \
                   torch.autograd.grad(uy_grad[:, :, 1].sum(), spatial_coords, create_graph=True)[0][:, :, 1] + \
                   torch.autograd.grad(uy_grad[:, :, 2].sum(), spatial_coords, create_graph=True)[0][:, :, 2]

    uz_laplacian = torch.autograd.grad(uz_grad[:, :, 0].sum(), spatial_coords, create_graph=True)[0][:, :, 0] + \
                   torch.autograd.grad(uz_grad[:, :, 1].sum(), spatial_coords, create_graph=True)[0][:, :, 1] + \
                   torch.autograd.grad(uz_grad[:, :, 2].sum(), spatial_coords, create_graph=True)[0][:, :, 2]

    # 计算速度场的拉普拉斯算子
    laplacian_u = torch.stack([ux_laplacian, uy_laplacian, uz_laplacian], dim=2)

    # 计算 Stokes 方程的残差
    residual = mu * laplacian_u - p_grad - f

    # 计算损失
    loss = torch.mean(residual ** 2)

    return loss


########################################### main函数 ###################################################################

# 读入网格点
vertices, elements = read_mesh('../../mesh/stokes/regular_domain.mphtxt')
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

# # 训练模型
for epoch in range(2000):
    start_time = time.time()
    optimizer.zero_grad()
    output = deeponet(f_data, BC_data, spatial_coords)
    output_mag = torch.sqrt(output[:, :, [0]]**2+output[:, :, [1]]**2+output[:, :, [2]]**2)
    data_loss = criterion(output_mag, targets)
    physics_loss = stoke_loss(output, f_data[:, 0, :], spatial_coords)
    loss = data_loss + 0.1 * physics_loss
    loss.backward()
    optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time
    print(
        f'Epoch {epoch}, Data Loss: {data_loss.item()}, Physics Loss: {physics_loss.item()}, Total Loss: {loss.item()}, time: {training_time:.2f} seconds')

    if epoch % 100 == 0:
        torch.save(deeponet.state_dict(), f'pideeponet_model.pth')

model_path = 'pideeponet_model.pth'
deeponet.load_state_dict(torch.load(model_path))

# # 使用模型进行预测
with torch.no_grad():
    output = deeponet(f_data, BC_data, spatial_coords)
    output_mag = torch.sqrt(output[:, :, [0]] ** 2 + output[:, :, [1]] ** 2 + output[:, :, [2]] ** 2)
    loss = criterion(output_mag, targets)
    print(loss.item())
    # save_to_vtk(vertices, elements, output_mag[0, ...].detach().cpu().numpy(), targets[0, ...].detach().cpu().numpy())