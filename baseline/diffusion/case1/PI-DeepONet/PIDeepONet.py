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


def F(x, case_index):
    u_exact_file = os.path.join('../../../../data/diffusion/case1/train_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    f = mesh.point_data.get("f", None)
    interpolated_values = griddata(points=points, values=f, xi=x, method='nearest')
    return interpolated_values


def boundary_v(x, case_index):
    u_exact_file = os.path.join('../../../../data/diffusion/case1/train_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)
    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values


def U_exact(x, case_index):
    u_exact_file = os.path.join('../../../../data/diffusion/case1/train_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)

    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
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
        return torch.sum(combined_branch_output * trunk_output, dim=2, keepdim=True)


def diffusion_reaction_loss(u_pred, f, x, targets_mean, targets_std, f_data_mean, f_data_std):
    """ Diffusion-Reaction equation loss
    Params:
    -------
    u_pred: predicted solution
    f: source term
    x: input coordinates
    a: diffusion coefficient
    r: reaction coefficient
    """
    # 确保 u_pred 和 x 需要计算梯度
    # u_pred.requires_grad_(True)
    # x.requires_grad_(True)

    u_pred_original = (u_pred+targets_mean)*targets_std
    f_original = (f+f_data_mean)*f_data_std

    # 计算 u 的梯度和拉普拉斯算子
    u_grad = torch.autograd.grad(u_pred_original.sum(), x, create_graph=True, retain_graph=True)[0]
    u_laplacian = torch.autograd.grad(u_grad[:, 0].sum(), x, create_graph=True, retain_graph=True)[0][..., 0] + \
                  torch.autograd.grad(u_grad[:, 1].sum(), x, create_graph=True, retain_graph=True)[0][..., 1] + \
                  torch.autograd.grad(u_grad[:, 2].sum(), x, create_graph=True, retain_graph=True)[0][..., 2]

    a = (x[...,0]-0.5)**2 + (x[...,1]-0.5)**2 + (x[...,2]-0.1)**2
    # 计算 a 的梯度
    a_grad = torch.autograd.grad(a.sum(), x, create_graph=True, retain_graph=True)[0]

    # 计算 (a_x * u_x + a_y * u_y + a_z * u_z)
    a_dot_u_grad = a_grad[..., 0] * u_grad[..., 0] + a_grad[..., 1] * u_grad[..., 1] + a_grad[..., 2] * u_grad[..., 2]
    # 计算 Diffusion-Reaction 方程的残差
    residual_u = -(a * u_laplacian + a_dot_u_grad) + u_pred_original.squeeze() - f_original
    # 总损失
    loss = torch.mean(torch.sqrt(residual_u**2))
    return loss


########################################### main函数 ###################################################################

# 读入网格点
vertices, elements = read_mesh('../../../../mesh/diffusion/regular_domain.mphtxt')
npoints = vertices.shape[0]

spatial_coords = []
f_data = []
BC_data = []
targets = []
for case_index in range(80):
    spatial_coords.append(vertices)
    f_data.append(np.tile(F(vertices, case_index), (npoints, 1)))
    BC_data.append(np.tile(boundary_v(vertices, case_index), (npoints, 1)))
    targets.append(U_exact(vertices, case_index)[..., np.newaxis])

# 将数据转换为 NumPy 数组
spatial_coords = np.array(spatial_coords)
f_data = np.array(f_data)
BC_data = np.array(BC_data)
targets = np.array(targets)

# 计算均值和标准差
spatial_coords_mean = np.mean(spatial_coords, axis=(0, 1, 2))
spatial_coords_std = np.std(spatial_coords, axis=(0, 1, 2))

f_data_mean = np.mean(f_data, axis=(0, 1, 2))
f_data_std = np.std(f_data, axis=(0, 1, 2))

BC_data_mean = np.mean(BC_data, axis=(0, 1, 2))
BC_data_std = np.std(BC_data, axis=(0, 1, 2))

targets_mean = np.mean(targets, axis=(0, 1, 2))
targets_std = np.std(targets, axis=(0, 1, 2))

# 归一化处理
# spatial_coords = (spatial_coords - spatial_coords_mean) / spatial_coords_std
f_data = (f_data - f_data_mean) / (f_data_std+1e-12)
BC_data = (BC_data - BC_data_mean) / (BC_data_std+1e-12)
# targets = (targets - targets_mean) / targets_std

# 转换为 PyTorch 张量
spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32).requires_grad_(True)
f_data = torch.tensor(f_data, dtype=torch.float32).requires_grad_(True)
BC_data = torch.tensor(BC_data, dtype=torch.float32).requires_grad_(True)
targets = torch.tensor(targets, dtype=torch.float32)


# 定义用户自定义的网络层数和每层的维度
branch_layers1 = [npoints, 12, 24, 12, 1]  # 第一个分支网络 ——> f
branch_layers2 = [npoints, 12, 24, 12, 1]  # 第二个分支网络 ——> BC
trunk_layers = [3, 12, 24, 12, 1]    # 主干网络 ——>  spatial_coords

# 创建DeepONet
deeponet = DeepONet(branch_layers1, branch_layers2, trunk_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(deeponet.parameters(), lr=0.001)


best_loss = 999
# # 训练模型
with open('training_log.txt', 'w') as log_file:
    for epoch in range(1000):
        start_time = time.time()
        optimizer.zero_grad()
        output = deeponet(f_data, BC_data, spatial_coords)
        data_loss = criterion((output + targets_mean) * targets_std, targets)
        physics_loss = diffusion_reaction_loss(output, f_data[:, 0, :], spatial_coords, targets_mean, targets_std, f_data_mean, f_data_std)
        loss = data_loss + 0.1 * physics_loss
        loss.backward()
        optimizer.step()
        end_time = time.time()
        training_time = end_time - start_time
        print(f'Epoch {epoch}, Data Loss: {data_loss.item()}, Physics Loss: {physics_loss.item()}, Total Loss: {loss.item()}, time: {training_time:.2f} seconds')
        # 将输出写入日志文件
        log_file.write(f'Epoch {epoch}, Data Loss: {data_loss.item()}, Physics Loss: {physics_loss.item()}, Total Loss: {loss.item()}, time: {training_time:.2f} seconds\n')

        if epoch % 100 == 0:
            if loss<best_loss:
                best_loss = loss
                torch.save(deeponet.state_dict(), f'pideeponet_model.pth')




# model_path = 'pideeponet_model.pth'
# deeponet.load_state_dict(torch.load(model_path))
#
# # # 使用模型进行预测
# with torch.no_grad():
#     output = deeponet(f_data, BC_data, spatial_coords)
#     output_mag = torch.sqrt(output[:, :, [0]] ** 2 + output[:, :, [1]] ** 2 + output[:, :, [2]] ** 2)
#     loss = criterion(output_mag, targets)
#     print(loss.item())
    # save_to_vtk(vertices, elements, output_mag[0, ...].detach().cpu().numpy(), targets[0, ...].detach().cpu().numpy())