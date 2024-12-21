import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import meshio
from scipy.interpolate import griddata
import numpy as np


def F(x):
    u_exact_file = os.path.join('../../../../data/diffusion/case2/test_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    f = mesh.point_data.get("f", None)
    interpolated_values = griddata(points=points, values=f, xi=x, method='nearest')
    return interpolated_values


def boundary_v(x, case_index):
    u_exact_file = os.path.join('../../../../data/diffusion/case2/test_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)
    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values


def U_exact(x, case_index):
    u_exact_file = os.path.join('../../../../data/diffusion/case2/test_data', f'res{case_index + 1}.vtu')
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
        point_data={"deeponet_u_pred": u_pred, "deeponet_u_exac": u_exact, "deeponet_u_error": np.abs(u_pred - u_exact)}
    )
    mesh.write('case19_deeponet.vtu')

    np.savetxt('case19_deeponet.txt', np.column_stack((vertices, u_pred)), fmt='%.6f')


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


########################################### main函数 ###################################################################

# 读入网格点
vertices, elements = read_mesh('../../../../mesh/diffusion/pipe_v2_domain.mphtxt')
npoints = vertices.shape[0]

spatial_coords = []
f_data = []
BC_data = []
targets = []
for case_index in range(18, 20):
    spatial_coords.append(vertices)
    f_data.append(np.tile(F(vertices), (npoints, 1)))
    BC_data.append(np.tile(boundary_v(vertices, case_index), (npoints, 1)))
    targets.append(U_exact(vertices, case_index)[..., np.newaxis])

# 将数据转换为 NumPy 数组
spatial_coords = np.array(spatial_coords)
f_data = np.array(f_data)
BC_data = np.array(BC_data)
targets = np.array(targets)

# 计算均值和标准差
spatial_coords_mean = np.array([0.080198, 0.478339, 0.080727])
spatial_coords_std = np.array([0.04329757, 0.24127041, 0.04305752])
f_data_mean = 4.955388963630654
f_data_std = 0.720049732226556
BC_data_mean = -4.923893111164164
BC_data_std = 0.9872762289684819
targets_mean = -4.92389311
targets_std = 0.98727623

# 归一化处理
spatial_coords = (spatial_coords - spatial_coords_mean) / (spatial_coords_std+1e-12)
f_data = (f_data - f_data_mean) / (f_data_std+1e-12)
BC_data = (BC_data - BC_data_mean) / (BC_data_std+1e-12)

spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32)
f_data = torch.tensor(f_data, dtype=torch.float32)
BC_data = torch.tensor(BC_data, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

targets_mean = torch.tensor(targets_mean, dtype=torch.float32)
targets_std = torch.tensor(targets_std, dtype=torch.float32)


# 定义用户自定义的网络层数和每层的维度
branch_layers1 = [npoints, 24, 24, 24, 4]  # 第一个分支网络 ——> f
branch_layers2 = [npoints, 24, 24, 24, 4]  # 第二个分支网络 ——> BC
trunk_layers = [3, 24, 24, 24, 4]    # 主干网络 ——>  spatial_coords

# 创建DeepONet
deeponet = DeepONet(branch_layers1, branch_layers2, trunk_layers)

# 定义损失函数
criterion = nn.MSELoss()

model_path = 'deeponet_model.pth'
deeponet.load_state_dict(torch.load(model_path))


# 使用模型进行预测
with torch.no_grad():
    t1 = time.time()
    output = deeponet(f_data, BC_data, spatial_coords)
    output = (output + targets_mean) * targets_std
    loss = criterion(output, targets)
    t2 = time.time()
    print('loss:', loss.item(), ' time:',t2-t1)
    save_to_vtk(vertices, elements, output[-2, ...].detach().cpu().numpy(), targets[-2, ...].detach().cpu().numpy())