import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import meshio
from scipy.interpolate import griddata
import numpy as np
import torch.nn.functional as F

################################################################
# 3d fourier layers
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=3, out_channels=3):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: (x_velocity, y_velocity, pressure) in [0, T)
        input shape: (batchsize, x=64, y=64, t=T, c=3)
        output: (x_velocity, y_velocity, pressure) in [T, 2T)
        output shape: (batchsize, x=64, y=64, t=T, c=3)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic

        self.p = nn.Linear(in_channels + 3,
                           self.width)  # input channel is 6: (x_velocity, y_velocity, z_velocity) + 3 locations (u, v, w, x, y, z)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        # self.mlp1 = MLP(self.width, self.width, self.width)
        # self.mlp2 = MLP(self.width, self.width, self.width)
        # self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        # self.w1 = nn.Conv3d(self.width, self.width, 1)
        # self.w2 = nn.Conv3d(self.width, self.width, 1)
        # self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, out_channels, self.width * 4)  # output channel is 3: (u, v, w)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x1 = self.mlp2(x1)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x1 = self.mlp3(x1)
        # x2 = self.w3(x)
        # x = x1 + x2

        x = x[..., :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)



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


def generate_grid(vertices, n=10):
    # 生成x, y, z方向上的网格点
    x = np.linspace(np.min(vertices[:,0]), np.max(vertices[:,0]), n)
    y = np.linspace(np.min(vertices[:,1]), np.max(vertices[:,1]), n)
    z = np.linspace(np.min(vertices[:,2]), np.max(vertices[:,2]), n)

    # 使用numpy的meshgrid生成网格
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 将网格点展平成一维数组
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # 将网格点组合成一个二维数组，每一行是一个点的坐标
    grid_points = np.vstack((X_flat, Y_flat, Z_flat)).T

    return grid_points



def U_exact(x, case_index):
    u_exact_file = os.path.join('../../../../data/diffusion/case2/test_data', f'res{case_index + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)

    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values



def Force(x, case_index):
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



def save_to_vtk(vertices, xyz, elements, u_pred, u_exact):
    interpolated_u_pred = griddata(points=xyz, values=u_pred.flatten(), xi=vertices, method='nearest')
    interpolated_u_exact = U_exact(vertices, 18)
    error = np.mean((interpolated_u_pred - interpolated_u_exact)**2)
    print('error: ', error)
    mesh = meshio.Mesh(
        points = vertices,
        cells = [("tetra", elements)],
        point_data={"fno_u_pred": interpolated_u_pred, "fno_u_exac": interpolated_u_exact, "fno_u_error": np.abs(interpolated_u_pred - interpolated_u_exact)}
    )
    mesh.write('case19_fno.vtu')

    np.savetxt('case19_fno.txt', np.column_stack((vertices, interpolated_u_pred)), fmt='%.6f')


########################################### main函数 ###################################################################
# 读入网格点
vertices, elements = read_mesh('../../../../mesh/diffusion/pipe_v2_domain.mphtxt')


npoints = 14
xyz = generate_grid(vertices, n=npoints)

f_data = []
BC_data = []
targets = []
targets_ori = []
for case_index in range(15, 20):
    f_data.append(Force(xyz, case_index).reshape((npoints, npoints, npoints, -1)))
    BC_data.append(boundary_v(xyz, case_index).reshape((npoints, npoints, npoints, -1)))
    targets.append(U_exact(xyz, case_index).reshape((npoints, npoints, npoints, -1)))
    targets_ori.append(U_exact(vertices, case_index))

inputs = np.concatenate((np.array(f_data), np.array(BC_data)), axis=4)
targets = np.array(targets)
targets_ori = np.array(targets_ori)

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
inputs = torch.from_numpy(inputs).float().to(device)
targets = torch.from_numpy(targets).float().to(device)
targets_ori = torch.from_numpy(targets_ori).float().to(device)

# 计算 inputs 的均值和标准差
inputs_mean = torch.tensor([4.9620, -4.9146]).float().to(device)
inputs_std = torch.tensor([0.7179, 1.0031]).float().to(device)

# 计算 targets 的均值和标准差
targets_mean = torch.mean(targets, dim=(0, 1, 2, 3))  # 对所有维度求均值
targets_std = torch.std(targets, dim=(0, 1, 2, 3))    # 对所有维度求标准差

inputs = (inputs-inputs_mean)/(inputs_std+1e-12)

modes = 4
width = 6
input_dim = inputs.shape[-1]
output_dim = targets.shape[-1]
# 创建fno
fno = FNO3d(modes, modes, modes, width,
              in_channels=input_dim, out_channels=output_dim).cuda()

model_path = 'fno_model.pth'
fno.load_state_dict(torch.load(model_path))

criterion = nn.MSELoss()
# 使用模型进行预测
with torch.no_grad():
    t1 = time.time()
    output = fno(inputs)
    # output = (output + targets_mean) * targets_std
    loss = criterion(output, targets)
    t2 = time.time()
    print(loss.item(), 'time:', t2-t1)
    save_to_vtk(vertices, xyz, elements, output[-2, ...].detach().cpu().numpy(), targets[-2, ...].detach().cpu().numpy())