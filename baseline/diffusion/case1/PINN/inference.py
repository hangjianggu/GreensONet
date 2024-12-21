import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import meshio



class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x



def diffusion_reaction_loss(model, x, f, boundary_x, boundary_u):
    x.requires_grad_(True)
    u_pred = model(x)

    # 计算梯度和拉普拉斯算子
    u_grad = torch.autograd.grad(u_pred.sum(), x, create_graph=True, retain_graph=True)[0]
    u_laplacian = torch.autograd.grad(u_grad[:, 0].sum(), x, create_graph=True, retain_graph=True)[0][..., 0] + \
                  torch.autograd.grad(u_grad[:, 1].sum(), x, create_graph=True, retain_graph=True)[0][..., 1] + \
                  torch.autograd.grad(u_grad[:, 2].sum(), x, create_graph=True, retain_graph=True)[0][..., 2]

    a = (x[...,0]-0.5)**2 + (x[...,1]-0.5)**2 + (x[...,2]-0.1)**2
    # 计算 a 的梯度
    a_grad = torch.autograd.grad(a.sum(), x, create_graph=True, retain_graph=True)[0]

    # 计算 (a_x * u_x + a_y * u_y + a_z * u_z)
    a_dot_u_grad = a_grad[..., 0] * u_grad[..., 0] + a_grad[..., 1] * u_grad[..., 1] + a_grad[..., 2] * u_grad[..., 2]

    # 计算pde残差
    residual_u = -(a * u_laplacian + a_dot_u_grad) + u_pred.squeeze() - f

    # 计算边界条件的损失
    boundary_u_pred = model(boundary_x)
    boundary_loss = torch.mean((boundary_u_pred - boundary_u)**2)

    # 总损失
    loss = torch.mean(residual_u**2) + boundary_loss

    return loss



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



def find_interior_points(vertices, boundary_vertices):
    interior_vertices = np.empty((0, 3))
    for p in vertices:
        if p.size == 0 or not np.any(np.isclose(boundary_vertices, p, atol=1e-15).all(axis=1)):
            interior_vertices = np.vstack([interior_vertices, p])
    return interior_vertices



def read_boundary(boundaryfile):
    with open(boundaryfile, 'r') as file:
        line1 = next(file).strip()
        line2 = next(file).strip()
        boundary_node_count_str, boundary_element_count_str, _, _ = line2.split(',')
        boundary_node_count = int(boundary_node_count_str.split('=')[1])
        boundary_element_count = int(boundary_element_count_str.split('=')[1])

        boundary_vertices = []
        for _ in range(boundary_node_count):
            line = next(file).strip()
            values = line.split()
            x, y, z = map(float, values)
            boundary_vertices.append((x, y, z))
    boundary_vertices = np.array(boundary_vertices)

    return boundary_vertices



def F(x):
    u_exact_file = os.path.join('../../../../data/diffusion/case1/test_data', f'res{18 + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    f = mesh.point_data.get("f", None)
    interpolated_values = griddata(points=points, values=f, xi=x, method='nearest')
    return interpolated_values


def boundary_v(x):
    u_exact_file = os.path.join('../../../../data/diffusion/case1/test_data', f'res{18 + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)
    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values


def U_exact(x):
    u_exact_file = os.path.join('../../../../data/diffusion/case1/test_data', f'res{18 + 1}.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)
    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values



def save_to_vtk(vertices, elements, coords, boundary_x, u_pred, boundary_u_pred, u_exact, boundary_u):
    sorted_data = []
    error = np.abs(np.abs(u_pred) - np.abs(u_exact))
    error_boundary = np.abs(np.abs(boundary_u_pred)-np.abs(boundary_u))
    for node in vertices:
        if any(np.allclose(node, coord) for coord in coords):
            index = np.where(np.all(np.isclose(coords, node), axis=1))[0][0]
            sorted_data.append(np.concatenate((node[np.newaxis, :], u_pred[index][..., np.newaxis], u_exact[index][..., np.newaxis], error[index][..., np.newaxis]), axis=1))
        else:
            index = np.where(np.all(np.isclose(boundary_x, node), axis=1))[0][0]
            sorted_data.append(np.concatenate((node[np.newaxis, :], boundary_u_pred[index][..., np.newaxis], boundary_u[index][..., np.newaxis], error_boundary[index][..., np.newaxis]), axis=1))

    sorted_data = np.array(sorted_data).squeeze(1)
    mesh = meshio.Mesh(
        points = vertices,
        cells = [("tetra", elements)],
        point_data={"pinn_u_pred": sorted_data[:, 3], "pinn_u_exac": sorted_data[:, 4], "pinn_u_error": sorted_data[:, 5]}
    )
    mesh.write('case19_pinn.vtu')

    np.savetxt('case19_pinn.txt', np.column_stack((vertices, sorted_data[:, 3])), fmt='%.6f')



# 主函数
def main():
    # 定义网络结构
    layers = [3, 12, 12, 12, 1]  # 输入层3个神经元，输出层4个神经元
    model = PINN(layers)
    vertices, elements = read_mesh('../../../../mesh/diffusion/regular_domain.mphtxt')
    boundary_x = read_boundary('../../../../mesh/diffusion/regular_boundary.mphtxt')

    x = find_interior_points(vertices, boundary_x)
    f = F(x)
    boundary_u = boundary_v(boundary_x)

    x = torch.tensor(x, dtype=torch.float32)
    f = torch.tensor(f, dtype=torch.float32)
    boundary_x = torch.tensor(boundary_x, dtype=torch.float32)
    boundary_u = torch.tensor(boundary_u, dtype=torch.float32)

    # train_model(model, x, f, boundary_x, boundary_u)
    model_path = 'pinn_model.pth'
    model.load_state_dict(torch.load(model_path))

    # # 可视化结果
    model.eval()
    u_pred = model(x)
    boundary_u_pred = model(boundary_x)

    pde_loss = diffusion_reaction_loss(model, x, f, boundary_x, boundary_u)
    print('pde_loss:', pde_loss.item())

    x = x.detach().cpu().numpy()
    u_exact = U_exact(x)[..., np.newaxis]

    criterion = nn.MSELoss()
    loss = (criterion(torch.tensor(u_exact, dtype=torch.float32), u_pred) + criterion(boundary_u_pred, boundary_u))/2
    print('loss:', loss.item())

    save_to_vtk(vertices, elements, x, boundary_x.detach().cpu().numpy(), u_pred.detach().cpu().numpy(), boundary_u_pred.detach().cpu().numpy(), u_exact, boundary_u.detach().cpu().numpy()[..., np.newaxis])



if __name__ == "__main__":
    main()