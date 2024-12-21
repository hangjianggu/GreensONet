import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import meshio

# 定义物理常数
mu = 1.0/100  # 动力粘度



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



def stokes_loss(model, x, f, boundary_x, boundary_u):
    x.requires_grad_(True)
    u_pred = model(x)
    u_x, u_y, u_z, p = u_pred[:, 0], u_pred[:, 1], u_pred[:, 2], u_pred[:, 3]

    # 计算梯度和拉普拉斯算子
    u_x_grad = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y_grad = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0]
    u_z_grad = torch.autograd.grad(u_z.sum(), x, create_graph=True)[0]
    p_grad = torch.autograd.grad(p.sum(), x, create_graph=True)[0]

    u_x_laplacian = torch.autograd.grad(u_x_grad[:, 0].sum(), x, create_graph=True)[0][:, 0] + \
                    torch.autograd.grad(u_x_grad[:, 1].sum(), x, create_graph=True)[0][:, 1] + \
                    torch.autograd.grad(u_x_grad[:, 2].sum(), x, create_graph=True)[0][:, 2]

    u_y_laplacian = torch.autograd.grad(u_y_grad[:, 0].sum(), x, create_graph=True)[0][:, 0] + \
                    torch.autograd.grad(u_y_grad[:, 1].sum(), x, create_graph=True)[0][:, 1] + \
                    torch.autograd.grad(u_y_grad[:, 2].sum(), x, create_graph=True)[0][:, 2]

    u_z_laplacian = torch.autograd.grad(u_z_grad[:, 0].sum(), x, create_graph=True)[0][:, 0] + \
                    torch.autograd.grad(u_z_grad[:, 1].sum(), x, create_graph=True)[0][:, 1] + \
                    torch.autograd.grad(u_z_grad[:, 2].sum(), x, create_graph=True)[0][:, 2]

    # 计算Stokes方程的残差
    residual_u_x = mu * u_x_laplacian - p_grad[:, 0] - f[:, 0]
    residual_u_y = mu * u_y_laplacian - p_grad[:, 1] - f[:, 1]
    residual_u_z = mu * u_z_laplacian - p_grad[:, 2] - f[:, 2]

    # 计算连续性方程的残差
    div_u = u_x_grad[:, 0] + u_y_grad[:, 1] + u_z_grad[:, 2]

    # 计算边界条件的损失
    boundary_u_pred = model(boundary_x)
    boundary_loss = torch.mean((boundary_u_pred[:, :3] - boundary_u)**2)

    # 总损失
    loss = torch.mean(residual_u_x**2) + torch.mean(residual_u_y**2) + torch.mean(residual_u_z**2) + torch.mean(div_u**2) + boundary_loss
    print('boundary_loss', boundary_loss.item())
    print('pde_loss', loss.item()-boundary_loss.item())
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
    f_file = os.path.join('../../data/stokes/test_data', f'res16.vtu')
    mesh = meshio.read(f_file)
    points = mesh.points
    fx = mesh.point_data.get("fx", None)
    interpolated_fx = griddata(points=points, values=fx, xi=x, method='nearest')
    fy = mesh.point_data.get("fy", None)
    interpolated_fy = griddata(points=points, values=fy, xi=x, method='nearest')
    fz = mesh.point_data.get("fz", None)
    interpolated_fz = griddata(points=points, values=fz, xi=x, method='nearest')
    res = np.column_stack((interpolated_fx, interpolated_fy, interpolated_fz))
    return res



def boundary_condition(x):
    result = np.zeros_like(x)
    mask = x[:, 2] == 1
    result[mask] = [1, 0, 0]
    return result



def boundary_v_mag(x):
    result = np.zeros_like(x[:, 0])
    mask = x[:, 2] == 1
    result[mask] = 1
    return result


def U_exact(x):
    u_exact_file = os.path.join('../../data/stokes/test_data', f'res16.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)
    v_exact = mesh.point_data.get("v", None)
    w_exact = mesh.point_data.get("w", None)

    U = (u_exact**2 + v_exact**2 + w_exact**2)**0.5
    interpolated_values = griddata(points=points, values=U, xi=x, method='nearest')
    return interpolated_values



def save_to_vtk(vertices, elements, coords, u_pred, boundary_u_pred, u_exact):
    sorted_data = []
    for node in vertices:
        if any(np.allclose(node, coord) for coord in coords):
            index = np.where(np.all(np.isclose(coords, node), axis=1))[0][0]
            sorted_data.append(
                [node[0], node[1], node[2], u_pred[index], u_exact[index], np.abs(u_pred[index] - u_exact[index])])
        else:
            boundary_point_value = boundary_v_mag(node[np.newaxis, ...])
            sorted_data.append([node[0], node[1], node[2], boundary_point_value[0], boundary_point_value[0], 0])

    sorted_data = np.array(sorted_data)
    mesh = meshio.Mesh(
        points = vertices,
        cells = [("tetra", elements)],
        point_data={"pinn_u_pred": sorted_data[:, 3], "pinn_u_exac": sorted_data[:, 4], "pinn_u_error": sorted_data[:, 5]}
    )
    mesh.write('case16_pinn.vtu')



def save_to_vtk2(vertices, elements, coords, u_pred, boundary_x, boundary_u_pred):
    sorted_data = []
    for node in vertices:
        if any(np.allclose(node, coord) for coord in coords):
            index = np.where(np.all(np.isclose(coords, node), axis=1))[0][0]
            sorted_data.append(np.concatenate((node[np.newaxis, :], u_pred[index, :3][np.newaxis, :]), axis=1))
        else:
            index = np.where(np.all(np.isclose(boundary_x, node), axis=1))[0][0]
            sorted_data.append(np.concatenate((node[np.newaxis, :], boundary_u_pred[index, :3][np.newaxis, :]), axis=1))

    sorted_data = np.array(sorted_data).squeeze(1)
    mesh = meshio.Mesh(
        points = vertices,
        cells = [("tetra", elements)],
        point_data={"pinn_u": sorted_data[:, 3], "pinn_v": sorted_data[:, 4], "pinn_w": sorted_data[:, 5]}
    )
    mesh.write('case16_pinn_streamline.vtu')



# 训练模型
def train_model(model, x, f, boundary_x, boundary_u, epochs=2000, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = stokes_loss(model, x, f, boundary_x, boundary_u)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            torch.save(model.state_dict(), f'pinn_model.pth')



# 主函数
def main():
    # 定义网络结构
    layers = [3, 12, 24, 12, 4]  # 输入层3个神经元，输出层4个神经元
    model = PINN(layers)
    vertices, elements = read_mesh('../../mesh/stokes/regular_domain.mphtxt')
    boundary_x = read_boundary('../../mesh/stokes/regular_boundary.mphtxt')

    x = find_interior_points(vertices, boundary_x)
    f = F(x)
    boundary_u = boundary_condition(boundary_x)

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

    pde_loss = stokes_loss(model, x, f, boundary_x, boundary_u)
    # print('pde_loss:', pde_loss.item())

    x = x.detach().cpu().numpy()
    u_exact = U_exact(x)
    u_pred_ = (u_pred[:, 0] ** 2 + u_pred[:, 1] ** 2 + u_pred[:, 2] ** 2) ** 0.5

    criterion = nn.MSELoss()
    loss = criterion(torch.tensor(u_exact, dtype=torch.float32), u_pred_)
    print('loss:', loss.item())

    u_pred_ = u_pred_.detach().cpu().numpy()
    boundary_u_pred = boundary_u_pred.detach().cpu().numpy()


    # u_exact_ = (u_exact[:, 0]**2 + u_exact[:, 1]**2 + u_exact[:, 2]**2)**0.5
    boundary_u_pred_ = (boundary_u_pred[:, 0]**2 + boundary_u_pred[:, 1]**2 + boundary_u_pred[:, 2]**2)**0.5
    save_to_vtk2(vertices, elements, x, u_pred.detach().cpu().numpy(), boundary_x.detach().cpu().numpy(), boundary_u_pred)



if __name__ == "__main__":
    main()