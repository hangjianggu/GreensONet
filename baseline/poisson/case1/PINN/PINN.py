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



def poisson_loss(model, x, f, boundary_x, boundary_u):
    x.requires_grad_(True)
    u_pred = model(x)

    # 计算梯度和拉普拉斯算子
    u_grad = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    u_laplacian = torch.autograd.grad(u_grad[:, 0].sum(), x, create_graph=True)[0][:, 0] + \
                  torch.autograd.grad(u_grad[:, 1].sum(), x, create_graph=True)[0][:, 1] + \
                  torch.autograd.grad(u_grad[:, 2].sum(), x, create_graph=True)[0][:, 2]

    # 计算泊松方程的残差
    residual_u = u_laplacian - f

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
    u_exact_file = os.path.join('../../../../data/poisson/case1/test_data', f'res99.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("f", None)
    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values



def boundary_condition(x):
    u_exact_file = os.path.join('../../../../data/poisson/case1/test_data', f'res99.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)
    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values



def U_exact(x):
    u_exact_file = os.path.join('../../../../data/poisson/case1/test_data', f'res99.vtu')
    mesh = meshio.read(u_exact_file)
    points = mesh.points
    u_exact = mesh.point_data.get("u", None)

    interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
    return interpolated_values



# 训练模型
def train_model(model, x, f, boundary_x, boundary_u, epochs=10000, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with open('training_log.txt', 'w') as log_file:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = poisson_loss(model, x, f, boundary_x, boundary_u)
            loss.backward()
            optimizer.step()
            # 打印到控制台
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            # 将输出写入日志文件
            log_file.write(f'Epoch {epoch}, Loss: {loss.item()}\n')

            if epoch % 100 == 0:
                # 保存模型
                torch.save(model.state_dict(), f'pinn_model.pth')



# 主函数
def main():
    # 定义网络结构
    layers = [3, 12, 24, 12, 1]  # 输入层3个神经元，输出层4个神经元
    model = PINN(layers)

    model_path = 'pinn_model.pth'
    model.load_state_dict(torch.load(model_path))

    vertices, elements = read_mesh('../../../../mesh/poisson/heat_sink_v2_domain.mphtxt')
    boundary_x = read_boundary('../../../../mesh/poisson/heat_sink_v2_boundary.mphtxt')

    x = find_interior_points(vertices, boundary_x)
    f = F(x)
    boundary_u = boundary_condition(boundary_x)

    x = torch.tensor(x, dtype=torch.float32)
    f = torch.tensor(f, dtype=torch.float32)
    boundary_x = torch.tensor(boundary_x, dtype=torch.float32)
    boundary_u = torch.tensor(boundary_u, dtype=torch.float32)

    train_model(model, x, f, boundary_x, boundary_u)




if __name__ == "__main__":
    main()