#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import shutil
import os


class Mesh:
    def __init__(self, meshfile, boundaryfile, domain, blocks_num, ngs_boundary=6, ngs_interior=5):
        self.vertices, self.elements = self.read_meshfile(meshfile)
        self.boundary_vertices, self.boundary_faces, self.boundary_type = self.read_boundaryfile(boundaryfile)
        self.interior_vertices = self.find_interior_points(self.vertices, self.boundary_vertices)
        self.domain = domain
        self.blocks_num = blocks_num
        self.ngs_interior = ngs_interior
        self.ngs_boundary = ngs_boundary

        self.gs_boundary = {1: {'wts': [1],
                                'pts': [[1/3, 1/3, 1/3]]
                                },
                            3: {'wts': [1/3, 1/3, 1/3],
                                'pts': [[2/3, 1/6, 1/6],
                                        [1/6, 2/3, 1/6],
                                        [1/6, 1/6, 2/3]]
                                },
                            6: {'wts': [0.109951743655322, 0.109951743655322, 0.109951743655322, 0.223381589678011, 0.223381589678011, 0.223381589678011],
                                'pts': [[0.816847572980459, 0.091576213509661, 0.091576213509661],
                                        [0.091576213509661, 0.816847572980459, 0.091576213509661],
                                        [0.091576213509661, 0.091576213509661, 0.816847572980459],
                                        [0.10810301816807, 0.445948490915965, 0.445948490915965],
                                        [0.445948490915965, 0.10810301816807, 0.445948490915965],
                                        [0.445948490915965, 0.445948490915965, 0.10810301816807]]
                                }
                            }

        self.gs_interior = {1: {'wts': [1],
                                'pts': [[1/4, 1/4, 1/4, 1/4]]
                                },
                            4: {'wts': [1/4, 1/4, 1/4, 1/4],
                                'pts':
                                [[0.585410196624970, 0.138196601125010, 0.138196601125010, 0.138196601125010],
                                [0.138196601125010, 0.585410196624970, 0.138196601125010, 0.138196601125010],
                                [0.138196601125010, 0.138196601125010, 0.585410196624970, 0.138196601125010],
                                [0.138196601125010, 0.138196601125010, 0.138196601125010, 0.585410196624970]]
                                },
                            5: {'wts': [-0.800000000000000,	0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000],
                                'pts':
                                [[0.250000000000000, 0.250000000000000, 0.250000000000000, 0.250000000000000],
                                [0.500000000000000, 0.166666666666667, 0.166666666666667, 0.166666666666667],
                                [0.166666666666667, 0.500000000000000, 0.166666666666667, 0.166666666666667],
                                [0.166666666666667, 0.166666666666667, 0.500000000000000, 0.166666666666667],
                                [0.166666666666667, 0.166666666666667, 0.166666666666667, 0.500000000000000]]
                                }
                            }

        self.blocks = self._blocks_info()
        self.X_interior = self._get_X_interior()
        self.X_boundary = self._get_X_boundary()
        # self.Z_boundary = self._get_Z_boundary()
        self.z_interior = self._get_Z_interior()
        self.z_blocks = self._z_blocks()

    def read_meshfile(self, meshfile):
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
        return np.array(vertices), np.array(tetrahedrons)

    def read_boundaryfile(self, boundaryfile):
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

            boundary_faces = []
            for _ in range(boundary_element_count):
                line = next(file).strip()
                values = line.split()
                ele = list(map(int, values))
                boundary_faces.append(ele)

            boundary_type = []
            # boundary_type.append(0)
            for _ in range(boundary_element_count):
                line = next(file).strip()
                values = line.split()
                ele = list(map(int, values))
                boundary_type.append(ele)

        return np.array(boundary_vertices), np.array(boundary_faces), np.array(boundary_type)

    def find_interior_points(self, vertices, boundary_vertices):
        interior_vertices = np.empty((0, 3))
        for p in vertices:
            if p.size == 0 or not np.any(np.isclose(boundary_vertices, p, atol=1e-15).all(axis=1)):
                interior_vertices = np.vstack([interior_vertices, p])
        return interior_vertices

    def calc_area(self, vertices):
        ### 向量的叉乘的模长是面积
        a = vertices[1] - vertices[0]
        b = vertices[2] - vertices[0]
        c = np.cross(a, b)
        return 0.5 * np.abs(np.linalg.norm(c))

    def calc_volume(self, vertices):
        # a = vertices[1] - vertices[0]
        # b = vertices[2] - vertices[0]
        # c = vertices[3] - vertices[0]
        # cross_product = np.cross(a, b)
        # volume = np.abs(np.dot(cross_product, c)) / 6

        matrix = np.array([vertices[1] - vertices[0], vertices[2] - vertices[0], vertices[3] - vertices[0]])
        # 计算行列式并除以 6
        volume = np.abs(np.linalg.det(matrix)) / 6
        # 返回体积
        return volume

    def _blocks_info(self):
        x0_min, x0_max, x1_min, x1_max, x2_min, x2_max = self.domain
        x0 = np.linspace(x0_min, x0_max, self.blocks_num[0] + 1)
        x1 = np.linspace(x1_min, x1_max, self.blocks_num[1] + 1)
        x2 = np.linspace(x2_min, x2_max, self.blocks_num[2] + 1)
        return np.array(
            [(x0[i], x0[i + 1], x1[j], x1[j + 1], x2[k], x2[k + 1]) for i in range(x0.shape[0] - 1) for j in
             range(x1.shape[0] - 1) for k in range(x2.shape[0] - 1)])

    def in_subdomain(self, z_interior, subdomain):
        mask = np.all([z_interior['coord'][:, 0] >= subdomain[0],
                       z_interior['coord'][:, 0] <= subdomain[1],
                       z_interior['coord'][:, 1] >= subdomain[2],
                       z_interior['coord'][:, 1] <= subdomain[3],
                       z_interior['coord'][:, 2] >= subdomain[4],
                       z_interior['coord'][:, 2] <= subdomain[5],
                       ], axis=0)
        return {'coord': z_interior['coord'][mask]}

    #### 计算边界线上的积分点和积分权重
    def _get_X_boundary(self):
        Normal = []
        Coord = []
        Wts = []
        for fh in self.boundary_faces:
            vertices = np.vstack([self.boundary_vertices[p - 1] for p in fh])
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[1]
            normal = np.cross(v1, v2)
            # 检查范数是否为零
            norm = np.linalg.norm(normal)
            if norm == 0:
                # 处理零范数情况，例如设置 normal 为零向量
                normal = np.zeros_like(normal)
            else:
                # 归一化法向量
                normal = normal / norm
            area = self.calc_area(vertices)
            pts = self.gs_boundary[self.ngs_boundary]['pts'] @ vertices
            # [np.array([1 - pt[0] - pt[1], pt[0], pt[1]] @ vertices) for pt in
            #        self.gs_boundary[self.ngs_boundary]['pts']]
            wts = area * np.array(self.gs_boundary[self.ngs_boundary]['wts'])
            Normal.append(normal)
            Coord.append(pts)
            Wts.append(wts)

        COORD = [np.array([Coord[fh_inx][k] for fh_inx in range(len(self.boundary_faces))]) for k in range(self.ngs_boundary)]
        WTS = [np.array([Wts[fh_inx][k] for fh_inx in range(len(self.boundary_faces))]) for k in range(self.ngs_boundary)]
        NORMAL = np.array([Normal[fh_inx] for fh_inx in range(len(self.boundary_faces))])
        boundary_type = np.array([self.boundary_type[fh_inx] for fh_inx in range(len(self.boundary_faces))])
        return {'coord': COORD, 'normal': NORMAL, 'wts': WTS, 'boundary_type': boundary_type}

    def _get_X_interior(self):
        Coord = []
        Wts = []
        for ele in self.elements:
            ### 遍历该四面体的各个顶点
            vertices = np.vstack([self.vertices[fh-1] for fh in ele])
            volume = self.calc_volume(vertices)
            ### 计算该四面体的内部积分点
            pts = self.gs_interior[self.ngs_interior]['pts'] @ vertices
            # np.array([np.array([pt[0], pt[1], pt[2], 1 - pt[0] - pt[1] - pt[2]] @ vertices[:, :3]) for pt in
            #        self.gs_interior[self.ngs_interior]['pts']])
            wts = volume * np.array(self.gs_interior[self.ngs_interior]['wts'])
            ### 积分坐标点
            Coord.append(pts)
            ### 积分权重
            Wts.append(wts)

        COORD = [np.array([Coord[ele_inx][k] for ele_inx in range(len(self.elements))]) for k in range(self.ngs_interior)]
        WTS = [np.array([Wts[ele_inx][k] for ele_inx in range(len(self.elements))]) for k in range(self.ngs_interior)]
        return {'coord': COORD, 'wts': WTS}

    def _get_Z_interior(self):
        return {'coord': self.interior_vertices}

    ### 实际上没有用到z_boundary
    # def _get_Z_boundary(self):
    #     return {'coord': self.boundary_vertices}

    def _z_blocks(self):
        z_blocks = []
        for K in range(len(self.blocks)):
            block = self.blocks[K]
            z_part = self.in_subdomain(self.z_interior, block)
            if z_part is not None:
                z_blocks.append(z_part)
            else:
                z_blocks.append(None)
        return z_blocks




def show_image(x, u, elems):
    """
    show image using X and u

    params:
    ======
    x: torch tensor with device cuda and shape (:, 4)
       it should be transfered to cpu and then numpy ndarray
    u: torch tensor with device cuda and shape (:, 1)
       it should be firstly detached and then transfered to cpu and finally to numpy ndarray
    """    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    tri = Triangulation(x[:, 0], x[:, 1], elems)
    surf = ax.plot_trisurf(tri, u[:, 0], cmap=plt.cm.Spectral)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$G$')
    # Add a color bar which map values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.abs(x, y)



def save_checkpoints(k, state, is_best=None, save_dir=None):
    checkpoint = os.path.join(save_dir, f'checkpoint_block{k}.pkl')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, f'block{k}.pkl')
        shutil.copyfile(checkpoint, best_model) 


if __name__ == '__main__':
    x = np.random.rand(5, 2)
    y = np.random.rand(3, 2)

