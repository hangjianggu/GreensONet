#!/usr/bin/env python
import os
import numpy as np
# from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import meshio
import torch


class Stokes(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def u_exact(self, x, case_index):
        u_exact_file = os.path.join(self.data_folder, f'res{case_index+1}.vtu')
        mesh = meshio.read(u_exact_file)
        points = mesh.points
        u_exact = mesh.point_data.get("v", None)
        interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
        return interpolated_values[..., np.newaxis]


    def f(self, x, case_index, n_input, device):
        """right-hand term
        Params:
        -------
        x: ndarrays of float with shape (n, 3)
        """
        x = x.detach().cpu().numpy()
        f_file = os.path.join(self.data_folder, f'res{case_index+1}.vtu')
        mesh = meshio.read(f_file)
        points = mesh.points
        if n_input == 0:
            fx = mesh.point_data.get("fx", None)
            interpolated_values = griddata(points=points, values=fx, xi=x, method='nearest')
        elif n_input == 1:
            fy = mesh.point_data.get("fy", None)
            interpolated_values = griddata(points=points, values=fy, xi=x, method='nearest')
        else:
            fz = mesh.point_data.get("fz", None)
            interpolated_values = griddata(points=points, values=fz, xi=x, method='nearest')
        res = torch.from_numpy(interpolated_values[..., np.newaxis]).float().to(device)
        return res

    def a(self, x, device):
        x = x.detach().cpu().numpy()
        res = np.ones_like(x[:, [0]])/100
        res = torch.from_numpy(res).float().to(device)
        return res


    def g(self, x, boundary_type, case_index, n_input, device):
        """ Dirichlet boundary condition
        Params:
        -------
        x: ndarrays of float with shape (n, 3)
        boundary_type: ndarrays of int with shape (n, 1)
        """
        x = x.detach().cpu().numpy()
        boundary_type = boundary_type.detach().cpu().numpy()
        if x.shape[0] != boundary_type.shape[0]:
            raise ValueError("The number of points in x and boundary_type must be the same.")

        result = np.zeros_like(x[:, [0]])
        if n_input == 0:
            mask = boundary_type == 3
            result[mask.flatten()] = 1

        result = torch.from_numpy(result).float().to(device)
        return result




    
if __name__ == '__main__':
    problem = Stokes()
    print(problem)

    # problem = DiffusionReaction(case=1)
    # print(problem)

    
