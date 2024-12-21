#!/usr/bin/env python
import os
import numpy as np
# from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import meshio
import torch


class DiffusionReaction(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def u_exact(self, x, case_index):
        u_exact_file = os.path.join(self.data_folder, f'res{case_index+1}.vtu')
        mesh = meshio.read(u_exact_file)
        points = mesh.points
        u_exact = mesh.point_data.get("u", None)
        interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')
        return interpolated_values[..., np.newaxis]


    def f(self, x, case_index, n_input, device):
        """right-hand term
        Params:
        -------
        x: ndarrays of float with shape (n, 3)
        """
        x = x.detach().cpu().numpy()

        u_exact_file = os.path.join(self.data_folder, f'res{case_index+1}.vtu')
        mesh = meshio.read(u_exact_file)
        points = mesh.points
        u_exact = mesh.point_data.get("f", None)
        interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')

        result = torch.from_numpy(interpolated_values).float().to(device)

        return result


    def a(self, x, case_index, device):
        # res = (x[:,0]-0.5)**2 + (x[:,1]-0.5)**2 + x[:,2]**2
        # res = -1*torch.ones_like(x[:, 0])
        res = 2e-2 * (x[:,0] > 0.08) + 1e-2 * (x[:,0] <= 0.08)
        return res


    def g(self, x, boundary_type, case_index, n_input, device):
        """ Dirichlet boundary condition
        Params:
        -------
        x: ndarrays of float with shape (n, 3)
        boundary_type: ndarrays of int with shape (n, 1)
        """
        x = x.detach().cpu().numpy()

        u_exact_file = os.path.join(self.data_folder, f'res{case_index+1}.vtu')
        mesh = meshio.read(u_exact_file)
        points = mesh.points
        u_exact = mesh.point_data.get("u", None)
        interpolated_values = griddata(points=points, values=u_exact, xi=x, method='nearest')

        result = torch.from_numpy(interpolated_values).float().to(device)

        return result


                


    
