#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import time
import os
from problem import Poisson
from model import Net_Integral
from options import Options
from utils import Mesh, save_checkpoints
import numpy as np
import meshio

class Tester(object):
    def __init__(self, args):
        self.args         = args
        self.device       = args.device
        self.cuda_index   = args.cuda_index
        self.pde_case     = args.pde_case
        # self.geo_type     = args.geo_type
        self.problem      = args.problem

        # Trainset
        self.domain       = args.domain
        self.blocks_num   = args.blocks_num
        self.ngs_boundary = args.ngs_boundary
        self.ngs_interior = args.ngs_interior
        self.test_samples = args.test_samples

        meshfile, boundaryfile = self.get_mesh_path(args.mesh_path, args.boundary_mesh_path)
        self.mesh = Mesh(meshfile, boundaryfile, self.domain, self.blocks_num, ngs_boundary = self.ngs_boundary, ngs_interior = self.ngs_interior)

        # Criterion
        self.criterion = nn.MSELoss()

        # HyperParameters setting
        self.model_path = args.model_path

        # Learning rate
        self.result = {'coord':[], 'u_pred':[], 'u_exac':[], 'error':[]}

    def get_mesh_path(self, mesh_path, boundary_mesh_path):
        return mesh_path, boundary_mesh_path


    def calculate(self):
        print(self.device)
        if not isinstance(self.mesh.X_boundary['normal'], torch.Tensor):
            self.mesh.X_boundary['normal'] = torch.from_numpy(self.mesh.X_boundary['normal']).float().to(self.device)
            self.mesh.X_boundary['boundary_type'] = torch.from_numpy(self.mesh.X_boundary['boundary_type']).float().to(
                self.device)
            for k in range(self.ngs_interior):
                self.mesh.X_interior['coord'][k] = torch.from_numpy(self.mesh.X_interior['coord'][k]).float().to(self.device)
                self.mesh.X_interior['wts'][k] = torch.from_numpy(self.mesh.X_interior['wts'][k]).float().to(self.device)
            for k in range(self.ngs_boundary):
                self.mesh.X_boundary['coord'][k] = torch.from_numpy(self.mesh.X_boundary['coord'][k]).float().to(self.device).requires_grad_(True)
                self.mesh.X_boundary['wts'][k] = torch.from_numpy(self.mesh.X_boundary['wts'][k]).float().to(self.device).requires_grad_(True)

        loss = 0
        for K in range(len(self.mesh.blocks)):
            if self.mesh.z_blocks[K] is not None and len(self.mesh.z_blocks[K])>0:
                if not isinstance(self.mesh.z_blocks[K]['coord'], torch.Tensor):
                    self.mesh.z_blocks[K]['coord'] = torch.from_numpy(self.mesh.z_blocks[K]['coord']).float().to(self.device)
                loss += self.calculate_block(K)
            else:
                print(f'Block #{K} is empty!!!')
        print(f'total loss is {loss/len(self.mesh.blocks)}')


    def calculate_block(self, k):
        # Networks
        self.net_pde      = Net_Integral(args.layers, args.shape, self.problem, self.device, self.ngs_boundary, self.ngs_interior)

        loss = 0
        for case_index in range(98, 99):
            if os.path.isfile(self.model_path):
                print(f'Resuming training, loading {self.model_path} ...')
                # self.checkpoint = torch.load(resume_path)
                for i in range(args.shape[0]):
                    for j in range(args.shape[1]):
                        self.net_pde.G = torch.load(self.model_path, map_location=self.device)

            for i in range(args.shape[0]):
                for j in range(args.shape[1]):
                    self.net_pde.G[i][j].eval()
                    self.net_pde.G[i][j].to(self.device)

            tt = time.time()
            u_pred = self.net_pde(self.mesh.X_interior, self.mesh.X_boundary, self.mesh.z_blocks[k], case_index)
            coord = self.mesh.z_blocks[k]['coord']
            u_exac = self.problem.u_exact(coord.detach().cpu().numpy(), case_index)
            u_exac = torch.from_numpy(u_exac).float().to(self.device)
            loss = self.criterion(u_pred, u_exac)
            total_loss = torch.sum((u_pred - u_exac)**2)/len(self.mesh.vertices)
            print(f'Finish block {k}, spent time: {time.time()-tt}, loss: {loss.item()}, total loss: {total_loss}')
            self.result['coord'] = coord.detach().cpu().numpy()
            self.result['u_pred'] = u_pred.detach().cpu().numpy()
            self.result['u_exac'] = u_exac.detach().cpu().numpy()
            self.result['error'] = torch.abs(torch.abs(u_pred)-torch.abs(u_exac)).detach().cpu().numpy()
            self.save_to_vtk(case_index)
            torch.cuda.empty_cache()
        return loss


    def save_to_vtk(self, case_index):
        sorted_data = []

        for node in self.mesh.vertices:
            if any(np.allclose(node, coord) for coord in self.result['coord']):
                index = np.where(np.all(np.isclose(self.result['coord'], node), axis=1))[0][0]
                sorted_data.append(
                    [node[0], node[1], node[2], self.result['u_pred'][index][0], self.result['u_exac'][index][0],
                     self.result['error'][index][0]])
            else:
                res = self.problem.u_exact(node[np.newaxis, ...], case_index)
                boundary_point_value = res
                sorted_data.append(
                    [node[0], node[1], node[2], boundary_point_value[0][0], boundary_point_value[0][0], 0])

        sorted_data = np.array(sorted_data)

        mesh = meshio.Mesh(
            points=self.mesh.vertices,
            cells= [("tetra", self.mesh.elements)],
            point_data={"u_pred": sorted_data[:, 3], "u_exac": sorted_data[:, 4], "u_error": sorted_data[:, 5]}
        )

        mesh.write( f'case{case_index+1}_Greenonet.vtu')

        np.savetxt(f'case{case_index+1}_Greenonet.txt', np.column_stack((self.mesh.vertices, sorted_data[:, 3])), fmt='%.6f')

        np.savetxt(f'case{case_index+1}.txt', np.column_stack((self.mesh.vertices, sorted_data[:, 4])), fmt='%.6f')


    def visual_solution(self, case_index):
        sol = self.problem.u_exact(self.mesh.vertices, case_index)
        mesh = meshio.Mesh(
            points=self.mesh.vertices,
            cells= [("tetra", self.mesh.elements)],
            point_data={"sol": sol}
        )
        mesh.write( f'results/case{self.args.solution_case}.vtk', binary=False)



if __name__ == '__main__':
    args = Options().parse()

    args.mesh_path = '../../../../mesh/poisson/heat_sink_v2_domain.mphtxt'
    args.boundary_mesh_path = '../../../../mesh/poisson/heat_sink_v2_boundary.mphtxt'
    args.domain = [0, 1, 0, 1, 0, 1]
    args.blocks_num = [1, 1, 1]
    args.shape = [1, 2]
    args.ngs_boundary = 1
    args.ngs_interior = 1
    args.test_samples = 1

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.pde_case = 'poisson'
    args.problem = Poisson('../../../../data/poisson/case1/test_data')

    # args.layers = [6, 9, 20, 20, 1]
    # args.layers = [6, 24, 24, 24, 24, 1]
    args.layers = [[[6, 12, 12, 12, 1], [6, 12, 12, 12, 1]]]

    args.model_path = f'checkpoints/block0.pkl'

    tester = Tester(args)
    # tester.visual_solution()
    tester.calculate()


