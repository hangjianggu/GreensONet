#!/usr/bin/env python
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
from problem import DiffusionReaction
from model import Net_Integral
from options import Options
from utils import Mesh, save_checkpoints, LpLoss



class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.device       = args.device
        self.cuda_index   = args.cuda_index
        self.pde_case     = args.pde_case
        self.resume       = args.resume
        self.problem      = args.problem
        self.tol          = args.tol
        self.tol_change   = args.tol_change

        # Trainset
        self.domain       = args.domain
        self.blocks_num   = args.blocks_num
        self.ngs_boundary = args.ngs_boundary
        self.ngs_interior = args.ngs_interior
        self.train_samples = args.train_samples
        self.test_samples = args.test_samples

        meshfile, boundaryfile = self.get_mesh_path(args.mesh_path, args.boundary_mesh_path)
        self.mesh = Mesh(meshfile, boundaryfile, self.domain, self.blocks_num, ngs_boundary = self.ngs_boundary, ngs_interior = self.ngs_interior)

        # Criterion
        self.criterion    = nn.MSELoss()
        # self.criterion    = LpLoss()

        # HyperParameters setting
        self.epochs_Adam  = self.args.epochs_Adam
        self.lam          = self.args.lam
        # self.model_path   = self._model_path()
        
        # Learning rate
        self.lr           = self.args.lr
        self.net_pde      = Net_Integral(args.layers, args.shape, self.problem, self.device, self.ngs_boundary, self.ngs_interior)
        # params            = [param for param in self.net_pde.parameters() if param.requires_grad == True]
        # params = [param for i in range(args.shape[0]) for j in range(args.shape[1]) for param in
        #           self.net_pde.G[i][j].parameters() if param.requires_grad == True]
        # 定义余弦退火调度器
        # self.scheduler = CosineAnnealingLR(self.optimizer_Adam, T_max=100, eta_min=0)

    def get_mesh_path(self, mesh_path, boundary_mesh_path):
        return mesh_path, boundary_mesh_path


    def _model_path(self):
        """ Create directory of saved model"""
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')

        pde = os.path.join('checkpoints', f'{self.pde_case}')
        if not os.path.exists(pde):
            os.mkdir(pde)

        geo = os.path.join('checkpoints', f'{self.pde_case}', f'{self.geo_type}')
        if not os.path.exists(geo):
            os.mkdir(geo)

        model_path = os.path.join('checkpoints', f'{self.pde_case}', f'{self.geo_type}')
        return model_path


    def train(self):
        if not isinstance(self.mesh.X_boundary['normal'], torch.Tensor):
            self.mesh.X_boundary['normal'] = torch.from_numpy(self.mesh.X_boundary['normal']).float().to(self.device)
            self.mesh.X_boundary['boundary_type'] = torch.from_numpy(self.mesh.X_boundary['boundary_type']).float().to(self.device)
            for k in range(self.ngs_interior):
                self.mesh.X_interior['coord'][k] = torch.from_numpy(self.mesh.X_interior['coord'][k]).float().to(self.device)
                self.mesh.X_interior['wts'][k] = torch.from_numpy(self.mesh.X_interior['wts'][k]).float().to(self.device)
            for k in range(self.ngs_boundary):
                self.mesh.X_boundary['coord'][k] = torch.from_numpy(self.mesh.X_boundary['coord'][k]).float().to(self.device).requires_grad_(True)
                self.mesh.X_boundary['wts'][k] = torch.from_numpy(self.mesh.X_boundary['wts'][k]).float().to(self.device).requires_grad_(True)

        for K in range(len(self.mesh.blocks)):
            if self.mesh.z_blocks[K] is not None and len(self.mesh.z_blocks[K])>0:
                if not isinstance(self.mesh.z_blocks[K]['coord'], torch.Tensor):
                    self.mesh.z_blocks[K]['coord'] = torch.from_numpy(self.mesh.z_blocks[K]['coord']).float().to(self.device)
                self.train_block(K)
            else:
                print(f'Block #{K} is empty!!!')


    def train_block(self, k):
        # resume checkpoint if needed
        if self.resume:
            resume_path = os.path.join('checkpoints', f'block{k}.pkl')
            if os.path.isfile(resume_path):
                print(f'Resuming training, loading {resume_path} ...')
                # self.checkpoint = torch.load(resume_path)
                for i in range(args.shape[0]):
                    for j in range(args.shape[1]):
                        self.net_pde.G = torch.load(resume_path, map_location=self.device)

        # Optimizers
        for i in range(args.shape[0]):
            for j in range(args.shape[1]):
                self.net_pde.G[i][j].train()
                self.net_pde.G[i][j].to(self.device)
                self.net_pde.G[i][j].zero_grad()

        params = [param for i in range(args.shape[0]) for j in range(args.shape[1]) for param in
                  self.net_pde.G[i][j].params if param.requires_grad == True]
        self.optimizer_Adam  = optim.Adam(params, lr=self.lr)
        self.lr_scheduler = StepLR(self.optimizer_Adam, step_size=100, gamma=0.9)

        best_loss = 1.e10
        print(f"Start Trainning (blocks #{k})")
        
        tt = time.time()
        output_file = open(os.path.join(f'output_{k}.txt'), 'w+')
        # Training Process using ADAM Optimizer
        for epoch in range(self.epochs_Adam):
            train_loss = 0
            total_loss = 0
            for case_index in range(self.train_samples):
                self.optimizer_Adam.zero_grad()

                u_pred = self.net_pde(self.mesh.X_interior, self.mesh.X_boundary, self.mesh.z_blocks[k], case_index)
                coord = self.mesh.z_blocks[k]['coord']
                u_exac = self.problem.u_exact(coord.detach().cpu().numpy(), case_index)
                u_exac = torch.from_numpy(u_exac).float().to(self.device)

                loss = self.criterion(u_pred, u_exac)
                mean_loss = torch.sum((u_pred - u_exac)**2)/len(self.mesh.vertices)
                loss.backward()
                self.optimizer_Adam.step()

                train_loss += loss.item()
                total_loss += mean_loss

            self.lr_scheduler.step()
            train_loss = train_loss/self.train_samples
            total_loss = total_loss/self.train_samples
            t2 = time.time()
            infos = f"Epoch: {epoch:5d}/{self.epochs_Adam:5d} " + \
                    f"time: {t2 - tt:.2f} " +\
                    f"lr: {self.lr_scheduler.get_lr()[0]:.2e} " + \
                    f"loss: {train_loss}  " + \
                    f"total loss: {total_loss}  "
            print(infos)
            output_file.write(f"epoch:{epoch:5d}, time: {(t2 - tt):.2f}, loss: {train_loss}, total loss: {total_loss}\n")
            tt = time.time()

            if (epoch + 1) % 5 == 0:
                is_best = train_loss < best_loss
                if is_best:
                    best_loss = train_loss
                save_checkpoints(k, self.net_pde.G, is_best, save_dir='checkpoints')

            if train_loss < self.tol:
                print(f'train_loss after Adam is {train_loss:.4e} ')
                is_best = train_loss < best_loss
                save_checkpoints(k, self.net_pde.G, is_best, save_dir='checkpoints')
                break

        output_file.close()


if __name__ == '__main__':
    args = Options().parse()

    args.mesh_path = '../../../../mesh/diffusion/regular_domain.mphtxt'
    args.boundary_mesh_path = '../../../../mesh/diffusion/regular_boundary.mphtxt'
    # args.geo_type = 'Poisson_regular_solution'
    args.solution_case = 1
    args.domain = [-1, 1, -1, 1, -1, 1]
    args.blocks_num = [1, 1, 1]
    args.shape = [1, 2]
    args.epochs_Adam = 1500
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.train_samples = 15
    args.test_samples = 5
    args.lr = 1e-3

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.pde_case = 'Diffusion'
    args.problem = DiffusionReaction('../../../../data/diffusion/case1/train_data')

    args.resume = True

    args.layers = [6, 12, 12, 12, 1]

    trainer = Trainer(args)    
    trainer.train()


