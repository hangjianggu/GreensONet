#!/usr/bin/env python
"""
model.py
--------
Physics Informed Neural Network for solving Poisson equation
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from options import Options
from torch.nn.parameter import Parameter


def tile(x, y):
    X = x.repeat(y.shape[0], 1)
    Y = torch.cat([y[i].repeat(x.shape[0], 1) for i in range(y.shape[0])], dim=0)
    return torch.cat((X, Y), dim=1)


class Net(nn.Module):
    """
    Basic Network for PINNs
    """
    def __init__(self, layers, scale=1.0):
        """
        Initialization for Net
        """
        super().__init__()
        self.scale   = scale
        self.layers  = layers
        self.fcs     = []
        self.params  = []

        self.fc0 = nn.Linear(self.layers[0], self.layers[1], bias=False).float()
        setattr(self, f'fc{0}', self.fc0)

        weight = torch.tensor([
            [1.0, 0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0, 1.0,  0.0,  0.0,  0.0,  0.0],
            [0.0, 0.0,  1.0,  0.0,  0.0,  0.0],
            [0.0, 0.0,  0.0,  1.0,  0.0,  0.0],
            [0.0, 0.0,  0.0,  0.0,  1.0,  0.0],
            [0.0, 0.0,  0.0,  0.0,  0.0,  1.0],
            [1.0, 0.0,  0.0, -1.0,  0.0,  0.0],
            [0.0, 1.0,  0.0,  0.0, -1.0,  0.0],
            [0.0, 0.0,  1.0,  0.0,  0.0, -1.0],
        ])
        
        self.fc0.weight = torch.nn.Parameter(weight)
        self.fc0.weight.requires_grad = False

        for i in range(1, len(layers) - 2):
            fc    = nn.Linear(self.layers[i], self.layers[i+1]).float()
            setattr(self, f'fc{i}', fc)
            self._init_weights(fc)
            self.fcs.append(fc)
            
            param = nn.Parameter(torch.randn(self.layers[i+1]))
            setattr(self, f'param{i}', param)
            self.params.append(param)

        fc = nn.Linear(self.layers[-2], self.layers[-1])
        setattr(self, f'fc{len(layers)-2}', fc)
        self._init_weights(fc)
        self.fcs.append(fc)

    def _init_weights(self, layer):
        init.xavier_normal_(layer.weight)
        init.constant_(layer.bias, 0.01)

    def forward(self, X):
        X = self.fc0(X)
        X = self.fcs[0](X)
        for k in range(2):
            for i in range(1, len(self.fcs)-1):
                X = torch.mul(self.params[i], X) * self.scale
                X = torch.tanh(X)
        return self.fcs[-1](X)


class BSNN(nn.Module):
    def __init__(self, Layers, device):
        super().__init__()
        # Initialize parameters
        self.Layers = Layers
        self.device = device
        self.num_Layers = len(self.Layers)
        self.params = []
        self.act = torch.sin
        # Binary neural network related
        self.width = [Layers[0]] + [int(pow(2, i - 1) * Layers[i]) for i in range(1, len(Layers) - 1)] + [Layers[-1]]
        self.masks = self.construct_mask()
        self.num_param = self.cal_param()
        self.weights, self.biases = self.initialize_NN(self.Layers)

    # Calculate the number of parameters
    def cal_param(self):
        ret = 0
        for i in range(self.num_Layers - 2):
            temp = pow(2, i) * (self.Layers[i] * self.Layers[i + 1] + self.Layers[i + 1])
            ret += temp
        ret += (self.width[-2] * self.Layers[-1] + self.Layers[-1])
        return ret

    # Initialize binary neural network parameters
    def initialize_NN(self, Layers):
        weights = []
        biases = []
        # First hidden layer
        tempw = torch.zeros(self.Layers[0], self.width[1]).to(self.device)
        w = Parameter(tempw, requires_grad=True)
        nn.init.xavier_uniform_(w, gain=1)
        tempb = torch.zeros(1, self.width[1]).to(self.device)
        b = Parameter(tempb, requires_grad=True)
        weights.append(w)
        biases.append(b)
        self.params.append(w)
        self.params.append(b)
        # Middle hidden layers
        for l in range(1, self.num_Layers - 2):
            # Weights w
            tempw = torch.zeros(self.width[l], self.width[l + 1]).to(self.device)
            # Block diagonal matrix initialization
            for i in range(int(pow(2, l))):  # Traverse each small matrix and initialize
                tempw2 = torch.zeros(Layers[l], Layers[l + 1])
                w2 = Parameter(tempw2, requires_grad=True)
                nn.init.xavier_uniform_(w2, gain=1)
                row_index = int(i / 2)
                tempw[row_index * Layers[l]: (row_index + 1) * Layers[l],
                i * Layers[l + 1]: (i + 1) * Layers[l + 1]] = w2.data
            w = Parameter(tempw, requires_grad=True)
            # Bias b
            tempb = torch.zeros(1, self.width[l + 1]).to(self.device)
            b = Parameter(tempb, requires_grad=True)
            weights.append(w)
            biases.append(b)
            self.params.append(w)
            self.params.append(b)
        # Last hidden layer
        tempw = torch.zeros(self.width[-2], self.Layers[-1]).to(self.device)
        w = Parameter(tempw, requires_grad=True)
        tempb = torch.zeros(1, self.Layers[-1]).to(self.device)
        b = Parameter(tempb, requires_grad=True)
        weights.append(w)
        biases.append(b)
        self.params.append(w)
        self.params.append(b)
        return weights, biases

    # Create mask
    def construct_mask(self):
        masks = []
        for l in range(2, self.num_Layers - 2):
            # Calculate block matrix dimensions
            num_blocks = int(pow(2, l - 1))
            blocksize1 = int(self.width[l] / num_blocks)
            blocksize2 = 2 * self.Layers[l + 1]
            blocks = [torch.ones((blocksize1, blocksize2)) for i in range(num_blocks)]
            mask = torch.block_diag(*blocks).to(self.device)
            masks.append(mask)
        return masks

    # Binary neural network part
    def forward(self, X):
        # Network part
        for l in range(0, self.num_Layers - 2):
            if l >= 2 and l <= self.num_Layers - 3:
                W = self.weights[l]
                W2 = W * self.masks[l - 2]
                b = self.biases[l]
                X = self.act(torch.add(torch.matmul(X, W2), b))
            else:
                W = self.weights[l]
                b = self.biases[l]
                X = self.act(torch.add(torch.matmul(X, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        X = torch.add(torch.matmul(X, W), b)
        return X


class Net_Integral(nn.Module):
    def __init__(self, layers, shape, problem, device, ngs_boundary, ngs_interior):
        super().__init__()
        # self.net = Net(layers)

        # Initialize the array
        self.G = []
        # Create a matrix
        # if len(shape) > 1:
        #     for i in range(shape[0]):
        #         Row = []
        #         for j in range(shape[1]):
        #             Row.append(Net(layers))
        #         self.G.append(Row)
        # # Create a vector
        # else:
        #     for i in range(shape[0]):
        #         self.G.append(Net(layers))

        if len(shape) > 1:
            for i in range(shape[0]):
                Row = []
                for j in range(shape[1]):
                    Row.append(BSNN(layers, device))
                    # Row.append(Net(layers))
                self.G.append(Row)
        # Create a vector
        else:
            for i in range(shape[0]):
                self.G.append(BSNN(layers, device))
                # self.G.append(Net(layers))

        self.problem = problem
        self.device = device
        self.ngs_interior = ngs_interior
        self.ngs_boundary = ngs_boundary

    def forward(self, X_interior, X_boundary, Z_block, case_index):
        N_interior = X_interior['coord'][0].shape[0]
        N_boundary = X_boundary['coord'][0].shape[0]

        n_input = len(self.G[0])
        n_output = len(self.G)

        fG_quad = torch.zeros_like(Z_block['coord'])[:, 0]
        gGn_quad = torch.zeros_like(fG_quad)

        for i in range(n_output):
            for j in range(n_input):

                # loop quadrature point in domain
                for k in range(self.ngs_interior):
                    INPUT_interior = tile(X_interior['coord'][k], Z_block['coord'])

                    # Evaluate Gij at all spatial points
                    G_interior = self.G[i][j](INPUT_interior)
                    f_interior = self.problem.f(X_interior['coord'][k], case_index, j, self.device)
                    f_interior = f_interior.repeat(Z_block['coord'].shape[0], 1)
                    G_interior = G_interior.view(-1, N_interior).transpose(0, 1)
                    f_interior = f_interior.view(-1, N_interior).transpose(0, 1)

                    ### *是逐元素乘法, np.multiply也是点乘, np.matmul是矩阵乘法, @是矩阵乘法
                    fG_interior = f_interior * G_interior
                    fG_quad += fG_interior.transpose(0, 1) @ X_interior['wts'][k]

                # loop quadrature point on boundary
                for k in range(self.ngs_boundary):
                    INPUT_boundary = tile(X_boundary['coord'][k], Z_block['coord'])
                    INPUT_boundary_type = tile(X_boundary['boundary_type'], Z_block['coord'])
                    G_boundary = self.G[i][j](INPUT_boundary)

                    Ggrad_boundary = torch.autograd.grad(G_boundary, INPUT_boundary, torch.ones_like(G_boundary))[0][:, :3]
                    Gx_boundary = Ggrad_boundary[:, [0]].view(-1, N_boundary).transpose(0, 1)
                    Gy_boundary = Ggrad_boundary[:, [1]].view(-1, N_boundary).transpose(0, 1)
                    Gz_boundary = Ggrad_boundary[:, [2]].view(-1, N_boundary).transpose(0, 1)
                    Gn_boundary = Gx_boundary * X_boundary['normal'][:, [0]] + Gy_boundary * X_boundary['normal'][:, [1]] + Gz_boundary * X_boundary['normal'][:, [2]]

                    g_boundary =  self.problem.g(INPUT_boundary[:, :3], INPUT_boundary_type[:, 0], case_index, j, self.device).view(-1, N_boundary).transpose(0, 1)
                    a_boundary = self.problem.a(INPUT_boundary[:, :3], self.device).view(-1, N_boundary).transpose(0, 1)
                    gGn_boundary = a_boundary * g_boundary * Gn_boundary
                    gGn_quad += gGn_boundary.transpose(0, 1) @ X_boundary['wts'][k]

        quad_res = (fG_quad - gGn_quad)[:, None]

        return quad_res


if __name__ == '__main__':
    layers = [4, 6, 5, 5, 1]
    args = Options().parse()
    net     = Net(layers)
    params  = list(net.parameters())
    for name, value in net.named_parameters():
        print(name)
    # print(net.fc1.weight.shape)
    # print(net.fc1.bias)
        
    
