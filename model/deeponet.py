import torch
from torch import nn
import numpy as np


class deeponet(nn.Module):
    def __init__(self, branch_hidden, trunk_hidden, act, output_d_out, use_bias, use_gram, quad_w=None, size_domain=None):
        super(deeponet, self).__init__()
        self.branch_hidden=branch_hidden
        self.branch_num_layer=len(branch_hidden)
        self.trunk_hidden=trunk_hidden
        self.trunk_num_layer=len(trunk_hidden)
        self.output_d_out=output_d_out
        self.use_bias=use_bias
        # use_bias = False or 'vanila' or 'depend'
        if self.use_bias=='vanila':
            self.bias = nn.Parameter(torch.zeros(self.output_d_out))

        if trunk_hidden[-1]==branch_hidden[-1]:
            self.n_basis=branch_hidden[-1]
        self.use_gram=use_gram
        self.quad_w=quad_w
        self.size_domain=size_domain

        # Type of activation function
        if act=='tanh':
            self.act=nn.Tanh()
        elif act=='prelu':
            self.act=nn.PReLU()
        elif act=='relu':
            self.act=nn.ReLU()
        elif act=='gelu':
            self.act=nn.GELU()
        else:
            print('Error!!! No activation!')

        
        self.branch_list = []
        for i in range(self.branch_num_layer-1):
            # hidden layers
            if i!=self.branch_num_layer-2:
                self.branch_list.append(nn.Linear(branch_hidden[i], branch_hidden[i+1]))
                self.branch_list.append(self.act)
            # final layers without final activation
            else:
                self.branch_list.append(nn.Linear(branch_hidden[i], self.output_d_out*branch_hidden[i+1]))
        self.branch_list = nn.Sequential(*self.branch_list)
        
        self.trunk_list = []
        for i in range(self.trunk_num_layer-1):
            # hidden layers
            if i!=self.trunk_num_layer-2:
                self.trunk_list.append(nn.Linear(trunk_hidden[i], trunk_hidden[i+1])) 
                self.trunk_list.append(self.act)
            # final layers with final activation
            else:
                self.trunk_list.append(nn.Linear(trunk_hidden[i], self.output_d_out*trunk_hidden[i+1]))
                self.trunk_list.append(self.act)
                
        self.trunk_list = nn.Sequential(*self.trunk_list)
    
    def modified_gram_schmidt(self, matrix, quad_w):
        ## matrix  = (N, d)
        ## Output q1.T is also (N, d) shape (q1 is (d, N) shape, r1 is (N, N))
        ## 
        ## 
        matrix=torch.transpose(matrix,0,1)
        ## Now matrix  = (d, N)
        """
        For d x N matrix return Q1 and R1 components of QR decomposition using
        the modified Gram-Schmidt process, where R1 is N x N upper triangular
        and Q1 is d x N and have orthogonal columns.
        """
        def projection(u, v):
            return (v * u*quad_w).sum() / (u * u*quad_w).sum() * u
        N = matrix.shape[1]
        q1 = matrix.clone()
        r1 = torch.zeros(N, N)
        for k in range(N):
            #a_k = q1[..., k]
            r1[k,k] = ((q1[..., k].clone() *q1[..., k].clone()*quad_w).sum()**0.5)
            q1[..., k] = q1[..., k].clone() / r1[k, k].clone()
            for i in range(k+1, N):
                #a_i = q1[..., i]
                #r1[k,i] = np.transpose(a_k) @ a_i
                #a_i -= r1[k, i] * a_k
                q1[..., i]=q1[..., i].clone() - projection(q1[..., k].clone(),q1[..., i].clone())
        return torch.transpose(q1,0,1)

    # def gram_schmidt(self, vv, quad_w):
    #     ## vv = (N , d)
    #     ## N basis functions at d quad pts values
    #     def projection(u, v):
    #         #print(u.shape, v.shape)
    #         return (v * u*quad_w).sum() / (u * u*quad_w).sum() * u

    #     nk = vv.size(0)
    #     uu = torch.zeros_like(vv, device=vv.device)
    #     uu[0] = vv[0].clone()
    #     for k in range(1, nk):
    #         vk = vv[k].clone()
    #         uk = 0
    #         for j in range(0, k):
    #             uj = uu[j].clone()
    #             uk = uk + projection(uj, vk)
    #         uu[k] = vk - uk
    #     for k in range(nk):
    #         uk = uu[k].clone()
    #         uu[k] = uk / ((uk * uk*quad_w).sum()**0.5)
    #     return uu

    def get_basis(self, data_grid, add_one_basis=False):
        ## add_one_basis : To get the all basis instead of the output of trunk net.
        if self.use_gram:
            # Use gram
            add_basis=torch.cat((torch.ones(1,self.branch_hidden[0]).to(data_grid.device),self.trunk_list(data_grid).transpose(0,1)))
            basis_all=self.modified_gram_schmidt(add_basis, self.quad_w.to(data_grid.device))
            #basis_all=self.gram_schmidt(add_basis, self.quad_w.to(data_grid.device))
            if add_one_basis:
                basis=basis_all.transpose(0,1)
            else:
                basis=basis_all[1:].transpose(0,1) 
        else:
            # No use gram
            basis=self.trunk_list(data_grid)
            if add_one_basis:
                basis=torch.cat((torch.ones(self.branch_hidden[0],1).to(data_grid.device),basis), dim=1)
        return basis

    def forward(self, data_sensor, data_grid):
        # Input data_sensor shape : B_sensor x num_sensor
        # Input data_grid shape : B_sensor x d_in
        B_sensor=data_sensor.shape[0]
        B_grid=data_grid.shape[0]
        
        # Coefficient in branch net
        coeff=self.branch_list(data_sensor).reshape(B_sensor,self.output_d_out,1,self.n_basis).repeat(1,1,B_grid,1)
        
        # Basis in trunk net
        basis=self.get_basis(data_grid).reshape(1,self.output_d_out,B_grid,self.n_basis).repeat(B_sensor,1,1,1)

        # Set bias to enforce properties of collision operator (bias is a coefficient for 1-constant bases)
        if self.use_bias=='depend':
            self.bias = -(1/self.size_domain)*torch.sum(self.quad_w.to(data_grid.device).reshape(-1,1)*coeff*basis, dim=[-1,-2]).reshape(B_sensor,self.output_d_out,1).repeat(1,1,B_grid)

        # y shape : B_sensor x output_d_out x B_grid
        y=torch.einsum("bijk,bijk->bij", coeff, basis)
        if self.use_bias=='vanila' or self.use_bias=='depend':
            y+=self.bias
        return y