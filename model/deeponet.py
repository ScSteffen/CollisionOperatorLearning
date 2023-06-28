import torch.nn as nn
import torch
    
class deeponet(nn.Module):
    def __init__(self, depth_trunk, width_trunk, depth_branch, width_branch, act, num_basis, num_sensor, output_d_in, output_d_out, fix_bias=False, val_bias=0):
        super(deeponet, self).__init__()
        self.output_d_out=output_d_out
        self.n_basis=num_basis
        self.num_sensor=num_sensor
        self.fix_bias=fix_bias
        if self.fix_bias:
            self.val_bias = val_bias
        
        ##trunk net
        if act=='tanh':
            self.activation=nn.Tanh()
        elif act=='prelu':
            self.activation=nn.PReLU()
        elif act=='relu':
            self.activation=nn.ReLU()
        else:
            print('activation error!!')
            
        if width_trunk!=width_branch:
            print('width no same error!!')
            
        self.trunk_list = []
        self.trunk_list.append(nn.Linear(output_d_in,width_trunk))
        self.trunk_list.append(self.activation)
        for i in range(depth_trunk-1):
            self.trunk_list.append(nn.Linear(width_trunk, width_trunk))
            self.trunk_list.append(self.activation)
        self.trunk_list.append(nn.Linear(width_trunk, self.output_d_out*self.n_basis))
        self.trunk_list.append(self.activation)
        self.trunk_list = nn.Sequential(*self.trunk_list)
        
        ##branch net
        self.branch_list = []
        self.branch_list.append(nn.Linear(self.num_sensor,width_branch))
        self.branch_list.append(self.activation)
        for i in range(depth_branch-1):
            self.branch_list.append(nn.Linear(width_branch, width_branch))
            self.branch_list.append(self.activation)

        self.branch_list.append(nn.Linear(width_branch, self.output_d_out*(self.n_basis+1)))
        self.branch_list = nn.Sequential(*self.branch_list)
        

        
    def forward(self, data_grid, data_sensor):
        
        B_sensor=data_sensor.shape[0]
        B_grid=data_grid.shape[0]
        coeffs=self.branch_list(data_sensor).reshape(B_sensor,self.output_d_out,1,self.n_basis+1).repeat(1,1,B_grid,1)
        coeff=coeffs[...,:-1]
        bias=coeffs[...,-1]
        if self.fix_bias:
            bias=bias*self.val_bias
        basis=self.trunk_list(data_grid).reshape(1,self.output_d_out,B_grid,self.n_basis).repeat(B_sensor,1,1,1)
        
        y=torch.einsum("bijk,bijk->bij", coeff, basis)+bias

        return y