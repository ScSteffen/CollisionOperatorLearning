#python3 train.py 3_8_3_8 --model deeponet --seed 0 --gpu 1 --epochs 100000 --lambda 0 --d_t 3 --w_t 8 --d_b 3 --w_b 8 --act tanh --n_basis 8

#python3 train.py 3_8_3_8_enforce --model deeponet --seed 0 --gpu 2 --epochs 100000 --lambda 0.1 --d_t 3 --w_t 8 --d_b 3 --w_b 8 --act tanh --n_basis 8 --fix_bias


from model.deeponet import *
from utils import *
import numpy as np
from src.collision_operator_1D import CollisionOperator1D
import matplotlib.pyplot as plt
import os
from os import path
import sys
from tqdm import tqdm
from math import pi
import argparse
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn


torch.set_default_dtype(torch.float64)


## ARGS
parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

### Setting
parser.add_argument('name', type=str, help='experiments name')
parser.add_argument('--model', default='deeponet', type=str)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_squeue", action='store_true')
parser.add_argument("--gpu", type=int, default=0)


### Train parameters
parser.add_argument('--batch_size', default=0, type=int, help = 'batch size for train data (0 is full batch)')
parser.add_argument('--epochs', default=100000, type=int, help = 'Number of Epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lambda', default=0, type=float, help='Loss weight for orthogonality')
parser.add_argument('--d_t', default=2, type=int, help='depth of target network (except basis)')
parser.add_argument('--w_t', default=20, type=int, help='width of target network (except basis)')
parser.add_argument('--d_b', default=2, type=int, help='depth of branch network (except basis)')
parser.add_argument('--w_b', default=20, type=int, help='width of branch network (except basis)')
parser.add_argument('--act', default='tanh', type=str, help='activation function')
parser.add_argument('--n_basis', default=20, type=int, help='number of basis') 
parser.add_argument('--d_in', default=1, type=int, help='dimension of input for target function')
parser.add_argument('--d_out', default=1, type=int, help='dimension of output for target function')
parser.add_argument("--fix_bias", action='store_true')



args = parser.parse_args()
gparams = args.__dict__



## Random seed
random_seed=gparams['seed']
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

## Choose gpu
if not gparams['use_squeue']:
    gpu_id=str(gparams['gpu'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    use_cuda = torch.cuda.is_available()
    print("Is available to use cuda? : ",use_cuda)
    if use_cuda:
        print("-> GPU number ",gpu_id)
    
## File name and path
NAME=gparams['name']
PATH = 'results/{}/'.format(gparams['model'])
if not os.path.exists(PATH):    
    os.mkdir(PATH)
PATH = os.path.join(PATH, NAME)
os.mkdir(PATH)
torch.save(args, os.path.join(PATH, 'args.bin'))

## Load data
train_data=np.load('data/train_data.npz')
test_data=np.load('data/test_data.npz')

train_data_f, train_data_Q = torch.DoubleTensor(train_data['data_f']), torch.DoubleTensor(train_data['data_Q'])
test_data_f, test_data_Q = torch.DoubleTensor(test_data['data_f']), torch.DoubleTensor(test_data['data_Q'])

resol=train_data_f.shape[-1]
num_train=train_data_f.shape[0]
num_test=test_data_f.shape[0]

if gparams['batch_size']==0:
    batch_size_train=train_data_f.shape[0]
else:
    batch_size_train = gparams['batch_size']

## Train data
dataset=TensorDataset(train_data_f.unsqueeze(1),train_data_Q)
train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

# Test data
batch_size_test = num_test
dataset=TensorDataset(test_data_f.unsqueeze(1),test_data_Q)
test_dataloader = DataLoader(dataset, batch_size=batch_size_test, shuffle=False)




d_t=gparams['d_t']
w_t=gparams['w_t']
d_b=gparams['d_b']
w_b=gparams['w_b']
act=gparams['act']
n_basis=gparams['n_basis']
n_sensor=resol
output_d_in=gparams['d_in']
output_d_out=gparams['d_out']

## Set model
if gparams['model']=='deeponet':
    DeepONet=deeponet(d_t, w_t, d_b, w_b, act, n_basis, n_sensor, output_d_in, output_d_in, fix_bias=gparams['fix_bias']).cuda()

## Train model
num_epochs=gparams['epochs']
lr=gparams['lr']
lambd=gparams['lambda']
optimizer = torch.optim.Adam(params=DeepONet.parameters(), lr=lr)
loss_func=nn.MSELoss()


## guad pts and weights
integration_order = 100
Q = CollisionOperator1D(integration_order)
grid = torch.DoubleTensor(Q.get_quad_pts()).reshape(-1,1).cuda()
quad_w = torch.DoubleTensor(Q.get_quad_weights()).cuda()

list_train_loss_deeponet=[]
list_train_loss_ortho=[]
list_train_Q_rel_error=[]
list_test_Q_rel_error=[]
pbar = tqdm(total=num_epochs, file=sys.stdout)
best_loss=1
for epoch in tqdm(range(1,num_epochs+1)):
    DeepONet.train()
    train_loss_deeponet = 0.0   
    train_loss_ortho = 0.0   
    train_Q_rel_error = 0.0
    test_Q_rel_error = 0.0
    num_ex_train = 0
    num_ex_test = 0        
    # Train
    for x, y in train_dataloader:
        x, y = x.cuda(), y.cuda()

        # take a step
        output=DeepONet(grid,x)
        loss_deeponet=loss_func(output.squeeze(), y)
        
        loss_ortho=torch.zeros(1).cuda()
        basis=DeepONet.trunk_list(grid).transpose(0,1)
        basis=torch.cat((basis,torch.ones(1,basis.shape[-1]).cuda()))
        for i in range(w_b+1):
            for j in range(w_b+1):
                if j!=i:
                    loss_ortho+=torch.abs(torch.sum(basis[i]*basis[j]*quad_w))
        loss_ortho/=(w_b*(w_b+1))
        
        loss=loss_deeponet+lambd*loss_ortho

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            DeepONet.eval()
            output=DeepONet(grid,x)
            Q_rel_error=rel_L2_error(output.squeeze(),y).detach().cpu()
        # update history
        num_batch=x.shape[0]
        train_loss_deeponet += loss_deeponet.item()*num_batch
        train_loss_ortho += loss_ortho.item()*num_batch
        train_Q_rel_error += torch.sum(Q_rel_error)
        num_ex_train += num_batch

    train_loss_deeponet /= num_ex_train
    train_loss_ortho /= num_ex_train
    train_Q_rel_error /= num_ex_train
        
    
    list_train_loss_deeponet.append(round(train_loss_deeponet,8))
    list_train_loss_ortho.append(round(train_loss_ortho,8))
    list_train_Q_rel_error.append(train_Q_rel_error)
    
    if epoch%10000==0:
        # Test
        with torch.no_grad():
            DeepONet.eval()
            for x, y in test_dataloader:
                x, y = x.cuda(), y.cuda()
                with torch.no_grad():
                    DeepONet.eval()
                    output=DeepONet(grid,x)
                    Q_rel_error=rel_L2_error(output.squeeze(),y).detach().cpu()
                num_batch=x.shape[0]
                test_Q_rel_error += torch.sum(Q_rel_error)
                num_ex_test += num_batch

            test_Q_rel_error /= num_ex_test
            list_test_Q_rel_error.append(test_Q_rel_error)

        pbar.set_description("###### Epoch : %d, Loss_train : %.8f, Loss_ortho : %.8f, rel_train : %.5f, rel_test : %.5f ######"%(epoch, list_train_loss_deeponet[-1], list_train_loss_ortho[-1], list_train_Q_rel_error[-1], list_test_Q_rel_error[-1]))
        if best_loss>list_train_loss_deeponet[-1]:
            torch.save({
                'state_dict': DeepONet.state_dict(),
                'best_loss' : list_train_loss_deeponet[-1],
                'best_rel_train' : list_train_Q_rel_error[-1],
                'best_rel_test' : list_test_Q_rel_error[-1],
                'epoch': epoch,
                }, os.path.join(PATH, 'best.bin'))
            best_loss=list_train_loss_deeponet[-1]

torch.save({
    'state_dict': DeepONet.state_dict(),
    'loss' : list_train_loss_deeponet,
    'rel_train' : list_train_Q_rel_error,
    'rel_test' : list_test_Q_rel_error,
    'epoch': epoch,
    'optimizer':optimizer.state_dict()
    }, os.path.join(PATH, 'final.bin'))