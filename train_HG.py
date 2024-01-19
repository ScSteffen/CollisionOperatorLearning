from model.deeponet import *
from utils import *
import numpy as np
from src.collision_operator_HG_1D import CollisionOperatorHG1D
from src.collision_operator_HG_3D import CollisionOperatorHG3D
import os
import sys
from tqdm import tqdm
import argparse
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


#torch.set_default_dtype(torch.float64)


## ARGS
parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

### Setting
parser.add_argument('name', type=str, help='experiments name')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_squeue", action='store_true')
parser.add_argument("--gpu", type=int, default=0)


### Data
parser.add_argument("--data_file", default='toy', type=str)
parser.add_argument("--dimension", type=int, default=1)
parser.add_argument("--integration_order", type=int)
parser.add_argument('--aniso_param', default=0.0, type=float, help = 'Anisotropy parameter (between -1 and 1)')

### Model
parser.add_argument('--model', default='deeponet', type=str)
parser.add_argument('--branch_hidden', default=[100,8,8,8], nargs='+', type=int, help='branch network')
parser.add_argument('--trunk_hidden', default=[1,8,8,8], nargs='+', type=int, help='trunk network')
parser.add_argument('--act', default='tanh', type=str, help='activation function')
parser.add_argument('--d_out', default=1, type=int, help='dimension of output for target function')
parser.add_argument("--use_bias", type=str, choices=['no','vanila', 'vanila_ortho', 'gram', 'depend'], required=True)
parser.add_argument("--use_gram", action='store_true')

### Train parameters
parser.add_argument('--batch_size', default=0, type=int, help = 'batch size for train data (0 is full batch)')
parser.add_argument('--epochs', default=100000, type=int, help = 'Number of Epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument("--use_sche", action='store_true')
parser.add_argument('--step_size_sche', default=100, type=float, help='learning rate')
parser.add_argument('--lr_sche', default=0.9, type=float, help = 'Number of Epochs')
parser.add_argument('--lambda', default=0, type=float, help='Loss weight for orthogonality')



args = parser.parse_args()
gparams = args.__dict__



### Setting
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
data_file=gparams['data_file']
dimension=gparams['dimension']
NAME=gparams['name']
if not os.path.exists('../results'):    
    os.mkdir('../results')

PATH = os.path.join('../results', f'{dimension}D')
if not os.path.exists(PATH):    
    os.mkdir(PATH)

PATH = os.path.join(PATH, f'{data_file}')
if not os.path.exists(PATH):    
    os.mkdir(PATH)

PATH = os.path.join(PATH, NAME)
os.mkdir(PATH)
torch.save(args, os.path.join(PATH, 'args.bin'))



### Data
## Load data
train_data=np.load('../data/'+f'{dimension}D/'+data_file+'_train_data.npz')
test_data=np.load('../data/'+f'{dimension}D/'+data_file+'_test_data.npz')

train_data_f, train_data_Q = torch.FloatTensor(train_data['data_f']), torch.FloatTensor(train_data['data_Q'])
test_data_f, test_data_Q = torch.FloatTensor(test_data['data_f']), torch.FloatTensor(test_data['data_Q'])

resol=train_data_f.shape[-1]
num_train=train_data_f.shape[0]
num_test=test_data_f.shape[0]

if gparams['batch_size']==0:
    batch_size_train=num_train
else:
    batch_size_train = gparams['batch_size']

## Train data
dataset=TensorDataset(train_data_f.unsqueeze(1),train_data_Q)
train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

## Test data
batch_size_test = num_test
dataset=TensorDataset(test_data_f.unsqueeze(1),test_data_Q)
test_dataloader = DataLoader(dataset, batch_size=batch_size_test, shuffle=False)

## guad pts and weights
integration_order=gparams['integration_order']
aniso_param=gparams['aniso_param']
if dimension==1:
    Q = CollisionOperatorHG1D(integration_order, aniso_param)
    grid = torch.FloatTensor(Q.get_quad_pts()).reshape(-1,dimension).cuda()
    quad_w = torch.FloatTensor(Q.get_quad_weights()).cuda()
    size_domain=torch.sum(quad_w.cpu().detach())
elif dimension==3:
    Q = CollisionOperatorHG3D(integration_order, aniso_param)
    grid = torch.FloatTensor(Q.get_quad_pts().transpose(1,2,0)).reshape(-1,dimension).cuda()
    quad_w = torch.FloatTensor(Q.get_quad_weights()).reshape(-1).cuda()
    size_domain=torch.sum(quad_w.cpu().detach())



### Model
branch_hidden=gparams['branch_hidden']
trunk_hidden=gparams['trunk_hidden']
act=gparams['act']
output_d_out=gparams['d_out']
if gparams['use_bias']=='no':
    use_bias=False
else:
    use_bias=gparams['use_bias']
if gparams['use_bias']=='no':
    n_basis=trunk_hidden[-1]
else:
    n_basis=trunk_hidden[-1]+1
use_gram=gparams['use_gram']
## Set model
if gparams['model']=='deeponet':
    DeepONet=deeponet(branch_hidden, trunk_hidden, act, output_d_out, use_bias, use_gram, quad_w, size_domain).cuda()
print(DeepONet)


### Train parameters
num_epochs=gparams['epochs']
lr=gparams['lr']
use_sche=gparams['use_sche']
step_size_sche=gparams['step_size_sche']
lr_sche=gparams['lr_sche']
lambd=gparams['lambda']
optimizer = torch.optim.Adam(params=DeepONet.parameters(), lr=lr)
loss_func=nn.MSELoss()

if use_sche:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size_sche, gamma=lr_sche)

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
        output=DeepONet(x,grid)
        loss_deeponet=loss_func(output.squeeze(), y)
        
        loss_ortho=0.0

        # basis=DeepONet.trunk_list(grid).transpose(0,1)
        # if not (gparams['use_bias']=='no' and gparams['use_gram']==False):
        #     basis=torch.cat((basis,torch.ones(1,basis.shape[-1]).cuda()))
        basis=DeepONet.get_basis(grid,add_one_basis=use_bias).transpose(0,1)
        
        for i in range(n_basis):
            for j in range(n_basis):
                if j!=i:
                    loss_ortho+=torch.abs(torch.sum(basis[i]*basis[j]*quad_w))
        loss_ortho/=(n_basis*(n_basis-1))
        
        loss=loss_deeponet+lambd*loss_ortho

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            DeepONet.eval()
            output=DeepONet(x,grid)
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
    
    if epoch%100==0:
        # Test
        with torch.no_grad():
            DeepONet.eval()
            for x, y in test_dataloader:
                x, y = x.cuda(), y.cuda()
                with torch.no_grad():
                    DeepONet.eval()
                    output=DeepONet(x,grid)
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
    if use_sche:    
        scheduler.step()
torch.save({
    'state_dict': DeepONet.state_dict(),
    'loss' : list_train_loss_deeponet,
    'rel_train' : list_train_Q_rel_error,
    'rel_test' : list_test_Q_rel_error,
    'epoch': epoch,
    'optimizer':optimizer.state_dict()
    }, os.path.join(PATH, 'final.bin'))