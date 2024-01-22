# python3 toy_data_generation_isotropic.py train_300_test_300 --seed 0 --dimension 1 --integration_order 100 --num_train 100 --num_test 100
# python3 toy_data_generation_isotropic.py train_300_test_300 --seed 0 --dimension 3 --integration_order 10 --num_train 100 --num_test 100

import numpy as np

import os
import sys
from tqdm import tqdm
from math import pi
import argparse
import random

src_dir = '../'
sys.path.append(os.path.dirname(src_dir))
from src.collision_operator_1D import CollisionOperator1D
from src.collision_operator_3D import CollisionOperator3D



## ARGS
parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

### Setting
parser.add_argument('name', type=str, help='experiments name')
parser.add_argument("--seed", type=int, default=0)


### Data
parser.add_argument("--dimension", type=int, default=3)
parser.add_argument('--integration_order', default=100, type=int, help = 'Quadratic integration order')
parser.add_argument("--num_train", type=int, default=100, help='Number of train data for each type : Total number of data will be 3 times of it.')
parser.add_argument("--num_test", type=int, default=100, help='Number of test data for each type : Total number of data will be 3 times of it.')



args = parser.parse_args()
gparams = args.__dict__



## Random seed
random_seed=gparams['seed']
np.random.seed(random_seed)
random.seed(random_seed)

    
## File name and path
dimension=gparams['dimension']
PATH = '../data'+f'/{dimension}D'

## Make data
def f_gauss(quad_p, quad_w, d=dimension):#d=1,2,3
    #We will choose sigma=[0.1,0.4], center=[-0.2,0.2]
    rand_sigma=0.3*np.random.rand(d)+0.1
    rand_center=0.4*np.random.rand(d)-0.2

    def func(tensor, scale=1):
        const=np.prod((1/(2*pi*rand_sigma**2))**0.5)
        dist=np.prod(np.exp(-(1/(2*rand_sigma**2))*(tensor-rand_center)**2),axis=-1)
        return const*dist/scale
    volume=np.sum(func(quad_p)*quad_w)
    return lambda x : func(x, scale=volume)

def f_two_gauss(quad_p, quad_w, d=dimension):#d=1,2,3
    #We will choose sigma=[0.1,0.4], center=[-0.2,0.2]
    rand_sigma1=0.3*np.random.rand(d)+0.1
    rand_center1=0.4*np.random.rand(d)-0.2
    #We will choose sigma=[0.1,0.4], center=[-0.2,0.2]
    rand_sigma2=0.3*np.random.rand(d)+0.1
    rand_center2=0.4*np.random.rand(d)-0.2

    def func(tensor, scale=1):
        const1=np.prod((1/(2*pi*rand_sigma1**2))**0.5)
        dist1=np.prod(np.exp(-(1/(2*rand_sigma1**2))*(tensor-rand_center1)**2),axis=-1)
        const2=np.prod((1/(2*pi*rand_sigma2**2))**0.5)
        dist2=np.prod(np.exp(-(1/(2*rand_sigma2**2))*(tensor-rand_center2)**2),axis=-1)
        return (const1*dist1+const2*dist2)/scale
    volume=np.sum(func(quad_p)*quad_w)
    return lambda x : func(x, scale=volume)

def f_perturb_gauss(quad_p, quad_w, d=dimension):#d=1,2,3
    #We will choose sigma=[0.1,0.4], center=[-0.2,0.2]
    rand_sigma=0.3*np.random.rand(d)+0.1
    rand_center=0.4*np.random.rand(d)-0.2
    #We will choose coefficients=[0,1]
    coeff=1*np.random.rand(3,d)
    
    def func(tensor, scale=1):
        const=np.prod((1/(2*pi*rand_sigma**2))**0.5)
        dist=np.prod(np.exp(-(1/(2*rand_sigma**2))*(tensor-rand_center)**2),axis=-1)
        poly=np.sum(coeff[[0]]+coeff[[1]]*tensor+coeff[[2]]*tensor**2,axis=-1)
        return const*(dist+dist*poly)/scale
    volume=np.sum(func(quad_p)*quad_w)
    return lambda x : func(x, scale=volume)


num_train=gparams['num_train']
num_test=gparams['num_test']
integration_order=gparams['integration_order']

if dimension==1:
    Q = CollisionOperator1D(integration_order)
    quad_pts = Q.get_quad_pts().reshape(-1,dimension)
    quad_weights = Q.get_quad_weights()
elif dimension==3:
    Q = CollisionOperator3D(integration_order)
    quad_pts = Q.get_quad_pts().transpose(1,2,0).reshape(-1,dimension)
    quad_weights = Q.get_quad_weights().reshape(-1)

## Train data
data_f=[]
data_Q=[]
print("------Make train data!!------")
for i in tqdm(range(num_train)):
    if dimension==1:
        data_f.append(f_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1]))
        data_f.append(f_two_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1]))
        data_f.append(f_perturb_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1]))
    elif dimension==3:
        data_f.append(f_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1].reshape(2*integration_order,integration_order)).reshape(-1))
        data_f.append(f_two_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1].reshape(2*integration_order,integration_order)).reshape(-1))
        data_f.append(f_perturb_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1].reshape(2*integration_order,integration_order)).reshape(-1))

np.savez(os.path.join(PATH, 'toy_train_data.npz'),data_f=np.array(data_f), data_Q=np.array(data_Q))

## Test data
data_f=[]
data_Q=[]
print("------Make test data!!------")
for i in tqdm(range(num_test)):
    if dimension==1:
        data_f.append(f_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1]))
        data_f.append(f_two_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1]))
        data_f.append(f_perturb_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1]))
    elif dimension==3:
        data_f.append(f_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1].reshape(2*integration_order,integration_order)).reshape(-1))
        data_f.append(f_two_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1].reshape(2*integration_order,integration_order)).reshape(-1))
        data_f.append(f_perturb_gauss(quad_pts,quad_weights)(quad_pts))
        data_Q.append(Q.evaluate_Q(data_f[-1].reshape(2*integration_order,integration_order)).reshape(-1))

np.savez(os.path.join(PATH, 'toy_test_data.npz'),data_f=np.array(data_f), data_Q=np.array(data_Q))