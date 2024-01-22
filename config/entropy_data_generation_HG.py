# python3 entropy_data_generation_HG.py train_300_test_300 --seed 0 --dimension 1 --integration_order 100 --aniso_param 0.9 --num_train 3000 --num_test 3000
# python3 entropy_data_generation_HG.py train_300_test_300 --seed 0 --dimension 3 --integration_order 10 --aniso_param 0.9 --num_train 3000 --num_test 3000

import numpy as np

import os
import sys

sys.path.append(os.path.abspath('..'))
from src.entropy_utils import EntropyTools
from tqdm import tqdm
from math import pi
import argparse
import random

src_dir = '../'
sys.path.append(os.path.dirname(src_dir))
from src.collision_operator_HG_1D import CollisionOperatorHG1D
from src.collision_operator_HG_3D import CollisionOperatorHG3D

## ARGS
parser = argparse.ArgumentParser(description='Put your hyperparameters')

### Setting
parser.add_argument('name', type=str, default='1D_hg_05', help='experiments name')
parser.add_argument("--seed", type=int, default=0)

### Data
parser.add_argument("--dimension", type=int, default=1)
parser.add_argument('--integration_order', default=20, type=int, help='Quadratic integration order')
parser.add_argument('--aniso_param', default=0.5, type=float, help='Anisotropy parameter (between -1 and 1)')
parser.add_argument("--num_train", type=int, default=10000, help='Number of train data for each type.')
parser.add_argument("--num_test", type=int, default=100,
                    help='Number of test data for each type : Total number of data will be 3 times of it.')

args = parser.parse_args()
gparams = args.__dict__

## Random seed
random_seed = gparams['seed']
np.random.seed(random_seed)
random.seed(random_seed)

## File name and path
dimension = gparams['dimension']
PATH = '../data' + f'/{dimension}D'

num_train = gparams['num_train']
num_test = gparams['num_test']
integration_order = gparams['integration_order']
aniso_param = gparams['aniso_param']

## Make data
data_type = ['train', 'test'][1]
moment_degree = 4  # higher moment degree means multimodal densities etc ==> 2 is close to maxwellian
regularization = 0.0
et = EntropyTools(quad_order=integration_order, polynomial_degree=moment_degree, spatial_dimension=dimension,
                  gamma=regularization)

# Create Entropy Tools with given quadrature order
n_samples = num_train
condition_treshold = 6  # higher means more anisotropic densities
max_alpha = 8.0  # higher value means more anisotropic densities
f_train, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=max_alpha)
f_test, _, _, _ = et.rejection_sampling(n=num_test, sigma=condition_treshold, max_alpha=max_alpha)

if dimension == 1:
    Q = CollisionOperatorHG1D(integration_order, aniso_param)
    quad_pts = Q.get_quad_pts().reshape(-1, dimension)
    quad_weights = Q.get_quad_weights()
elif dimension == 3:
    Q = CollisionOperatorHG3D(integration_order, aniso_param)
    quad_pts = Q.get_quad_pts().transpose(1, 2, 0).reshape(-1, dimension)
    quad_weights = Q.get_quad_weights().reshape(-1)

## Train data
data_Q = []
print("------Make train data!!------")
for i in tqdm(range(num_train)):
    if dimension == 1:
        data_Q.append(Q.evaluate_Q(f_train.T[i]))
    elif dimension == 3:
        data_Q.append(Q.evaluate_Q(f_train.T[i]))
np.savez(os.path.join(PATH, 'entropy_HG_' + str(aniso_param) + '_train_data.npz'), data_f=np.array(f_train.T),
         data_Q=np.array(data_Q))

## Test data
data_Q = []
print("------Make test data!!------")
for i in tqdm(range(num_test)):
    if dimension == 1:
        data_Q.append(Q.evaluate_Q(f_test.T[i]))
    elif dimension == 3:
        data_Q.append(Q.evaluate_Q(f_test.T[i]))

np.savez(os.path.join(PATH, 'entropy_HG_' + str(aniso_param) + '_test_data.npz'), data_f=np.array(f_test.T),
         data_Q=np.array(data_Q))
