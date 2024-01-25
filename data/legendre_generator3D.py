# python3 legendre_generator.py train_100_test_100 --seed 0 --order 8 --num_train 100 --num_test 100

import numpy as np
import os
import sys
import argparse
import random
from tqdm import tqdm
from scipy.special import eval_legendre

src_dir = "../"
sys.path.append(os.path.dirname(src_dir))
from src.collision_operator_3D import CollisionOperator3D
from src.entropy_utils import EntropyTools

# %% Args
parser = argparse.ArgumentParser(description="Put your hyperparameters")

#parser.add_argument("name", type=str, help="experiments name")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--integration_order", default=4, type=int, help="Quadrature order")
parser.add_argument("--polynomial_order", default=8, type=int, help="Polynomial order")
parser.add_argument(
    "--num_train",
    type=int,
    default=100,
    help="Number of train data for each type : Total number of data will be 3 times of it.",
)
parser.add_argument(
    "--num_test",
    type=int,
    default=100000,
    help="Number of test data for each type : Total number of data will be 3 times of it.",
)

args = parser.parse_args()
gparams = args.__dict__

# %% Random seed
random_seed = gparams["seed"]
np.random.seed(random_seed)
random.seed(random_seed)


def f_legendre(quad_p, quad_w, order=8, d=1):
    cs = np.zeros(order)
    for i in range(order):
        cs[i] = 1 / (i + 1) ** 2 * np.random.randn()

    x = np.zeros(quad_p.shape)
    for i in range(order):
        x += cs[i] * (eval_legendre(i, quad_p) + 1)

    volume = np.sum(x * quad_w)

    return x / volume


num_train = gparams["num_train"]
num_test = gparams["num_test"]
integration_order = gparams["integration_order"]
polynomial_order = gparams["polynomial_order"]

Q = CollisionOperator3D(integration_order)
quad_pts = Q.get_quad_pts()
quad_weights = Q.get_quad_weights()

# %%  File name and path
PATH = "."

# Entropy

moment_degree = 2  # higher moment degree means multimodal densities etc ==> 2 is close to maxwellian
regularization = 0.0
et = EntropyTools(quad_order=integration_order, polynomial_degree=moment_degree, spatial_dimension=3, gamma=regularization)

n_samples = 1
condition_treshold = 10  # higher means more anisotropic densities
max_alpha = 3.0  # higher value means more anisotropic densities
#f, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=max_alpha)
v_q = et.quad_pts


# %% Train data
data_f = []
data_Q = []
print("------Make train data!!------")
for i in tqdm(range(num_train)):
    # Create Entropy Tools with given quadrature order
    f, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=max_alpha)

    data_f.append(f[:,0])
    f_reshape = f.reshape(integration_order*2,integration_order)
    Q_f = Q.evaluate_Q(f_reshape)
    data_Q.append(Q_f)

for i in tqdm(range(num_train)):
    # Create Entropy Tools with given quadrature order
    f, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=max_alpha*2)

    data_f.append(f[:,0])
    f_reshape = f.reshape(integration_order*2,integration_order)
    Q_f = Q.evaluate_Q(f_reshape)
    data_Q.append(Q_f)

np.savez(
    os.path.join(PATH, "3D_entropy_train_data.npz"),
    data_f=np.array(data_f),
    data_Q=np.array(data_Q),
)

# %%  Test data
data_f = []
data_Q = []
print("------Make test data!!------")
for i in tqdm(range(num_train)):
    # Create Entropy Tools with given quadrature order
    f, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=max_alpha)

    data_f.append(f[:,0])
    f_reshape = f.reshape(integration_order*2,integration_order)
    Q_f = Q.evaluate_Q(f_reshape)
    data_Q.append(Q_f)

for i in tqdm(range(num_train)):
    # Create Entropy Tools with given quadrature order
    f, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=max_alpha*2)

    data_f.append(f[:,0])
    f_reshape = f.reshape(integration_order*2,integration_order)
    Q_f = Q.evaluate_Q(f_reshape)
    data_Q.append(Q_f)

np.savez(
    os.path.join(PATH, "3D_entropy_test_data.npz"),
    data_f=np.array(data_f),
    data_Q=np.array(data_Q),
)
