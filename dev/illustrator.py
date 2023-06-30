#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import argparse
from tqdm import tqdm
from math import pi

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

torch.set_default_dtype(torch.float64)

# %%
sys.path.append("..")
from model.deeponet import *
from utils import *
from src.collision_operator_1D import CollisionOperator1D

# %%
parser = argparse.ArgumentParser(description="DeepOnet")

parser.add_argument("--name", default="3_8_3_8", type=str)
parser.add_argument("--model", default="deeponet", type=str)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_squeue", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument(
    "--batch_size",
    default=0,
    type=int,
    help="batch size for train data (0 is full batch)",
)
parser.add_argument("--epochs", default=100000, type=int, help="Number of Epochs")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument(
    "--lambda", default=0, type=float, help="Loss weight for orthogonality"
)
parser.add_argument(
    "--d_t", default=2, type=int, help="depth of target network (except basis)"
)
parser.add_argument(
    "--w_t", default=20, type=int, help="width of target network (except basis)"
)
parser.add_argument(
    "--d_b", default=2, type=int, help="depth of branch network (except basis)"
)
parser.add_argument(
    "--w_b", default=20, type=int, help="width of branch network (except basis)"
)
parser.add_argument("--act", default="tanh", type=str, help="activation function")
parser.add_argument("--n_basis", default=20, type=int, help="number of basis")
parser.add_argument(
    "--d_in", default=1, type=int, help="dimension of input for target function"
)
parser.add_argument(
    "--d_out", default=1, type=int, help="dimension of output for target function"
)
parser.add_argument("--fix_bias", action="store_true")

# %%
# args = parser.parse_args() # from cmd
args = parser.parse_args([])
gparams = args.__dict__

# %%
random_seed = gparams["seed"]
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# %%
if not gparams["use_squeue"]:
    gpu_id = str(gparams["gpu"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    use_cuda = torch.cuda.is_available()
    print("Is available to use cuda? : ", use_cuda)
    if use_cuda:
        print("-> GPU number ", gpu_id)

# %%
NAME = gparams["name"]
PATH = "../results/{}/".format(gparams["model"])
if not os.path.exists(PATH):
    os.mkdir(PATH)
PATH = os.path.join(PATH, NAME)
os.mkdir(PATH)
# torch.save(args, os.path.join(PATH, "args.bin"))

# %%
train_data = np.load("../data/train_data.npz")
test_data = np.load("../data/test_data.npz")

train_data_f, train_data_Q = torch.DoubleTensor(
    train_data["data_f"]
), torch.DoubleTensor(train_data["data_Q"])
test_data_f, test_data_Q = torch.DoubleTensor(test_data["data_f"]), torch.DoubleTensor(
    test_data["data_Q"]
)

resol = train_data_f.shape[-1]
num_train = train_data_f.shape[0]
num_test = test_data_f.shape[0]

if gparams["batch_size"] == 0:
    batch_size_train = train_data_f.shape[0]
else:
    batch_size_train = gparams["batch_size"]

# %%
## Train data
dataset = TensorDataset(train_data_f.unsqueeze(1), train_data_Q)
train_dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

# Test data
batch_size_test = num_test
dataset = TensorDataset(test_data_f.unsqueeze(1), test_data_Q)
test_dataloader = DataLoader(dataset, batch_size=batch_size_test, shuffle=False)

# %%
d_t = gparams["d_t"]
w_t = gparams["w_t"]
d_b = gparams["d_b"]
w_b = gparams["w_b"]
act = gparams["act"]
n_basis = gparams["n_basis"]
n_sensor = resol
output_d_in = gparams["d_in"]
output_d_out = gparams["d_out"]

# %%
DeepONet = deeponet(
    d_t,
    w_t,
    d_b,
    w_b,
    act,
    n_basis,
    n_sensor,
    output_d_in,
    output_d_in,
    fix_bias=gparams["fix_bias"],
)  # .cuda()

# %%
integration_order = 100
Q = CollisionOperator1D(integration_order)
grid = torch.DoubleTensor(Q.get_quad_pts()).reshape(-1, 1)  # .cuda()
quad_w = torch.DoubleTensor(Q.get_quad_weights())  # .cuda()

# %%
resb = DeepONet.branch_list(train_data_f)
rest = DeepONet.trunk_list(grid)

# %%
B_sensor = train_data_f.shape[0]
B_grid = grid.shape[0]
coeffs = resb.reshape(B_sensor, output_d_in, 1, n_basis + 1).repeat(1, 1, B_grid, 1)
coeff = coeffs[..., :-1]
basis = rest.reshape(1, output_d_in, B_grid, n_basis).repeat(B_sensor, 1, 1, 1)

# plt.plot(tmp[2, :])

# %%
# res = DeepONet(grid, train_data_f)
res = torch.einsum("bijk,bijk->bij", coeff, basis)
