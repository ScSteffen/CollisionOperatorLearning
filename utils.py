import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def rel_L2_error(pred, true):
    return (torch.sum((true-pred)**2, dim=-1)/torch.sum((true)**2, dim=-1))**0.5