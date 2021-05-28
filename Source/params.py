import numpy as np
import torch
import os
import random

torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

#--- PARAMETERS ---#

# Root path for simulations
simpathroot = "/projects/QUIJOTE/CAMELS/Sims/"

# Box size in comoving kpc/h
boxsize = 25.e3

# Validation and test size
valid_size, test_size = 0.15, 0.15

# 1 for testing only
only_test = 0
if only_test:   valid_size, test_size = 0.005, 0.99

# Batch size
batch_size = 128

# Number of epochs
epochs = 150

# Learning rate
learning_rate = 0.0008

# Weight decay
weight_decay = 4.e-7

# Number of nearest neighbors in KNN
k_nn = 6

# Weight of the message L1 regularization in the total loss respect to the standard loss
l1_reg = 0.01

data_aug = 1

# 1 if train for performing symbolic regression later, 0 otherwise
sym_reg = 0

# 1 if use L1 regularization with messages. Needed for symbolic regression
use_l1 = 0

repo_path = os.getcwd()+"/"
