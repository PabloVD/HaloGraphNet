import numpy as np
import torch
import os
import random

torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)

#--- PARAMETERS ---#

valid_size, test_size = 0.15, 0.15

# Batch size
batch_size = 128

#submin, submax = 5, 800 #5, 200

simpath = "/projects/QUIJOTE/CAMELS/Sims/IllustrisTNG/CV_0/fof_subhalo_tab_"
#path = "/projects/QUIJOTE/CAMELS/Sims/IllustrisTNG/0/fof_subhalo_tab_033.hdf5"
# To see ls of columns of file, type in shell: h5ls -r fof_subhalo_tab_033.hdf5

# Number of simulations
n_sims = 34

# Number of epochs
epochs = 100

# Learning rate
learning_rate = 0.001

# Weight decay
weight_decay = 1.e-8

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
