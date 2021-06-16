

# Choose the GNN architecture between "DeepSet", "GCN", "EdgeNet", "PointNet", "MetaNet"
use_model = "DeepSet"
#use_model = "GCN"
use_model = "EdgeNet"
use_model = "PointNet"
#use_model = "EdgePoint"
#use_model = "MetaNet"

# Learning rate
learning_rate = 0.001
# Weight decay
weight_decay = 1.e-8#1.e-7
# Number of layers of each graph layer
n_layers = 3
# Number of nearest neighbors in kNN / radius of NNs
k_nn = 8#3

# Number of epochs
n_epochs = 200
# If training, set to True, otherwise loads a pretrained model and tests it
training = True
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
simtype = "IllustrisTNG"
# Simulation set, choose between "CV" and "LH"
simset = "CV"
# Number of simulations considered, maximum 27 for CV and 1000 for LH
n_sims = 27

params = [use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simtype, simset, n_sims]
