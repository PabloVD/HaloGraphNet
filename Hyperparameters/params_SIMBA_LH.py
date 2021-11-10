

# Choose the GNN architecture between "DeepSet", "GCN", "EdgeNet", "PointNet", "MetaNet"
#use_model = "DeepSet"
#use_model = "GCN"
use_model = "EdgeNet"
#use_model = "PointNet"
#use_model = "EdgePoint"
#use_model = "MetaNet"

# Learning rate
learning_rate = 1.5516493513056266e-05
# Weight decay
weight_decay = 2.115143263673728e-08
# Number of layers of each graph layer
n_layers = 1
# Number of nearest neighbors in kNN / radius of NNs
k_nn = 0.3660069801281247

# Number of epochs
n_epochs = 150
# If training, set to True, otherwise loads a pretrained model and tests it
training = True
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
simsuite = "SIMBA"
#simsuite = "IllustrisTNG"
# Simulation set, choose between "CV" and "LH"
simset = "LH"
# Number of simulations considered, maximum 27 for CV and 1000 for LH
n_sims = 1000

params = [use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims]
