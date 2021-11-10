

# Choose the GNN architecture between "DeepSet", "GCN", "EdgeNet", "PointNet", "MetaNet"
#use_model = "DeepSet"
#use_model = "GCN"
use_model = "EdgeNet"
#use_model = "PointNet"
#use_model = "EdgePoint"
#use_model = "MetaNet"

# Learning rate
learning_rate = 0.0006303742576368124
# Weight decay
weight_decay = 1.991127058506617e-08
# Number of layers of each graph layer
n_layers = 1
# Number of nearest neighbors in kNN / radius of NNs
k_nn = 4.86294104668695

# Number of epochs
n_epochs = 150
# If training, set to True, otherwise loads a pretrained model and tests it
training = True
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
#simsuite = "SIMBA"
simsuite = "SIMBA"
# Simulation set, choose between "CV" and "LH"
simset = "CV"
# Number of simulations considered, maximum 27 for CV and 1000 for LH
n_sims = 27

params = [use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims]
