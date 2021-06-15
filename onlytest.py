from main import *

#--- MAIN ---#

time_ini = time.time()

use_model = "DeepSet"
#use_model = "GCN"
use_model = "EdgeNet"
#use_model = "PointNet"
#use_model = "MetaNet"

# Some of the following parameters are irrelevant for only testing, but needed for loading the proper model

# Learning rate
learning_rate = 0.002
# Weight decay
weight_decay = 6.e-6#1.e-7
# Number of layers of each graph layer
n_layers = 2
# Number of nearest neighbors in kNN / radius of NNs
k_nn = 7#3

# Number of epochs
n_epochs = 200
# If training, set to True, otherwise loads a pretrained model and tests it
training = False
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
simtype = "SIMBA"
# Simulation set, choose between "CV" and "LH"
simset = "CV"
# Number of simulations considered, maximum 27 for CV and 1000 for LH
n_sims = 27

params = [use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, simtype, simset]

main(params, training=training, n_sims=n_sims)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
