#----------------------------------------------------------------------
# Script to visualize halos as graphs
# Author: Pablo Villanueva Domingo
# Last update: 16/6/21
#----------------------------------------------------------------------

import time, datetime
from Source.networks import *
from Source.plotting import *
from Source.load_data import *


# Main routine to train the neural net
def display_graphs(simsuite, simset, n_sims, k_nn):

    # Load data and create dataset
    dataset, node_features = create_dataset(simsuite, simset, n_sims)

    for i, data in enumerate(dataset[:20]):
        if (i%2)==0:
            edge_index = radius_graph(data.pos, r=k_nn)
            #visualize_points(data, i, edge_index)
            visualize_points_3D(data, i, edge_index)


#--- MAIN ---#

time_ini = time.time()

# Number of nearest neighbors in kNN / radius of NNs
k_nn = 0.07
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
simsuite = "IllustrisTNG"
# Simulation set, choose between "CV" and "LH"
simset = "CV"
# Number of simulations considered, maximum 27 for CV and 1000 for LH
n_sims = 1

display_graphs(simsuite, simset, n_sims, k_nn)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
