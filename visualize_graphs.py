#----------------------------------------------------------------------
# Script to visualize halos as graphs
# Author: Pablo Villanueva Domingo
# Last update: 16/6/21
#----------------------------------------------------------------------

import time, datetime
from Source.networks import *
from Source.plotting import *
from Source.load_data import *

# Visualization routine
def visualize_points_3D(data, ind, edge_index=None):

    pos = data.x[:,:3]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection ="3d")

    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.1, color='black')

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=50, zorder=1000)

    #plt.axis('off')
    fig.savefig("Plots/visualize_graph_"+str(ind), bbox_inches='tight', dpi=300)

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
