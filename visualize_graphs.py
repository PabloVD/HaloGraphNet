#----------------------------------------------------------------------
# Script to visualize halos as graphs
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import time, datetime
from Source.networks import *
from Source.plotting import *
from Source.load_data import *


# Visualization routine for plotting graphs
def visualize_graph(data, ind, projection="3d", edge_index=None):

    fig = plt.figure(figsize=(4, 4))

    if projection=="3d":
        ax = fig.add_subplot(projection ="3d")
        pos = data.x[:,:3]
    elif projection=="2d":
        ax = fig.add_subplot()
        pos = data.x[:,:2]

    # Draw lines for each edge
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():

            src = pos[src].tolist()
            dst = pos[dst].tolist()

            if projection=="3d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.1, color='black')
            elif projection=="2d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.1, color='black')

    # Plot nodes
    if projection=="3d":
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=50, zorder=1000)
    elif projection=="2d":
        ax.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)

    #plt.axis('off')
    fig.savefig("Plots/visualize_graph_"+str(ind), bbox_inches='tight', dpi=300)


# Main routine to display graphs from several simulations
def display_graphs(simsuite, simset, n_sims, k_nn):

    # Max index of graphs to be displayed
    nmax = 20

    # Load data and create dataset
    dataset, node_features = create_dataset(simsuite, simset, n_sims)

    for i, data in enumerate(dataset[:nmax]):
        if (i%2)==0:    # take half of them

            # Get edges from nearest neighbors within a radius k_nn
            edge_index = radius_graph(data.pos, r=k_nn)

            #visualize_graph(data, i, "2d", edge_index)
            visualize_graph(data, i, "3d", edge_index)


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
