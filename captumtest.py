#--------------------------------------------
# Script for model interpretability
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#--------------------------------------------

import time, datetime, psutil
from Source.networks import *
from Source.training import *
from Source.load_data import *
from captum.attr import IntegratedGradients, Saliency
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12})

# Captum method for interpretability (see captum documentation)
method = "saliency"
#method = "intgradients"

#--- FUNCTIONS ---#

# Write model function in the appropriate format for captum
def model_forward(datax, model, data):
    u=torch.tensor([[datax.shape[0], torch.log10(torch.sum(10.**datax[:,3]))]], dtype=torch.float)
    graph = Data(x=datax, pos=data.pos, u=data.u, batch=data.batch)
    return model(graph)

# Visualization routine
def visualize_points_3D(datax, ind, colors="blue", edge_index=None):

    datax.detach().cpu().numpy()
    pos = datax[:,:3]*boxsize
    massstar = datax[:,3]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection ="3d")

    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.05, color='grey', alpha=0.7)

    sizes = 10.**(massstar+2.)
    scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, c=colors, zorder=1000, vmin=0., vmax=0.16)
    colbar = plt.colorbar(scat, ax=ax, fraction=0.04, pad=0.1)
    colbar.ax.set_ylabel("Saliency")#, loc="top")# rotation=270)
    ax.set_xlabel(r"$x$ [kpc/h]")
    ax.set_ylabel(r"$y$ [kpc/h]")
    ax.set_zlabel(r"$z$ [kpc/h]")
    #ax.xaxis.set_major_locator(MultipleLocator(100))
    #ax.yaxis.set_major_locator(MultipleLocator(100))
    #ax.zaxis.set_major_locator(MultipleLocator(100))

    fig.savefig("Plots/visualize_graph_"+str(ind)+".pdf", bbox_inches='tight', dpi=300)
    plt.close(fig)

# Plot average feature importances
def feature_importance_plot(importances, feature_names, method):

    x_pos = (np.arange(len(feature_names)))
    plt.figure()
    plt.bar(x_pos, importances, align='center')
    plt.xticks(x_pos, feature_names, wrap=True)
    plt.xlabel("Node features")
    #plt.yscale("log")
    plt.title("Average Feature Importances")
    plt.savefig("Plots/captum_importances_"+method+".pdf")
    plt.close()

# Routine for model interpretability making use of captum
# Provides saliency graphs, feature importance plot and saliency value with respect to the distance
def captum_routine(model, dataloader, radiusneigh):

    figdist, axdist = plt.subplots()

    model.eval()
    atrs_feat = []

    nhalos, nhalos_compl = 0, 0

    for i, data in enumerate(dataloader):

        data.to(device)
        data.x.requires_grad_(True)

        # Apply the method and get attributions
        if method=="saliency":
            atrmethod = Saliency(lambda datax: model_forward(datax, model, data))
        elif method=="intgradients":
            atrmethod = IntegratedGradients(lambda datax: model_forward(datax, model, data))    # not working
        attributions = atrmethod.attribute(data.x, target=0)

        atrs_feat.append(attributions.detach().cpu().numpy().mean(axis=0))
        atr_col = np.abs(attributions.detach().cpu().numpy()).mean(axis=1)
        dists = np.sqrt(np.sum(data.pos.detach().cpu().numpy()**2.,axis=1))*boxsize

        # Scatter plot of saliency vs distance to the center
        sizes = 10.**(data.x[:,3].detach().cpu().numpy()+2.)
        indxs = np.argwhere(dists>0.).reshape(-1)
        axdist.scatter(dists[indxs], atr_col[indxs], s=sizes[indxs]*0.2, c="blue", alpha=0.3)
        #axdist.scatter(dists[indxs], sizes[indxs], c=atr_col[indxs], alpha=0.5, vmin=0., vmax=0.3)

        out = model(data)
        y_out = out[:,0]
        err = (y_out.reshape(-1) - data.y)/data.y
        err = np.abs(err.detach().cpu().numpy())

        # Plot saliency graph
        for ibatch in range(batch_size):
            data.x.requires_grad_(False)

            # Choose only the subhalos of the graph within the batch
            indexes = np.argwhere(ibatch==data.batch).reshape(-1)
            datagraph = data.x[indexes]
            if indexes.shape[0]>0:  # Avoid possible segmentation fault

                edge_index = radius_graph(datagraph[:,:3], r=radiusneigh)

                num_nodes, num_edges = datagraph.shape[0], edge_index.shape[1]//2    # Divide by 2 since edges are counted doubled if not directed

                if num_edges==num_nodes*(num_nodes-1)//2:
                    nhalos_compl+=1
                nhalos+=1

                if err[ibatch]<0.015 and datagraph.shape[0]>=10:
                #if datagraph.shape[0]>=10 and datagraph.shape[0]<15:
                    #print("Ind", ibatch, "Error",err[ibatch])

                    visualize_points_3D(datagraph, ibatch, atr_col[indexes], edge_index)



    print("Number of graphs:", nhalos, "out of which ", nhalos_compl, "are complete. Fraction:", float(nhalos_compl)/float(nhalos))

    axdist.set_ylabel("Saliency")
    axdist.set_xlabel("Distance [kpc/h]")
    #axdist.set_xscale("log")
    #axdist.set_yscale("log")
    figdist.savefig("Plots/distance_attribute_"+method+".pdf")
    plt.close(figdist)

    # Feature importance plot
    importances = np.abs(np.array(atrs_feat).mean(0))
    feature_names = [r"$x$", r"$y$", r"$z$", r"$M_*$", r"$v$", r"$R_*$"]

    np.savetxt("Outputs/feature_importance.txt",importances)
    feature_importance_plot(importances, feature_names, method)


# Routine to load the trained model and examine its interpretability
def captum_main(params, verbose = True):

    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simsuite, simset, n_sims = params

    # Load data and create dataset
    dataset, node_features = create_dataset(simsuite, simset, n_sims)

    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Initialize model
    model = ModelGNN(use_model, node_features, n_layers, k_nn)
    model.to(device)
    if verbose: print("Model: " + namemodel(params)+"\n")

    state_dict = torch.load("Models/"+namemodel(params), map_location=device)
    model.load_state_dict(state_dict)

    captum_routine(model, test_loader, k_nn)



#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load default parameters
    from Hyperparameters.params_TNG_CV import params

    captum_main(params)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
