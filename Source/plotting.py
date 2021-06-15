import matplotlib.pyplot as plt
from Source.params import *
#from pysr import pysr, best, best_tex, get_hof
from sklearn.metrics import r2_score


# Plot loss trends
def plot_losses(train_losses, valid_losses, test_loss, err_min, params):

    epochs = params[5]
    plt.plot(range(epochs), train_losses, "r-",label="Training")
    plt.plot(range(epochs), valid_losses, "b:",label="Validation")
    plt.legend()
    plt.yscale("log")
    #plt.title(f"Minimum relative error: {err_min:.2e}")
    plt.title(f"Test loss: {test_loss:.2e}, Minimum relative error: {err_min:.2e}")
    plt.savefig("Plots/loss_"+namemodel(params)+".png", bbox_inches='tight', dpi=300)
    plt.close()

# Scatter plot of true vs predicted properties
def plot_out_true_scatter(params):

    # Dataset
    outputs = np.load("Outputs/outputs_"+namemodel(params)+".npy")
    trues = np.load("Outputs/trues_"+namemodel(params)+".npy")

    # There is a (0,0) point, fix it
    outputs = outputs[1:]
    trues = trues[1:]

    plt.plot(trues, trues, "r-")
    plt.scatter(trues, outputs, color="b", s=0.1)
    #plt.plot(trues,outputs,"bo",markersize=0.1)

    # Mean and std in bins
    inds = trues.argsort()
    trues = trues[inds]
    outputs = outputs[inds]

    bins, binsize = np.linspace(trues[0],trues[-1],10,retstep=True)
    #bins = np.linspace(np.amin(trues),np.amax(trues),10)

    means, stds, meanstrue = [], [], []
    for i, bin in enumerate(bins[:-1]):
        cond = (trues>=bin) & (trues<bins[i+1])
        outbin = outputs[cond]
        means.append(outbin.mean()); stds.append(outbin.std())

    means, stds = np.array(means), np.array(stds)

    plt.errorbar(bins[:-1]+binsize/2., means, yerr=stds, color="purple", marker="o", markersize=2, zorder=10)

    #err = np.abs(trues - outputs)/trues
    #plt.title(model+", Relative error: {:.2e}".format(err.mean()))
    plt.title(params[0]+r", $R^2$={:.2f}".format(r2_score(trues,outputs)))
    #plt.ylabel("Predicted Mass")
    plt.ylabel(r"log$_{10}\left(M_{h,pred}/(10^{10} M_\odot)\right)$")
    #plt.xlabel("True Mass")
    plt.xlabel(r"log$_{10}\left(M_{h,true}/(10^{10} M_\odot)\right)$")
    plt.savefig("Plots/out_true_"+namemodel(params)+".png", bbox_inches='tight', dpi=300)
    plt.close()

# Visualization routine
def visualize_points(data, edge_index=None, index=None):

    pos = data.x[:,:2]
    c_o_m = data.y
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)

    plt.axis('off')
    #plt.savefig("loss.pdf")
    plt.show()

# Plot total stellar mass versus halo mass
def scat_plot(shmasses, hmasses, simtype, simset):
    fig_scat, ax_scat = plt.subplots()
    ax_scat.scatter(shmasses, hmasses, color="r", s=0.1)#, label="Total mass of subhalos")
    #ax_scat.set_xlabel("Sum of stellar mass per halo")
    #ax_scat.set_ylabel("Halo mass")
    ax_scat.set_xlabel(r"log$_{10}\sum_{i}\left(M_{i,*}/(10^{10} M_\odot)\right)$")
    ax_scat.set_ylabel(r"log$_{10}\left(M_{h}/(10^{10} M_\odot)\right)$")
    fig_scat.savefig("Plots/scat_"+simtype+"_"+simset)
    plt.close(fig_scat)

# Histogram of number os subhalos per halos
def plot_histogram(hist, simtype, simset):
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(hist,bins=20)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Number of subhalos per halo")
    ax_hist.set_ylabel("Number of halos")
    fig_hist.savefig("Plots/histogram_"+simtype+"_"+simset)
    plt.close(fig_hist)
