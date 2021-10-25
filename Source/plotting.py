import matplotlib.pyplot as plt
from Source.constants import *
#from pysr import pysr, best, best_tex, get_hof
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator


# Plot loss trends
def plot_losses(train_losses, valid_losses, test_loss, err_min, params):

    epochs = params[5]
    plt.plot(range(epochs), np.exp(train_losses), "r-",label="Training")
    plt.plot(range(epochs), np.exp(valid_losses), "b:",label="Validation")
    plt.legend()
    plt.yscale("log")
    #plt.title(f"Minimum relative error: {err_min:.2e}")
    plt.title(f"Test loss: {test_loss:.2e}, Minimum relative error: {err_min:.2e}")
    plt.savefig("Plots/loss_"+namemodel(params)+".png", bbox_inches='tight', dpi=300)
    plt.close()

# Scatter plot of true vs predicted properties
def plot_out_true_scatter(params, testsuite = False):

    figscat, axscat = plt.subplots()
    suite, simset = params[7], params[8]
    col = colorsuite(suite)
    if testsuite: col = colorsuite(changesuite(suite))

    # Dataset
    outputs = np.load("Outputs/outputs_"+namemodel(params)+".npy")
    trues = np.load("Outputs/trues_"+namemodel(params)+".npy")
    errors = np.load("Outputs/errors_"+namemodel(params)+".npy")

    # There is a (0,0) point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    errors = errors[1:]

    # Use log(M) rather than log(M/1e10)
    outputs += 10.; trues+=10.;

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)), axis=0)
    #chi2 = np.mean( (outputs-trues)**2./errors**2., axis=0 )
    chi2s = (outputs-trues)**2./errors**2.
    #for i, chisq in enumerate(chi2s):
    #    if chisq>10.:    print(chisq, errors[i], trues[i], outputs[i])
    #chi2 = chi2s[np.abs(errors)>0.01].mean()    # Remove some outliers which make explode the chi2
    chi2 = chi2s[chi2s<1.e4].mean()    # Remove some outliers which make explode the chi2
    print("R^2={:.2f}, Relative error={:.2e}, Chi2={:.2f}".format(r2, err_rel, chi2))


    # Sort by true value
    indsort = trues.argsort()
    outputs, trues, errors = outputs[indsort], trues[indsort], errors[indsort]

    # Compute mean and std region
    truebins, binsize = np.linspace(trues[0], trues[-1], num=10, retstep=True)
    means, stds = [], []
    for i, bin in enumerate(truebins[:-1]):
        cond = (trues>=bin) & (trues<bin+binsize)
        outbin = outputs[cond]
        if len(outbin)==0:
            outmean, outstd = np.nan, np.nan    # Avoid error message from some bins without points
        else:
            outmean, outstd = outbin.mean(), outbin.std()
        means.append(outmean); stds.append(outstd)
    means, stds = np.array(means), np.array(stds)
    means -= (truebins[:-1]+binsize/2.)
    axscat.fill_between(truebins[:-1]+binsize/2., means-stds, means+stds, color=col, alpha=0.2)
    axscat.plot(truebins[:-1]+binsize/2., means, color=col, linestyle="--")
    print("Std in bins:",stds[~np.isnan(stds)].mean(),"Mean predicted uncertainty:", np.abs(errors.mean()))

    # Take 200 elements to plot randomly chosen
    indexes = np.random.choice(trues.shape[0], 200, replace=False)
    outputs = outputs[indexes]
    trues = trues[indexes]
    errors = errors[indexes]

    #plt.plot(trues, trues, "r-")
    #plt.scatter(trues, outputs, color="b", s=0.1)
    #plt.plot(trues,outputs,"bo",markersize=0.1)
    #plt.errorbar(trues, outputs, yerr=errors, color="b", marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)
    truemin, truemax = 10.5, 14.  # trues.min(), trues.max()
    axscat.plot([truemin, truemax], [0., 0.], "r-")
    axscat.errorbar(trues, outputs-trues, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)

    """
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
    """
    leg = "$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)
    at = AnchoredText(leg, frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)

    #err = np.abs(trues - outputs)/trues
    #plt.title(model+", Relative error: {:.2e}".format(err.mean()))
    #plt.title(params[0]+r", $R^2$={:.2f}".format(r2))
    #plt.title(r"$log_{10}\left[M_h/(10^{10} M_\odot/h) \right]$"+", \t"+"$R^2$={:.2f}".format(r2)+", \t"+"$\epsilon$={:.1f} \%".format(100.*err_rel))
    #axscat.set_title(r"$log_{10}\left[M_h/(M_\odot/h) \right]$"+", \t"+"$R^2$={:.2f}".format(r2)+", \t"+"$\epsilon$={:.1f} %".format(100.*err_rel))
    #plt.ylabel(r"log$_{10}\left(M_{h,infer}/(10^{10} M_\odot)\right)$")
    #plt.ylabel(r"log$_{10}\left(M_{h,infer}/(10^{10} M_\odot)\right)$ - log$_{10}\left(M_{h,truth}/(10^{10} M_\odot)\right)$")
    #plt.xlabel(r"log$_{10}\left(M_{h,truth}/(10^{10} M_\odot)\right)$")
    axscat.set_xlim([truemin, truemax])
    axscat.set_ylim([-1.,1.])
    axscat.set_ylabel(r"Prediction - Truth")
    axscat.set_xlabel(r"Truth")
    axscat.yaxis.set_major_locator(MultipleLocator(0.2))
    axscat.grid()


    if testsuite:
        titlefig = "Training in "+changesuite(suite)+" "+simset+", testing in "+suite+" "+simset
        namefig = "out_true_testsuite_"+namemodel(params)
    else:
        titlefig = "Training in "+suite+" "+simset+", testing in "+suite+" "+simset
        namefig = "out_true_"+namemodel(params)
    axscat.set_title(titlefig)

    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)

# Visualize saliency graphs
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
            ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.05, color='black')

    sizes = 10.**(massstar+2.)
    scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, c=colors, zorder=1000, vmin=0., vmax=0.16)
    colbar = plt.colorbar(scat, ax=ax, fraction=0.04, pad=0.1)
    colbar.ax.set_ylabel("Saliency")#, loc="top")# rotation=270)
    #ax.scatter(0., 0., 0., s=10., c="red", zorder=10000)
    ax.set_xlabel(r"$x$ [kpc/h]")
    ax.set_ylabel(r"$y$ [kpc/h]")
    ax.set_zlabel(r"$z$ [kpc/h]")
    #ax.xaxis.set_major_locator(MultipleLocator(100))
    #ax.yaxis.set_major_locator(MultipleLocator(100))
    #ax.zaxis.set_major_locator(MultipleLocator(100))

    #plt.axis('off')
    fig.savefig("Plots/visualize_graph_"+str(ind)+".pdf", bbox_inches='tight', dpi=300)
    plt.close(fig)

# Plot total stellar mass versus halo mass
def scat_plot(shmasses, hmasses, simsuite, simset):

    shmasses, hmasses = np.array(shmasses), np.array(hmasses)
    indexes = shmasses.argsort()
    shmasses, hmasses = shmasses[indexes], hmasses[indexes]
    starmassbins, binsize = np.linspace(shmasses[0], shmasses[-1], num=10, retstep=True)

    means, stds = [], []
    for i, bin in enumerate(starmassbins[:-1]):
        cond = (shmasses>=bin) & (shmasses<starmassbins[i+1])
        outbin = hmasses[cond]
        means.append(outbin.mean()); stds.append(outbin.std())

    means, stds = np.array(means), np.array(stds)

    fig_scat, ax_scat = plt.subplots()
    ax_scat.scatter(shmasses, hmasses, color="r", s=0.1)#, label="Total mass of subhalos")

    ax_scat.errorbar(starmassbins[:-1]+binsize/2., means, yerr=stds, color="purple", marker="o", markersize=2)
    ax_scat.fill_between(starmassbins[:-1]+binsize/2., means-stds, means+stds, color="purple", alpha=0.2)

    #ax_scat.set_xlabel("Sum of stellar mass per halo")
    #ax_scat.set_ylabel("Halo mass")
    ax_scat.set_xlabel(r"log$_{10}\sum_{i}\left(M_{i,*}/(10^{10} M_\odot)\right)$")
    ax_scat.set_ylabel(r"log$_{10}\left(M_{h}/(10^{10} M_\odot)\right)$")
    fig_scat.savefig("Plots/scat_"+simsuite+"_"+simset, bbox_inches='tight', dpi=300)
    plt.close(fig_scat)

# Histogram of number os subhalos per halos
def plot_histogram(hist, simsuite, simset):
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(hist,bins=20)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Number of subhalos per halo")
    ax_hist.set_ylabel("Number of halos")
    fig_hist.savefig("Plots/histogram_"+simsuite+"_"+simset, bbox_inches='tight', dpi=300)
    plt.close(fig_hist)
