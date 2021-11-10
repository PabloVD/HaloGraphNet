#----------------------------------------------------------------------
# Script for plotting some statistics
# Author: Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import matplotlib.pyplot as plt
from Source.constants import *
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
mpl.rcParams.update({'font.size': 12})

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

    # Load true values and predicted means and standard deviations
    outputs = np.load("Outputs/outputs_"+namemodel(params)+".npy")
    trues = np.load("Outputs/trues_"+namemodel(params)+".npy")
    errors = np.load("Outputs/errors_"+namemodel(params)+".npy")

    # There is a (0,0) initial point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    errors = errors[1:]

    # Use log(M) rather than log(M/1e10)
    outputs += 10.; trues+=10.;

    # Compute the number of points lying within 1 or 2 sigma regions from their uncertainties
    cond_success_1sig, cond_success_2sig = np.abs(outputs-trues)<=np.abs(errors), np.abs(outputs-trues)<=2.*np.abs(errors)
    tot_points = outputs.shape[0]
    successes1sig, successes2sig = outputs[cond_success_1sig].shape[0], outputs[cond_success_2sig].shape[0]

    # Compute the linear correlation coefficient
    r2 = r2_score(trues,outputs)
    err_rel = np.mean(np.abs((trues - outputs)/(trues)), axis=0)
    chi2s = (outputs-trues)**2./errors**2.
    #chi2 = chi2s[np.abs(errors)>0.01].mean()    # Remove some outliers which make explode the chi2
    chi2 = chi2s[chi2s<1.e4].mean()    # Remove some outliers which make explode the chi2
    print("R^2={:.2f}, Relative error={:.2e}, Chi2={:.2f}".format(r2, err_rel, chi2))
    print("A fraction of succeses of", successes1sig/tot_points, "at 1 sigma,", successes2sig/tot_points, "at 2 sigmas")

    # Sort by true value
    indsort = trues.argsort()
    outputs, trues, errors = outputs[indsort], trues[indsort], errors[indsort]

    # Compute mean and std region within several bins
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

    # Plot predictions vs true values
    truemin, truemax = 10.5, 14.  # trues.min(), trues.max()
    axscat.plot([truemin, truemax], [0., 0.], "r-")
    axscat.errorbar(trues, outputs-trues, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)

    # Legend
    leg = "$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)
    at = AnchoredText(leg, frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)

    # Labels etc
    axscat.set_xlim([truemin, truemax])
    axscat.set_ylim([-1.,1.])
    #axscat.set_ylabel(r"Prediction - Truth")
    #axscat.set_xlabel(r"Truth")
    plt.ylabel(r"log$_{10}\left[M_{h,infer}/(M_\odot/h)\right]$ - log$_{10}\left[M_{h,truth}/(M_\odot/h)\right]$")
    plt.xlabel(r"log$_{10}\left[M_{h,truth}/(M_\odot/h)\right]$")
    axscat.yaxis.set_major_locator(MultipleLocator(0.2))
    axscat.grid()

    # Title, indicating which are the training and testing suites
    if testsuite:
        titlefig = "Training in "+changesuite(suite)+" "+simset+", testing in "+suite+" "+simset
        namefig = "out_true_testsuite_"+namemodel(params)
    else:
        titlefig = "Training in "+suite+" "+simset+", testing in "+suite+" "+simset
        namefig = "out_true_"+namemodel(params)
    axscat.set_title(titlefig)

    figscat.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close(figscat)
