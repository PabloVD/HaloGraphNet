import matplotlib.pyplot as plt
import h5py
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import Compose, RandomRotate
from Source.constants import *
from Source.plotting import *
from Source.load_data import *
from matplotlib.lines import Line2D
from pysr import pysr, best, best_tex, get_hof
from sklearn.metrics import r2_score

hred = 0.7


cols = ["cyan", "purple"]



# Load data and create the dataset
# simsuite: type of simulation, either "IllustrisTNG" or "SIMBA"
# simset: set of simulations:
#       LH: Use simulations over latin-hypercube, varying over cosmological and astrophysical parameters, and different random seeds (1000 simulations total)
#       CV: Use simulations with fiducial cosmological and astrophysical parameters, but different random seeds (27 simulations total)
# n_sims: number of simulations, maximum 27 for CV and 1000 for LH
def create_dataset_plot(simset = "CV", n_sims = 27):

    fig_scat, (ax_starmass, ax_vel, ax_hmR) = plt.subplots(1,3, figsize=(12,3), sharey=True)
    fig_scat.subplots_adjust(wspace=0)
    fig_hist, ax_hist = plt.subplots()

    customlegend = []

    for ii, simsuite in enumerate(["SIMBA", "IllustrisTNG"]):

        simpath = simpathroot + simsuite + "/"+simset+"_"
        print("Using "+simsuite+" simulations, set "+simset)

        hist = []
        hmasses = []
        shmasses = []
        vels = []
        hmRs = []
        mags = []
        symbreg_err = []
        tottrue, totsym = [], []

        velsmeansimilarMW = []
        starmasssimilarMW = []
        halomasssimilarMW = []

        for sim in range(n_sims):

            # To see ls of columns of file, type in shell: h5ls -r fof_subhalo_tab_033.hdf5
            path = simpath + str(sim)+"/fof_subhalo_tab_033.hdf5"

            tab, HaloMass, HaloPos, HaloVel, halolist = general_tab(path)

            for ind in halolist:

                # Select subhalos within a halo with index ind
                tab_halo = tab[tab[:,0]==ind][:,1:]

                # Write the subhalo positions as the relative position to the host halo
                tab_halo[:,0:3] -= HaloPos[ind]
                tab_halo[:,-3:] -= HaloVel[ind]

                if tab_halo.shape[0]>1:
                #if HaloMass[ind]>=1.:

                    # Compute the modulus of the velocities and create a new table with these values
                    subhalovel = np.sqrt(np.sum(tab_halo[:,-3:]**2., 1))

                    num_subhalos = tab_halo.shape[0]
                    hist.append(num_subhalos)
                    hmasses.append(np.log10(HaloMass[ind]))
                    shmasses.append(np.log10(np.sum(10.**np.array(tab_halo[:,3]))))
                    subhalovel = subhalovel[1:].mean()
                    #print(subhalovel)
                    hmR = tab_halo[:,4].mean()
                    vels.append(subhalovel), hmRs.append(hmR)
                    #mags.append( tab_halo[:,5].mean() )

                    """if (HaloMass[ind]>0.9e2*hred) and (HaloMass[ind]<1.5e2*hred):
                    #if (10.**shmasses[-1]>4.*hred) and (10.**shmasses[-1]<6.*hred):
                        #print("Mean vel", subhalovel)
                        velsmeansimilarMW.append(subhalovel)"""

                    """if (subhalovel>150) and (subhalovel<250):
                        starmasssimilarMW.append(10.**shmasses[-1]/hred)
                        halomasssimilarMW.append(HaloMass[ind]/hred)"""



        scat_plot(shmasses, hmasses, simsuite, simset, ax_starmass, ii)
        scat_plot(np.log10(vels), hmasses, simsuite, simset, ax_vel, ii)
        scat_plot(np.log10(hmRs), hmasses, simsuite, simset, ax_hmR, ii)
        #scat_plot(mags, hmasses, simsuite, simset, ax_hmR, ii)
        #mags = np.array(mags)
        #print(len(mags[mags>-14.]), len(mags[mags<-14.]))

        #fit(shmasses, hmasses, ax_starmass, ii)

        ax_hist.hist(hist,bins=20, color=cols[ii], alpha=0.7)

        customlegend.append(Line2D([0], [0], color=cols[ii], lw=3., linestyle="-", label=simsuite))

        #print(simsuite, "Vel", np.array(velsmeansimilarMW).mean(), np.array(velsmeansimilarMW).std())
        #print(simsuite, "StarMass", np.array(starmasssimilarMW).mean(), np.array(starmasssimilarMW).std())
        #print(simsuite, "HaloMass", np.array(halomasssimilarMW).mean(), np.array(halomasssimilarMW).std())

    ax_starmass.set_xlabel(r"log$_{10}\left(M_{*}/(10^{10} M_\odot/h)\right)$")
    ax_vel.set_xlabel(r"log$_{10}\left(\bar{v}\right)$ [km/s]")
    ax_hmR.set_xlabel(r"log$_{10}\left(R_{hm}\right)$ [kpc/h]")
    #ax_hmR.set_xlabel(r"M_V")
    #ax_hmR.set_xlim([-30.,0.])

    ax_starmass.legend(handles=customlegend)

    #ax_vel.set_xscale("log")

    ax_starmass.set_ylabel(r"log$_{10}\left(M_{h}/(10^{10} M_\odot/h)\right)$")
    for ax in [ax_starmass, ax_vel, ax_hmR]:
        ax.grid()

    ax_vel.yaxis.set_ticklabels([])
    ax_hmR.yaxis.set_ticklabels([])

    #for ax in [ax_starmass, ax_vel, ax_hmR]:
    #    ax.set_ylabel(r"log$_{10}\left(M_{h}/(10^{10} M_\odot/h)\right)$")
    #    ax.legend(handles=customlegend)
    fig_scat.savefig("Plots/scat_"+simset, bbox_inches='tight', dpi=300)

    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Number of subhalos per halo")
    ax_hist.set_ylabel("Number of halos")
    ax_hist.legend(handles=customlegend)
    fig_hist.savefig("Plots/histogram_"+simset, bbox_inches='tight', dpi=300)


# Plot total stellar mass versus halo mass
def scat_plot(shmasses, hmasses, simsuite, simset, ax_scat, index):

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

    #symbolic_regression(shmasses, hmasses)

    indexes = np.random.choice(hmasses.shape[0], 2000, replace=False)
    ax_scat.scatter(shmasses[indexes], hmasses[indexes], color=cols[index], s=0.1)#, label="Total mass of subhalos")

    ax_scat.errorbar(starmassbins[:-1]+binsize/2., means, yerr=stds, color=cols[index], marker="o", markersize=2)
    ax_scat.fill_between(starmassbins[:-1]+binsize/2., means-stds, means+stds, color=cols[index], alpha=0.2)

    print("Standard deviation",np.mean(stds))

    #ax_scat.set_xlabel("Sum of stellar mass per halo")
    #ax_scat.set_ylabel("Halo mass")


def symbolic_regression(shmasses, hmasses):

    # Learn equations
    equations = pysr(shmasses, hmasses, niterations=10,
        binary_operators=["plus", "mult", "pow"],
        unary_operators=["exp", "log10_abs"],
        #binary_operators=["plus", "sub", "mult", "pow", "div"],
        #unary_operators=["exp", "logm"],
        batching=1, batchSize=128)
    #unary_operators=[ "exp", "abs", "logm", "square", "cube", "sqrtm"])

    print(best(equations))
    print(best_tex(equations))
    print(get_hof())

def fit(x, y, ax_scat, index):

    degree = 4

    pol0 = np.polyfit(x,y,degree)
    pol = np.poly1d(pol0)

    relerr = np.mean(np.abs((pol(x)-y)/y))
    r2 = r2_score(y,pol(x))
    print("Error", relerr, "R^2=", r2)

    #ax_scat.plot(sorted(x), pol(sorted(x)), color=cols[index], linestyle="--")


#--- MAIN ---#

if __name__ == "__main__":

    if not os.path.exists("Plots"): os.mkdir("Plots")

    create_dataset_plot(simset = "CV", n_sims = 27)

    #create_dataset_plot(simset = "LH", n_sims = 1000)
