#-------------------------------------
# Script for plotting CAMELS data
# Author: Pablo Villanueva Domingo
# Last update: 5/11/21
#-------------------------------------

from Source.load_data import *
from matplotlib.lines import Line2D
#from pysr import pysr, best, best_tex, get_hof
from sklearn.metrics import r2_score
import time, datetime
import matplotlib.patheffects as path_effects
from Source.galaxies import MW_gals, M31_gals


# 1 for writing positions and velocities in the central galaxy rest frame
galcen_frame = 1


# Scatter plot of two quantities (e.g. total stellar mass versus total halo mass)
def scat_plot(shmasses, hmasses, simsuite, simset, ax_scat):

    shmasses, hmasses = np.array(shmasses), np.array(hmasses)
    indexes = shmasses.argsort()
    shmasses, hmasses = shmasses[indexes], hmasses[indexes]
    starmassbins, binsize = np.linspace(shmasses[0], shmasses[-1], num=10, retstep=True)

    means, stds = [], []
    for i, bin in enumerate(starmassbins[:-1]):
        cond = (shmasses>=bin) & (shmasses<starmassbins[i+1])
        outbin = hmasses[cond]
        if len(outbin)==0:
            outmean, outstd = np.nan, np.nan    # Avoid error message from some bins without points
        else:
            outmean, outstd = outbin.mean(), outbin.std()
        means.append(outmean); stds.append(outstd)

    means, stds = np.array(means), np.array(stds)

    #symbolic_regression(shmasses, hmasses)

    indexes = np.random.choice(hmasses.shape[0], 2000, replace=False)
    ax_scat.scatter(shmasses[indexes], hmasses[indexes], color=colorsuite(simsuite), s=0.1)#, label="Total mass of subhalos")

    #ax_scat.errorbar(starmassbins[:-1]+binsize/2., means, yerr=stds, color=colorsuite(simsuite), marker="o", markersize=2)
    ax_scat.fill_between(starmassbins[:-1]+binsize/2., means-stds, means+stds, color=colorsuite(simsuite), alpha=0.2)

    print("Standard deviation",np.mean(stds))

    #ax_scat.set_xlabel("Sum of stellar mass per halo")
    #ax_scat.set_ylabel("Halo mass")


# Velocity vs stellar mass plot
def vel_vs_starmass_plot(simset, fig_vmass, ax_vmass):

    markers = ["d", "*"]
    #colgals = ["limegreen","orange"]
    colgals = ["blue", "red"]

    for j, gals in enumerate([MW_gals, M31_gals]):
        velCM = np.array(np.sum(np.array([gal.starmass*gal.vel3D for gal in gals], dtype=object),axis=0))/np.sum(np.array([gal.starmass for gal in gals]),axis=0)
        #for gal in gals:
        for gal in gals[1:]:
            #vel = np.sqrt(np.sum((np.array(gal.vel3D)-velCM)**2.))
            vel = np.sqrt(np.sum((np.array(gal.vel3D)-gals[0].vel3D)**2.))
            ax_vmass.scatter(gal.starmass, vel, s=10., marker=markers[j], color=colgals[j], alpha=1. )
            txt = ax_vmass.text(gal.starmass*1.1, vel, gal.name, color=colgals[j])
            txt.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),path_effects.Normal()])

    ax_vmass.set_xlabel(r"$M_*$ [$M_\odot$]")
    ax_vmass.set_ylabel(r"$v$ [km/s]")
    ax_vmass.set_xscale("log")
    ax_vmass.set_ylim([-10.,500.])
    ax_vmass.set_xlim([1.e8,6.e11])
    ax_vmass.set_axisbelow(True)
    ax_vmass.grid(zorder=-1)

    customlegend = []
    for ii, simsuite in enumerate(["SIMBA", "IllustrisTNG"]):
        customlegend.append( Line2D([0], [0], color=colorsuite(simsuite), marker=".", linestyle='None', label=simsuite) )
    for j, gals in enumerate([MW_gals, M31_gals]):
        customlegend.append( Line2D([0], [0], marker=markers[j], color=colgals[j], linestyle='None', label=gals[0].name+" group") )
    ax_vmass.legend(handles=customlegend)

    fig_vmass.savefig("Plots/vel_vs_starmass_"+simset, bbox_inches='tight', dpi=300)
    plt.close(fig_vmass)

# Velocity vs distance to the center plot
def vel_vs_distance_plot(simset, fig_vdist, ax_vdist):

    markers = ["d", "*"]
    #colgals = ["limegreen","orange"]
    colgals = ["blue", "red"]

    for j, gals in enumerate([MW_gals, M31_gals]):
        #for gal in gals:
        for gal in gals[1:]:
            #vel = np.sqrt(np.sum((np.array(gal.vel3D)-velCM)**2.))
            vel = np.sqrt(np.sum((np.array(gal.vel3D)-gals[0].vel3D)**2.))
            dist = np.sqrt( (gal.x-gals[0].x)**2. + (gal.y-gals[0].y)**2. + (gal.z-gals[0].z)**2. )
            ax_vdist.scatter(dist, vel, s=10., marker=markers[j], color=colgals[j], alpha=1. )
            txt = ax_vdist.text(dist*1.1, vel, gal.name, color=colgals[j])
            txt.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),path_effects.Normal()])

    ax_vdist.set_xlabel(r"$Distance$ [$kpc$]")
    ax_vdist.set_ylabel(r"$v$ [km/s]")
    ax_vdist.set_xscale("log")
    ax_vdist.set_ylim([-10.,500.])
    #ax_vdist.set_xlim([0.,1.e3])
    ax_vdist.set_axisbelow(True)
    ax_vdist.grid(zorder=-1)

    customlegend = []
    for ii, simsuite in enumerate(["SIMBA", "IllustrisTNG"]):
        customlegend.append( Line2D([0], [0], color=colorsuite(simsuite), marker=".", linestyle='None', label=simsuite) )
    for j, gals in enumerate([MW_gals, M31_gals]):
        customlegend.append( Line2D([0], [0], marker=markers[j], color=colgals[j], linestyle='None', label=gals[0].name+" group") )
    ax_vdist.legend(handles=customlegend)

    fig_vdist.savefig("Plots/vel_vs_dist_"+simset, bbox_inches='tight', dpi=300)
    plt.close(fig_vdist)

# Perform symbolic regression (import pysr for that above)
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

# Get polynomial fit
def fit(x, y):

    degree = 4

    pol0 = np.polyfit(x,y,degree)
    pol = np.poly1d(pol0)

    relerr = np.mean(np.abs((pol(x)-y)/y))
    r2 = r2_score(y,pol(x))
    print("Fit: rel. error=", relerr, ", R^2=", r2)

    return sorted(x), pol(sorted(x))


# Load data and plot some figures
# See create_dataset function in Source/load_data for more details
def summary_plots(simset = "CV", n_sims = 27):

    #fig_scat, (ax_starmass, ax_vel, ax_hmR) = plt.subplots(1,3, figsize=(12,3), sharey=True)
    #fig_scat.subplots_adjust(wspace=0)
    fig_scat, ax_starmass = plt.subplots()
    fig_hist, ax_hist = plt.subplots()
    fig_vmass, ax_vmass = plt.subplots()
    fig_vdist, ax_vdist = plt.subplots()
    fig_occ, ax_occ = plt.subplots()

    customlegend = []

    for ii, simsuite in enumerate(["SIMBA", "IllustrisTNG"]):

        simpath = simpathroot + simsuite + "/"+simset+"_"
        print("Using "+simsuite+" simulations, set "+simset)

        hist = []
        hmasses = []
        shmasses = []
        vels = []
        hmRs = []
        maxdist = []
        galvels, starmasses, galdists = [], [], []

        for sim in range(n_sims):

            # To see ls of columns of file, type in shell: h5ls -r fof_subhalo_tab_033.hdf5
            path = simpath + str(sim)+"/fof_subhalo_tab_033.hdf5"

            tab, HaloMass, HaloPos, HaloVel, halolist = general_tab(path)

            for ind in halolist:

                # Select subhalos within a halo with index ind
                tab_halo = tab[tab[:,0]==ind][:,1:]

                if tab_halo.shape[0]>1:

                    # If galcen_frame==1, write positions and velocities in the rest frame of the central galaxy
                    if galcen_frame:
                        tab_halo[:,:3] -= tab_halo[0,:3]
                        tab_halo[:,-3:] -= tab_halo[0,-3:]
                    else:
                        # Write the positions and velocities as the relative position and velocity to the host halo
                        tab_halo[:,0:3] -= HaloPos[ind]
                        tab_halo[:,-3:] -= HaloVel[ind]

                    # Correct periodic boundary effects
                    tab_halo[:,:3] = correct_boundary(tab_halo[:,:3])

                    # Compute the modulus of the velocities and create a new table with these values
                    subhalovel = np.sqrt(np.sum(tab_halo[:,-3:]**2., 1))

                    num_subhalos = tab_halo.shape[0]
                    hist.append(num_subhalos)
                    hmasses.append(np.log10(HaloMass[ind]))
                    shmasses.append(np.log10(np.sum(10.**np.array(tab_halo[:,3]))))

                    # Get features for MW-like halos
                    if (HaloMass[ind]>0.8e2*hred) and (HaloMass[ind]<1.5e2*hred):
                        galvels.extend( subhalovel[1:] ); starmasses.extend(tab_halo[1:,3]); galdists.extend(np.sqrt(tab_halo[1:,0]**2. + tab_halo[1:,1]**2. + tab_halo[1:,2]**2.))
                        #galvels.extend( subhalovel[:] ); starmasses.extend(tab_halo[:,3])

                    subhalovel = subhalovel[1:].mean()
                    hmR = tab_halo[:,4].mean()
                    vels.append(subhalovel), hmRs.append(hmR)
                    maxdist.append( np.max(np.sqrt(np.sum(tab_halo[:,0:3]**2.,1))) )


        shmasses=np.array(shmasses); hmasses=np.array(hmasses);
        # Use log(M) rather than log(M/1e10)
        shmasses+=10.; hmasses+=10.;

        scat_plot(shmasses, hmasses, simsuite, simset, ax_starmass)
        #scat_plot(np.log10(vels), hmasses, simsuite, simset, ax_vel)
        #scat_plot(np.log10(hmRs), hmasses, simsuite, simset, ax_hmR)

        starmasses, galvels, galdists = np.array(starmasses)+10.,  np.array(galvels)*velnorm, np.array(galdists)*boxsize
        ax_vmass.scatter(10.**(starmasses)/hred, galvels, s=0.1, color=colorsuite(simsuite), marker="o", alpha=0.2 )
        ax_vdist.scatter(galdists/hred, galvels, s=0.1, color=colorsuite(simsuite), marker="o", alpha=0.2 )

        xfit, yfit = fit(shmasses, hmasses)
        ax_starmass.plot(xfit, yfit, color=colorsuite(simsuite), linestyle="--")

        #ax_hist.hist(hist,bins=20, color=colorsuite(simsuite), alpha=0.7)
        ax_hist.hist(hist,bins=np.logspace(np.log10(2), np.log10(700),15), color=colorsuite(simsuite), alpha=0.7)
        #ax_hist.hist(maxdist,bins=50, color=colorsuite(simsuite), alpha=0.7)

        customlegend.append(Line2D([0], [0], color=colorsuite(simsuite), lw=3., linestyle="-", label=simsuite))

        ax_occ.scatter(hmasses, hist, color=colorsuite(simsuite), s=0.1)

    # Scatter plots

    ax_starmass.set_ylabel(r"log$_{10}\left[M_{h}/(M_\odot/h)\right]$")
    ax_starmass.set_xlabel(r"log$_{10}\left[M_{*,tot}/(M_\odot/h)\right]$")
    #ax_vel.set_xlabel(r"log$_{10}\left(\bar{v}\right)$ [km/s]")
    #ax_hmR.set_xlabel(r"log$_{10}\left(R_{hm}\right)$ [kpc/h]")
    #ax_hmR.set_xlabel(r"M_V")
    #ax_hmR.set_xlim([-30.,0.])
    ax_starmass.set_xlim([8.,13.])
    ax_starmass.set_ylim([10.,14.5])
    #ax_vel.yaxis.set_ticklabels([])
    #ax_hmR.yaxis.set_ticklabels([])
    #ax_vel.set_xscale("log")
    ax_starmass.grid()
    ax_starmass.set_title(simset+" set")
    ax_starmass.legend(handles=customlegend)

    #for ax in [ax_starmass, ax_vel, ax_hmR]:
    #    ax.grid()
    #for ax in [ax_starmass, ax_vel, ax_hmR]:
    #    ax.set_ylabel(r"log$_{10}\left(M_{h}/(10^{10} M_\odot/h)\right)$")
    #    ax.legend(handles=customlegend)

    fig_scat.savefig("Plots/scat_"+simset, bbox_inches='tight', dpi=300)

    # Histogram
    ax_hist.set_yscale("log")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Number of galaxies per halo")
    ax_hist.set_ylabel("Number of halos")
    ax_hist.legend(handles=customlegend)
    fig_hist.savefig("Plots/histogram_"+simset, bbox_inches='tight', dpi=300)

    # Velocity vs stellar mass plot
    vel_vs_starmass_plot(simset, fig_vmass, ax_vmass)

    # Velocity vs distance plot
    vel_vs_distance_plot(simset, fig_vdist, ax_vdist)

    # Occupancy plot
    ax_occ.legend(handles=customlegend)
    ax_occ.set_xlabel(r"log$_{10}\left[M_{h}/(M_\odot/h)\right]$")
    ax_occ.set_ylabel("Number of galaxies per halo")
    ax_occ.set_yscale("log")
    fig_occ.savefig("Plots/occupancy_"+simset, bbox_inches='tight', dpi=300)



#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    if not os.path.exists("Plots"): os.mkdir("Plots")

    summary_plots(simset = "CV", n_sims = 27)

    summary_plots(simset = "LH", n_sims = 1000)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
