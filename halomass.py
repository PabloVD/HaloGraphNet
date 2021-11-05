#-------------------------------------
# Apply the GNN already trained for the MW and M31 halos to infer their masses
# Author: Pablo Villanueva Domingo
# Last update: 5/11/21
#-------------------------------------

from main import *
from params_TNG import params as params_TNG
from params_SIMBA import params as params_SIMBA
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation as Rot
from Source.galaxies import MW_gals, M31_gals

params_list = [params_SIMBA, params_TNG]

#--- FUNCTIONS ---#

# Define the graph for a galaxy group
def halo_graph(galaxies, random_rotation=False):

    # pos: kpc, vel: km/s, starmass: Msun
    pos = [[gal.x, gal.y, gal.z] for gal in galaxies]
    starmass = [gal.starmass for gal in galaxies]
    vel = [gal.vel for gal in galaxies]
    #velCM = np.array(np.sum(np.array([gal.starmass*gal.vel3D for gal in galaxies], dtype=object),axis=0))/np.sum(np.array([gal.starmass for gal in galaxies]),axis=0)
    #vel = [np.sqrt(np.sum((np.array(gal.vel3D)-velCM)**2.)) for gal in galaxies]

    """
    # CM position
    xCM = np.array(np.sum(np.array([gal.starmass*gal.x for gal in galaxies]),axis=0))/np.sum(np.array([gal.starmass for gal in galaxies]),axis=0)
    yCM = np.array(np.sum(np.array([gal.starmass*gal.y for gal in galaxies]),axis=0))/np.sum(np.array([gal.starmass for gal in galaxies]),axis=0)
    zCM = np.array(np.sum(np.array([gal.starmass*gal.z for gal in galaxies]),axis=0))/np.sum(np.array([gal.starmass for gal in galaxies]),axis=0)
    print([[gal.x-pos[0][0], gal.y-pos[0][1], gal.z-pos[0][2]] for gal in galaxies])
    """

    # Normalize
    pos = (np.array(pos)-pos[0])*hred/boxsize    # Write position relative to the main galaxy (the first one)
    modvel = np.array(vel)/velnorm
    starmass = np.log10(np.array(starmass)/1.e10*hred)    # Write in units of 1e10 Msun/hred and take log


    # If random_rotation==1, rotate randomly all the galaxies around the center of the halo
    if random_rotation:
        rotmat = Rot.random().as_matrix()
        pos = np.array([rotmat.dot(p) for p in pos])

    if use_vel:
        modvel = np.log10(1.+modvel)
        tab_halo = np.column_stack((pos, starmass, modvel))
    else:
        tab_halo = np.column_stack((pos, starmass))

    # Take as global quantities of the halo the number of subhalos and total stellar mass
    u = np.zeros((1,2), dtype=np.float32)
    u[0,0] = tab_halo.shape[0]  # number of subhalos
    u[0,1] = np.log10(np.sum(10.**starmass))

    node_features = tab_halo.shape[1]

    # Define the graph
    graph = Data(x=torch.tensor(tab_halo, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32), u=torch.tensor(u, dtype=torch.float), batch=torch.tensor(np.zeros(tab_halo.shape[0]), dtype=torch.int64))

    return graph, node_features


# Load a trained model and predict the mass of a halo given its galaxies
def predict_halomass(galaxies, params, verbose=True, random_rotation=False):

    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simtype, simset, n_sims = params
    #simtype = simsuite
    #params[7] = simtype

    graph, node_features = halo_graph(galaxies, random_rotation)
    graph.to(device)

    # Initialize model
    model = ModelGNN(use_model, node_features, n_layers, k_nn)
    model.to(device)
    if verbose: print("\nModel: " + namemodel(params)+"\n")

    state_dict = torch.load("Models/"+namemodel(params), map_location=device)
    model.load_state_dict(state_dict)

    out = model(graph)

    # Mean mass and uncertainty from their logs
    mass, error = 10.**(out[0,0].item())/hred, 10.**(np.abs(out[0,1].item()))
    massmin, massplus = mass/error, mass*error
    massmean, errmin, errplus = mass/1.e2, (mass -massmin)/1.e2, (massplus-mass)/1.e2 # Factor 1.e2 for 1e12Msun units

    if verbose:
        print(galaxies[0].name)
        print("Mass;\tPlus;\tMinus; [1e12Msun]")
        print("{:.3f};\t{:.3f};\t{:.3f}\n".format(massmean, errplus, errmin))

    return massmean, errplus, errmin


# Compute the mass of the halo as a function of the number of satellite galaxies considered
def mass_as_numsats(galaxies_in, subscript, title):

    fignumsat, (axnumsat) = plt.subplots(1, figsize=(6,6))
    marks = ["d", "*"]
    linw = [3, 1]

    for ind, suite in enumerate(["SIMBA", "IllustrisTNG"]):

        galaxies = galaxies_in.copy()

        numsats, massvec, errminvec, errplusvec = [], [], [], []

        print([gal.name for gal in galaxies])
        mass, errplus, errmin = predict_halomass(galaxies, params_list[ind], verbose=False)
        numsats.append(len(galaxies)); massvec.append(mass); errminvec.append(errmin); errplusvec.append(errplus)

        for i in sorted(range(len(galaxies)),reverse=True):
            if len(galaxies)>2:
                galaxies.pop(i)     # Remove the lightest satellite galaxy
                print([gal.name for gal in galaxies])
                mass, errplus, errmin = predict_halomass(galaxies, params_list[ind], verbose=False)
                numsats.append(len(galaxies)); massvec.append(mass); errminvec.append(errmin); errplusvec.append(errplus)

        axnumsat.errorbar(numsats, massvec, yerr=[errminvec, errplusvec], lw=linw[ind], color=colorsuite(suite), marker=marks[ind], markersize=10, label=suite)

    axnumsat.legend()
    axnumsat.grid()
    axnumsat.set_xlabel("Num. of galaxies")
    axnumsat.set_ylabel(r"$M_{\rm "+subscript+",200}\; [10^{12} M_\odot]$")
    axnumsat.xaxis.set_major_locator(MaxNLocator(integer=True))
    axnumsat.set_title(title)
    fignumsat.savefig("Plots/mass_as_numsats_"+title+".png", bbox_inches='tight', dpi=300)


# Check the robustness of the prediction when random rotations are considered
def halomass_with_rotations(galaxies, params, numrots = 50):

    masses = []

    for i in range(numrots):
        mass, errplus, errmin = predict_halomass(galaxies, params, verbose=False, random_rotation=True)
        masses.append(mass)

    masses = np.array(masses)
    print("Results: mean={:.2f}, std={:.2f}, std/mean={:.3f}".format( masses.mean(), masses.std(), masses.std()/masses.mean()) )


#--- MAIN ---#

if __name__=="__main__":


    # Check rotation invariance
    for ind, suite in enumerate(["SIMBA", "IllustrisTNG"]):
        print("\n"+suite+"\n")
        halomass_with_rotations(MW_gals, params_list[ind])
        print("\n")
        halomass_with_rotations(M31_gals, params_list[ind])
        print("\n")


    # Check robustness with different number of satellites
    mass_as_numsats(MW_gals, "MW", "MW")
    mass_as_numsats(M31_gals, "M31", "M31")


    # Compute the mean mass and uncertainty for the galaxies
    for ind, suite in enumerate(["SIMBA", "IllustrisTNG"]):

        # Milky Way halo
        predict_halomass(MW_gals, params_list[ind])

        # Andromeda halo
        predict_halomass(M31_gals, params_list[ind])
