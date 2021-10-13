#--------
# Apply the GNN already trained for the MW, M31 and its most massive satellite galaxies
#--------

from main import *
from params import params
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, Galactic
#from astropy.coordinates import galactocentric_frame_defaults
#galactocentric_frame_defaults.set('pre-v4.0')
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation as Rot


frame_cmb = 0

# Reduced Hubble constant
hred = 0.7

# Galaxy class
class galaxy():
    def __init__(self, name, pos, vel, starmass, velcen, use_cart=0, use_pm=0):

        # Name of the galaxy
        self.name = name

        # If use_cart==1, read pos as galactocentric cartesian positions x,y,z in kpc (centered in the Milky Way frame)
        if use_cart:
            self.x = pos[0]
            self.y = pos[1]
            self.z = pos[2]

            # Convert to galactic coordinates l,b,distance (centered in the Sun)
            galcoords = SkyCoord(x=self.x*u.kpc, y=self.y*u.kpc, z=self.z*u.kpc, frame='galactocentric')
            galcen = galcoords.transform_to(Galactic)

            self.l = galcen.l.value
            self.b = galcen.b.value
            self.D = galcen.distance.value

        # If use_cart==1, read pos as galactic coordinates (centered in the Sun): l, b in degrees, distance in kpc
        else:
            self.l = pos[0]
            self.b = pos[1]
            self.D = pos[2]

            # Create an astropy SkyCoord object to transform to galactocentric cartesian coordinates
            galcoords = SkyCoord(l=self.l*u.degree, b=self.b*u.degree, distance=self.D*u.kpc, frame="galactic")
            galcen = galcoords.transform_to(Galactocentric)

            # Galactocentric cartesian coordinates (kpc)
            self.x = galcen.x.value
            self.y = galcen.y.value
            self.z = galcen.z.value


        # If use_pm, employ proper motion and radial velocity to compute cartesian galactocentric velocity.
        # vel must be a vector [pm_ra_cosdec, pm_dec, radial_velocity]
        # Then, use ICRS frame, and l and b are ra and dec instead
        """elif use_pm:
            # Create an astropy SkyCoord object in ICRS to transform to galactocentric cartesian coordinates
            galcoords = SkyCoord(ra=l, dec=b, distance=D*u.kpc, frame="icrs",
                                pm_ra_cosdec=vel[0]*u.mas/u.yr,
                                pm_dec=vel[1]*u.mas/u.yr,
                                radial_velocity=vel[2]*u.km/u.s,)
            galcen = galcoords.transform_to(Galactocentric)


            # Galactocentric cartesian coordinates (kpc)
            self.x = galcen.x.value
            self.y = galcen.y.value
            self.z = galcen.z.value

            self.vel = np.sqrt(galcen.v_x.value**2. + galcen.v_y.value**2. + galcen.v_z.value**2.)"""


        # Modulus of the peculiar velocity
        #if not use_pm:
        # If just a number is given, it is assumed as the velocity with respect the central halo
        if type(vel)=="float":
            self.vel = vel
        else:
            #if frame_cmb:
            #    vel_GC_CMB = np.array([-33.9, -484.2, 268.7])   # Galactic center velocity respect to the CMB arXiv:0711.3809
            #    vel = np.array(vel) - vel_GC_CMB
            self.vel = np.sqrt(np.sum((np.array(vel)-np.array(velcen))**2.))
            #self.vel = vel

        # Stellar mass (in Msun)
        self.starmass = starmass

# Visualization routine
def visualize_galaxies(MW_gals, M31_gals):

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection ="3d")
    col_gal = ["red", "blue"]

    for i, galaxies in enumerate([MW_gals, M31_gals]):

        pos = np.array([[gal.x, gal.y, gal.z] for gal in galaxies])

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=50, zorder=1000, color=col_gal[i])

    #plt.axis('off')
    fig.savefig("Plots/galaxies.png", bbox_inches='tight', dpi=300)


# Define the graph for the MW and its satellite galaxies
def halo_graph(galaxies, random_rotation=False):

    # pos: kpc, vel: km/s, starmass: Msun
    pos = [[gal.x, gal.y, gal.z] for gal in galaxies]
    vel = [gal.vel for gal in galaxies]
    starmass = [gal.starmass for gal in galaxies]

    # Normalize
    pos = (np.array(pos)-pos[0])/boxsize    # Write position relative to the main galaxy (the first one)
    modvel = np.array(vel)/velnorm
    starmass = np.log10(np.array(starmass)/1.e10*hred)    # Write in units of 1e10 Msun/hred and take log

    # Rotate randomly all the galaxies around the center of the halo
    if random_rotation:
        rotmat = Rot.random().as_matrix()
        pos = np.array([rotmat.dot(p) for p in pos])

    tab_halo = np.column_stack((pos, starmass, modvel))

    # Take as global quantities of the halo the number of subhalos and total stellar mass
    u = np.zeros((1,2), dtype=np.float32)
    u[0,0] = tab_halo.shape[0]  # number of subhalos
    u[0,1] = np.log10(np.sum(10.**starmass))

    node_features = tab_halo.shape[1]

    graph = Data(x=torch.tensor(tab_halo, dtype=torch.float32), pos=torch.tensor(pos, dtype=torch.float32), u=torch.tensor(u, dtype=torch.float), batch=torch.tensor(np.zeros(tab_halo.shape[0]), dtype=torch.int64))

    return graph, node_features


# Load a trained model and predict the mass of a halo given its galaxies
def predict_halomass(galaxies, simsuite, verbose=True, random_rotation=False):

    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simtype, simset, n_sims = params
    simtype = simsuite
    params[7] = simtype

    graph, node_features = halo_graph(galaxies, random_rotation)
    graph.to(device)

    # Initialize model
    model = ModelGNN(use_model, node_features, n_layers, k_nn)
    model.to(device)
    if verbose: print("Model: " + namemodel(params)+"\n")

    state_dict = torch.load("Models/"+namemodel(params), map_location=device)
    model.load_state_dict(state_dict)

    out = model(graph)
    #print(out)

    mass, error = 10.**(out[0,0].item())/hred, 10.**(out[0,1].item())
    massmin, massplus = mass/error, mass*error
    massmean, errmin, errplus = mass/1.e2, (mass -massmin)/1.e2, (massplus-mass)/1.e2 # Factor 1.e2 for 1e12Msun units
    if verbose:
        print(galaxies[0].name)
        #print("M/(1e12*Msun)=",mass)
        #print("Error minus/plus [1e12Msun]", errmin, errplus)
        print("Mass;\tPlus;\tMinus; [1e12Msun]", errmin, errplus)
        print("{:.3f};\t{:.3f};\t{:.3f};".format(massmean, errplus, errmin))

    return massmean, errplus, errmin


# Compute the mass of the halo as a function of the number of satellite galaxies considered
def mass_as_numsats(galaxies_in, subscript, title):

    fignumsat, (axnumsat) = plt.subplots(1, figsize=(6,6))
    cols = ["cyan", "purple"]
    marks = [r"$\clubsuit$",r"$\spadesuit$"]
    linw = [3, 1]

    for ind, suite in enumerate(["SIMBA", "IllustrisTNG"]):

        galaxies = galaxies_in.copy()

        numsats, massvec, errminvec, errplusvec = [], [], [], []

        print([gal.name for gal in galaxies])

        mass, errplus, errmin = predict_halomass(galaxies, suite, verbose=False)
        numsats.append(len(galaxies)); massvec.append(mass); errminvec.append(errmin); errplusvec.append(errplus)

        for i in sorted(range(len(galaxies)),reverse=True):
            if len(galaxies)>=2:
                galaxies.pop(i)
                print([gal.name for gal in galaxies])

                mass, errplus, errmin = predict_halomass(galaxies, suite, verbose=False)
                numsats.append(len(galaxies)); massvec.append(mass); errminvec.append(errmin); errplusvec.append(errplus)

        axnumsat.errorbar(numsats, massvec, yerr=[errminvec, errplusvec], lw=linw[ind], color=cols[ind], marker=marks[ind], markersize=10, label=suite)

    axnumsat.legend()
    axnumsat.set_xlabel("Num. of galaxies")
    axnumsat.set_ylabel(r"$M_{\rm "+subscript+",200}\; [10^{12} M_\odot]$")
    axnumsat.xaxis.set_major_locator(MaxNLocator(integer=True))
    axnumsat.set_title(title)
    fignumsat.savefig("Plots/mass_as_numsats_"+title+".png", bbox_inches='tight', dpi=300)


# Check the robustness of the prediction when random rotations are considered
def halomass_with_rotations(galaxies, suite, numrots = 50):

    masses = []

    for i in range(numrots):
        mass, errplus, errmin = predict_halomass(galaxies, suite, verbose=False, random_rotation=True)
        masses.append(mass)
        #print(mass, errmin, errplus)

    masses = np.array(masses)
    print("Results: {:.2f} {:.2f} {:.3f}".format( masses.mean(), masses.std(), masses.std()/masses.mean()) )


#--- GALAXY DEFINITIONS ---#


# McConnachie12, Price-Whelan mail
velMW = [0., 0., 0.]
MW = galaxy("MW",[0., 0., 0.], velMW, 5.e10, velcen=velMW, use_cart=1)
#MW = galaxy("MW",[0., 0., 0.], [-1.8, -537.2, 293.2], 5.e10, use_cart=1)
LMC = galaxy("LMC",[-0.56968243, -41.26318337, -27.13271726], [-75.0292706, -225.7814354,  217.12662424], 2.7e9, velcen=velMW, use_cart=1)
SMC = galaxy("SMC",[15.78079071, -37.26661931, -43.30274336], [41.3607973, -165.05340375,  165.54205222], 3.1e8, velcen=velMW, use_cart=1)
SgrdSph = galaxy("Sgr dSph",[17.5, 2.5, -6.5], [237.9, -24.3, 209.0], 1.e8, velcen=velMW, use_cart=1)
"""LMC = galaxy("LMC",[280.5, -32.9, 51], [-75.0292706, -225.7814354,  217.12662424], 2.7e9)
SMC = galaxy("SMC",[302.8, -44.3, 64], [41.3607973, -165.05340375,  165.54205222], 3.1e8)
SgrdSph = galaxy("Sgr dSph",[5.6, -14.2, 26], [237.9, -24.3, 209.0], 1.e8)"""
MW_gals = [MW, LMC, SMC, SgrdSph]



# Stellar masses and positions from McConnachie12
velM31 = [34., -123., -19.]  # MW velocity
#M31 = galaxy("M31",[121.2, -21.6, 783], 34., 1.03e11)
#M31 = galaxy("M31",[121.2, -21.6, 783], [66.1, -76.3, 45.1], 1.03e11)    # velocity from 1205.6864
M31 = galaxy("M31",[121.2, -21.6, 783], velM31, 1.03e11, velcen=velM31)  # Velocity from 1805.04079, stellar mass from 1410.0017
#Tri = galaxy("Triangulum",[133.6, -31.3, 809], [43.1, 101.3, 138.8], 2.90e9, velcen=velM31)    # velocity from 1205.6864
Tri = galaxy("Triangulum",[133.6, -31.3, 809], [45., 91., 124.], 2.90e9, velcen=velM31)  # Velocity from 1805.04079
M110 = galaxy("M110",[120.7, -21.1, 824], 67., 3.30e8, velcen=velM31)
M32 = galaxy("M32",[121.2, -22.0, 805], 22., 3.20e8, velcen=velM31)
#M32 = galaxy("M32",[121.2, -22.0, 805], 499., 3.20e8)     # galactic frame, velocity wrt CMB from http://ned.ipac.caltech.edu/byname?objname=M+32&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
NGC147 = galaxy("NGC147",[-357.2, 608.8, -178.3], [11.9, 71.1, 174.6], 1.e8, use_cart=1, velcen=velM31) # 2008.06055, 1310.0814
#NGC185 = galaxy("NGC185",[-323.5, 529.0, -158.1], [47.0, 47.0, 71.6], 6.7e7, use_cart=1) # 2008.06055, 1310.0814
M31_gals = [M31, Tri, M110, M32, NGC147]
M31_noTri = [M31, M110, M32, NGC147]

"""
galaxies = MW_gals + M31_gals
print("Galaxy  \t x  \t y  \t z  \t vel  \t starmass")
for gal in galaxies:
    print(gal.name,"\t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f} \t {:.2e}".format(gal.x, gal.y, gal.z, gal.vel, gal.starmass))
"""

"""velsMW = np.array([gal.vel for gal in MW_gals[1:]])
velsM31 = np.array([gal.vel for gal in M31_gals[1:]])
print(velsMW.mean(), velsMW.std())
print(velsM31.mean(), velsM31.std())
exit()"""



#--- MAIN ---#

"""

for suite in ["SIMBA", "IllustrisTNG"]:
    print("\n"+suite+"\n")
    halomass_with_rotations(MW_gals, suite)
    print("\n")
    halomass_with_rotations(M31_gals, suite)
    print("\n")
    #halomass_with_rotations(M31_noTri, suite)
    #print("\n")

"""



mass_as_numsats(MW_gals, "MW", "Milky Way")
mass_as_numsats(M31_gals, "M31", "M31 (with M33)")
#mass_as_numsats(M31_noTri, "M31", "M31 (without M33)")


exit()

# Visualize Local Group
#visualize_galaxies(MW_gals, M31_gals)

for suite in ["SIMBA", "IllustrisTNG"]:

    # Milky Way halo
    predict_halomass(MW_gals, suite)

    # Andromeda halo
    predict_halomass(M31_gals, suite)

    # Andromeda halo (excluding the Triangulum galaxy)
    #predict_halomass(M31_noTri, suite)
