#--------------------------------------------------------
# Define galaxies with kinematic and internal properties
# Author: Pablo Villanueva Domingo
# Last update: 5/11/21
#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, Galactic
#from astropy.coordinates import galactocentric_frame_defaults
#galactocentric_frame_defaults.set('pre-v4.0')


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


        # Peculiar velocity
        # If just a number is given, it is assumed as the velocity with respect the central halo, and write 3D velocity as a estimate
        if type(vel)=="float" or np.isscalar(vel):
            self.vel = vel
            self.vel3D = np.array([vel,vel,vel])/np.sqrt(3.)+np.array(velcen)
        else:
            self.vel = np.sqrt(np.sum((np.array(vel)-np.array(velcen))**2.))
            self.vel3D = np.array(vel)

        # Stellar mass (in Msun)
        self.starmass = starmass

# Visualization routine
def visualize_galaxies(MW_gals, M31_gals):

    fig = plt.figure(figsize=(4, 4))
    ax3d = fig.add_subplot(projection ="3d")
    #ax2d = fig.add_subplot()

    col_gal = ["blue", "red"]
    markers = ["d", "*"]

    for i, galaxies in enumerate([MW_gals, M31_gals]):

        pos = np.array([[gal.x, gal.y, gal.z] for gal in galaxies])

        ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=5, zorder=1000, marker=markers[i], color=col_gal[i])
        #ax2d.scatter(pos[:, 0], pos[:, 1], s=5, zorder=1000, marker=markers[i], color=col_gal[i])


    #plt.axis('off')
    fig.savefig("Plots/LocalGroup.png", bbox_inches='tight', dpi=300)




#--- GALAXY DEFINITIONS ---#

# SgrdSph and NGC147 are below the CAMELS threshold mass and thus should not be included, but we employ them for testing purposes

# Milky Way halo
# McConnachie12, see our paper for more details
velMW = [0., 0., 0.]    # MW velocity in galactocentric frame
MW = galaxy("MW",[0., 0., 0.], velMW, 5.e10, velcen=velMW, use_cart=1)
LMC = galaxy("LMC",[-0.56968243, -41.26318337, -27.13271726], [-75.0292706, -225.7814354,  217.12662424], 2.7e9, velcen=velMW, use_cart=1)
SMC = galaxy("SMC",[15.78079071, -37.26661931, -43.30274336], [41.3607973, -165.05340375,  165.54205222], 3.1e8, velcen=velMW, use_cart=1)
SgrdSph = galaxy("Sgr dSph",[17.5, 2.5, -6.5], [237.9, -24.3, 209.0], 1.e8, velcen=velMW, use_cart=1)
"""
# The definitions below use galactic coordinates for the positions rather than (cartesian) galactocentric coordinates
LMC = galaxy("LMC",[280.5, -32.9, 51], [-75.0292706, -225.7814354,  217.12662424], 2.7e9, velcen=velMW)
SMC = galaxy("SMC",[302.8, -44.3, 64], [41.3607973, -165.05340375,  165.54205222], 3.1e8, velcen=velMW)
SgrdSph = galaxy("Sgr dSph",[5.6, -14.2, 26], [237.9, -24.3, 209.0], 1.e8, velcen=velMW)
"""

MW_gals = [MW, LMC, SMC]
#MW_gals = [MW, LMC, SMC, SgrdSph]


# Andromeda halo
# Stellar masses and positions from McConnachie12
velM31 = [34., -123., -19.]  # M31 velocity from 1805.04079
#velM31 = [66.1, -76.3, 45.1] # M31 velocity from 1205.6864 (outdated)
M31 = galaxy("M31",[121.2, -21.6, 783], velM31, 1.03e11, velcen=velM31)  # Velocity from 1805.04079, stellar mass from 1410.0017
Tri = galaxy("Triangulum",[133.6, -31.3, 809], [45., 91., 124.], 2.90e9, velcen=velM31)  # Velocity from 1805.04079
#Tri = galaxy("Triangulum",[133.6, -31.3, 809], [43.1, 101.3, 138.8], 2.90e9, velcen=velM31)    # M33 velocity from 1205.6864
M110 = galaxy("M110",[120.7, -21.1, 824], np.sqrt(3.)*67., 3.30e8, velcen=velM31)   # Only radial component in M31 frame, get modulus by multiplying by sqrt(3) as estimate
M32 = galaxy("M32",[121.2, -22.0, 805], np.sqrt(3.)*22., 3.20e8, velcen=velM31)     # Only radial component in M31 frame, get modulus by multiplying by sqrt(3) as estimate
NGC147 = galaxy("NGC147",[-357.2, 608.8, -178.3], [11.9, 71.1, 174.6], 1.e8, velcen=velM31, use_cart=1) # 2008.06055, 1310.0814
#NGC185 = galaxy("NGC185",[-323.5, 529.0, -158.1], [47.0, 47.0, 71.6], 6.7e7, velcen=velM31, use_cart=1) # 2008.06055, 1310.0814

M31_gals = [M31, Tri, M110, M32]
#M31_gals = [M31, Tri, M110, M32, NGC147]
#M31_noTri = [M31, M110, M32, NGC147]


#--- MAIN ---#

if __name__=="__main__":

    galaxies = MW_gals + M31_gals
    print("Galaxy  \t x \t\t y \t\t z \t\t vel \t\t starmass")
    for gal in galaxies:
        print(gal.name,"\t\t {:.2f} \t\t {:.2f} \t\t {:.2f} \t\t {:.2f} \t\t {:.2e}".format(gal.x, gal.y, gal.z, gal.vel, gal.starmass))

    # Visualize Local Group
    visualize_galaxies(MW_gals, M31_gals)
