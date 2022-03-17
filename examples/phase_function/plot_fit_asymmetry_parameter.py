"""
Fit Asymmetry Parameter
=======================

Fit an asymmetry parameter to a phase function and plot the Henyey-Greenstein
phase function.

"""

# %%
# First import everything needed for this example.
import matplotlib.pyplot as plt
import numpy as np
import pyrt

# %%
# Let's grab a strongly forward scattering phase function and the angles over
# which it is defined.
dust_dir = '/home/kyle/repos/pyRT_DISORT/anc/mars_dust/'
phase_function = np.load(dust_dir + 'phase_function.npy')[:, 23, 0]
scattering_angles = np.load(dust_dir + 'scattering_angles.npy')

# %%
# Fit an asymmetry parameter to this phase function.
g = pyrt.fit_asymmetry_parameter(phase_function, scattering_angles)
print(g)

# %%
# We can construct a Henyey-Greenstein phase function from this asymmetry
# parameter. Let's do this, multiply by 4*pi since the
# Henyey-Greenstein phase function is normalized to 1, and see how well it
# matches the original phase function.
hg_pf = pyrt.construct_hg(g, scattering_angles) * 4 * np.pi

plt.rc('mathtext', fontset='stix')
plt.rc('font', **{'family': 'STIXGeneral'})
plt.rc('font', size=8)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=12)
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)
plt.rc('lines', linewidth=0.5)
plt.rc('axes', linewidth=0.5)
plt.rc('xtick.major', width=0.5)
plt.rc('xtick.minor', width=0.5)
plt.rc('ytick.major', width=0.5)
plt.rc('ytick.minor', width=0.5)
dpi = 150

fig, ax = plt.subplots()
ax.semilogy(scattering_angles, phase_function,
            label='Original phase function')
ax.semilogy(scattering_angles, hg_pf,
            label='Fit H-G phase function')
plt.legend()
ax.set_xlim(0, 180)
ax.set_ylim(10**-2, 10**4)
ax.set_xlabel('Scattering Angle [degrees]')
ax.set_ylabel('Phase Function')
ax.set_xticks(np.linspace(0, 180, num=180//30+1))
ax.set_xticks(np.linspace(0, 180, num=180//10+1), minor=True)
plt.show()
