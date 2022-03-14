"""
Phase Function Decomposition
============================

Decompose a phase function into Legendre coefficients.

"""

# %%
# First import everything needed for this example.
import matplotlib.pyplot as plt
import numpy as np
import pyrt

# %%
# Let's grab a phase function and the angles over which it's defined. The phase
# function has shape (181, 24, 317), where it's defined over 181 scattering
# angles, 24 particle sizes, and 317 wavelengths. For this example, let's just
# pick the first one so we have an array to work with.
dust_dir = '/home/kyle/repos/pyRT_DISORT/anc/mars_dust/'
phase_function = np.load(dust_dir + 'phase_function.npy')[:, 0, 0]
scattering_angles = np.load(dust_dir + 'scattering_angles.npy')
print(scattering_angles)

# %%
# Let's resample the phase function to increase its resolution to 361 points.
phase_function, scattering_angles = \
        pyrt.resample_pf(phase_function, scattering_angles, 361)

# %%
# We can now decompose the phase function. This method automatically normalizes
# the phase function so we don't need to explicitly do that. Let's decompose it
# into 129 moments (128 moments in addition to the zeroth moment, which is
# always 1).
lc = pyrt.decompose(phase_function, scattering_angles, 129)
print(lc)

# %%
# At index 7 the coefficient is negative, and it appears the coefficients
# oscillate around 0 after this. Let's set these to 0.
lc = pyrt.set_negative_coefficients_to_0(lc)

# %%
# We can test how well the fit did by converting back into a phase function.
# Let's do that and see how it performed.
reconstructed_pf = pyrt.reconstruct_phase_function(lc, scattering_angles)

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
ax.plot(scattering_angles, phase_function,
        color='k',
        label='Original phase function')
ax.plot(scattering_angles, reconstructed_pf,
        color='r',
        label='Reconstructed phase function',
        linestyle='dotted')
plt.legend()
ax.set_xlim(0, 180)
ax.set_xlabel('Scattering Angle [degrees]')
ax.set_ylabel('Phase Function')
ax.set_xticks(np.linspace(0, 180, num=180//30+1))
ax.set_xticks(np.linspace(0, 180, num=180//10+1), minor=True)
plt.show()
