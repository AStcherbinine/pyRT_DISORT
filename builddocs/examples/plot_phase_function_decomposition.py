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
phase_function = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')[:, 0, 0]
scattering_angles = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')
print(scattering_angles)

# %%
# Let's put these into a :class:`~pyrt.PhaseFunction` object. This object
# ensures the phase function and scattering angles look plausible and provides
# methods to manipulate these arrays.
pf = pyrt.PhaseFunction(phase_function, np.radians(scattering_angles))

# %%
# The scattering angles are defined each degree. Let's double the resolution
# of the arrays by resampling them.
pf.resample(362)
print(pf.phase_function.shape, pf.scattering_angles.shape)

# %%
# We can now decompose the phase function. This method normalizes the phase
# function and creates a :class:`~pyrt.LegendreCoefficients` object that acts
# just like a np.ndarray but with some methods. Let's decompose this phase
# function into 129 moments and look at the moments.
lc = pf.decompose(129)
print(lc)

# %%
# At index 7 the coefficient is negative, and it appears the coefficients
# oscillate around 0 after this. Let's set these to 0.
lc.set_negative_coefficients_to_0()

# %%
# This object can also convert back into a phase function. Let's do that and
# plot how the fit performed.
reconstructed_pf = lc.reconstruct_phase_function()

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
ax.plot(np.degrees(pf.scattering_angles), pf.phase_function,
        color='k',
        label='Original phase function')
ax.plot(np.degrees(reconstructed_pf.scattering_angles), reconstructed_pf.phase_function,
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
