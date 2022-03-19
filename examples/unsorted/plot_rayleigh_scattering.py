"""
Rayleigh scattering
===================

Plot Rayleigh scattering optical depths

"""

# %%
# First import everything needed for this example.
import matplotlib.pyplot as plt
import numpy as np
import pyrt

# %%
# Rayleigh scattering depends on 2 factors: wavelength and column density.
# Let's create wavelengths to compute the column optical depth at.
wavelengths = np.linspace(0.1, 1, num=100)

# %%
# Now let's make the column density. I'll assume the pressure profile decreases
# exponentially with altitude (with scale height of 10 km) and a constant
# temperature profile. The column density function can compute column density
# from functions of pressure and temperature that are both dependent on
# altitude. The function's inputs go into the pressure and temperature args.
altitude = np.linspace(100, 0, num=15)
column_density = pyrt.column_density(
    pyrt.exponential_profile, pyrt.constant_profile, altitude, (610, 10), (150,))

# %%
# Now we can turn these into Rayleigh scattering optical depths.
od = pyrt.rayleigh_co2_optical_depth(column_density, pyrt.wavenumber(wavelengths))
print(od.shape)

# %%
# These are the Rayleigh scattering optical depths in each layer at each
# wavelength. We just need to sum over the columns to get the column integrated
# optical depths.

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
ax.semilogy(wavelengths * 1000, np.sum(od, axis=0))

ax.set_xlim(100, 1000)
ax.set_ylim(10**-4, 10)
ax.set_xlabel('Wavelength [nm]')
ax.set_ylabel('Column integrated optical depth')
ax.set_xticks(np.linspace(100, 1000, num=1000//100))
ax.set_xticks(np.linspace(100, 1000, num=1000//50-1), minor=True)
plt.show()
