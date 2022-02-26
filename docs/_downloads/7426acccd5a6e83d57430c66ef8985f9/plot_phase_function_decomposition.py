"""
Phase Function Decomposition
============================

Decompose a phase function.

"""

# %%
# Import everything that we'll need

import matplotlib.pyplot as plt
import numpy as np

# %%
# Let's grab a phase function and the angles over which it's defined. The phase
# function has shape (181, M, N) so I'll just pick the first particle size and
# wavelength.

phase_function = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')[:, 0, 0]
scattering_angles = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')

# %%
# For now, just plot the phase function

plt.semilogy(scattering_angles, phase_function)
plt.show()
