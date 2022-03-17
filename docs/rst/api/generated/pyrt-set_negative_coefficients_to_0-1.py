from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyrt

dust_dir = Path('/home/kyle/repos/pyRT_DISORT/anc/mars_dust')
phsfn = np.load(dust_dir / 'phase_function.npy')[:, 23, 0]
ang = np.load(dust_dir / 'scattering_angles.npy')

coeff = pyrt.decompose(phsfn, ang, 129)
trimmed_coeff = pyrt.set_negative_coefficients_to_0(coeff)

reconst_pf = pyrt.reconstruct(coeff, ang)
trim_reconst_pf = pyrt.reconstruct(trimmed_coeff, ang)

plt.semilogy(ang, phsfn, label='Original phase function')
plt.semilogy(ang, reconst_pf, label='Reconstructed phase function')
plt.semilogy(ang, trim_reconst_pf, label='Trimmed phase function')
plt.xlim(0, 180)
plt.legend()

plt.show()