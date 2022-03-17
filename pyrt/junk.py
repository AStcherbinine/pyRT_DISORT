from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(5, 1)
pos = np.arange(0, 169)

p = Path('/media/kyle/Samsung_T5/IUVS_data/orbit07200')
files = sorted(p.glob('*apoapse*7281*muv*.gz'))
#for i in files:
#    hdul = fits.open(i)
#    primary = hdul['primary'].data
#    print(primary.shape)

file = files[7]
hdul = fits.open(file)

primary = hdul['primary'].data
print(primary.shape)
for i in range(5):
    ax[i].plot(pos, np.sum(primary[:, i*30, :], axis=-1))
    ax[i].set_xlim(0, 168)
    if i != 4:
        ax[i].set_xticks([100])

ax[-1].set_xlabel('Integration number')
ax[2].set_ylabel('Total brightness (kR)')

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('/home/kyle/tcb.png')
