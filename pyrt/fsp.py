import numpy as np
from pyrt.grid import regrid


def extinction_ratio_grid(extinction_cross_section, particle_size_grid, wavelength_grid, wavelength_reference: float) -> np.ndarray:
    """Make a grid of extinction cross section ratios.

    This is the extinction cross section at the input wavelengths divided by
    the extinction cross section at the reference wavelength.

    Parameters
    ----------
    extinction_cross_section
    particle_size_grid
    wavelength_grid
    wavelength_reference

    Returns
    -------

    """
    cext_slice = np.squeeze(regrid(extinction_cross_section, particle_size_grid, wavelength_grid, particle_size_grid, wavelength_reference))
    return (extinction_cross_section.T / cext_slice).T


def optical_depth(q_prof, column_density, extinction_ratio, column_integrated_od):
    normalization = np.sum(q_prof * column_density)
    profile = q_prof * column_density * column_integrated_od / normalization
    return (profile * extinction_ratio.T).T


if __name__ == '__main__':
    # TODO: consider making all of this into a function
    import numpy as np
    from pyrt import column_density, exponential_profile, constant_profile, conrath

    dust_dir = '/home/kyle/repos/pyRT_DISORT/anc/mars_dust/'
    cext = np.load(dust_dir + 'extinction_cross_section.npy')
    psize = np.load(dust_dir + 'particle_sizes.npy')
    wavs = np.load(dust_dir + 'wavelengths.npy')

    psizesicareabout = np.linspace(1.4, 1.6, num=14)
    mywavs = np.array([0.2, 0.4, 0.6, 0.8, 1, 9.3])

    eg = extinction_ratio_grid(cext, psize, wavs, [9.3])
    # I think I Need to select the psize I care about here
    eg = regrid(eg, psize, wavs, psizesicareabout, mywavs)
    z = np.linspace(100, 0, num=15)
    zmid = (z[1:] + z[:-1]) / 2
    q = conrath(zmid, 1, 10, 0.1)

    colden = column_density(exponential_profile, constant_profile, z, (610, 10), (150,))

    od = optical_depth(q, colden, eg, 1)
    print(np.sum(od, axis=0))
