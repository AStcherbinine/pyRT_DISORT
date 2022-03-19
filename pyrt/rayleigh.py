"""The ``rayleigh`` module contains structures for computing Rayleigh
scattering.
"""
import numpy as np
from pyrt.spectral import wavenumber


def rayleigh_single_scattering_albedo(n_layers: int, wavelengths: int) -> np.ndarray:
    """Make a generic Rayleigh single scattering albedo.

    Parameters
    ----------
    n_layers
    wavelengths

    Returns
    -------

    """
    return np.ones((n_layers, wavelengths))


def rayleigh_legendre(n_layers: int, n_wavelengths: int) -> np.ndarray:
    """Make the generic Rayleigh Legendre decomposition.

    Parameters
    ----------
    n_layers
    n_wavelengths

    Returns
    -------

    """
    pf = np.zeros((3, n_layers, n_wavelengths))
    pf[0, :] = 1
    pf[2, :] = 0.5
    return pf


def rayleigh_co2_optical_depth(column_density: np.ndarray, wavenumber: np.ndarray) -> np.ndarray:
    """Compute the Rayleigh CO2 scattering optical depth.

    Parameters
    ----------
    column_density
    wavenumber

    Returns
    -------

    """
    colden = column_density[:, None]
    mol_cs = _molecular_cross_section(wavenumber)[:, None]
    scattering_od = np.multiply(colden[:, None, :], mol_cs[None, :])
    return np.squeeze(scattering_od)


def _molecular_cross_section(wavenumber):
    number_density = 25.47 * 10 ** 18  # laboratory molecules / cm**3
    king_factor = 1.1364 + 25.3 * 10 ** -12 * wavenumber ** 2
    index_of_refraction = _co2_index_of_refraction(wavenumber)
    return _co2_cross_section(
        number_density, wavenumber, king_factor, index_of_refraction) * 10 ** -4


def _co2_index_of_refraction(wavenumber) -> np.ndarray:
    n = 1 + 1.1427 * 10 ** 3 * (
                5799.25 / (128908.9 ** 2 - wavenumber ** 2) +
                120.05 / (89223.8 ** 2 - wavenumber ** 2) +
                5.3334 / (75037.5 ** 2 - wavenumber ** 2) +
                4.3244 / (67837.7 ** 2 - wavenumber ** 2) +
                0.00001218145 / (2418.136 ** 2 - wavenumber ** 2))
    return n


def _co2_cross_section(number_density: float, wavenumber, king_factor: np.ndarray,
                    index_of_refraction: np.ndarray) -> np.ndarray:
    coefficient = 24 * np.pi**3 * wavenumber**4 / number_density**2
    middle_term = ((index_of_refraction ** 2 - 1) /
                   (index_of_refraction ** 2 + 2)) ** 2
    return coefficient * middle_term * king_factor   # cm**2 / molecule


if __name__ == '__main__':
    from scipy.constants import Boltzmann
    from scipy.integrate import quadrature as quad
    from pyrt import exponential_profile, constant_profile, column_density, linear_profile


    def hydrostatic_profile(altitude: np.ndarray, surface: float, scale_height: np.ndarray):
        return surface * np.exp(-altitude / scale_height)


    t = np.linspace(150, 200, num=15)
    z = np.linspace(100, 0, num=15)
    p = hydrostatic_profile(z, 670, 10)
    n = p / t / Boltzmann
    foo = np.array([quad(hydrostatic_profile, z[i+1], z[i], args=(n[-1], 10))[0] for i in range(len(z)-1)])
    for f in foo:
        print(f)

    w = np.linspace(0.2, 1, num=5)

    r = RayleighCO2(w, foo*1000)
    print(np.sum(r.optical_depth, axis=0))

    cd = column_density(exponential_profile, linear_profile, z, (610, 10), (150, 200))
    r = RayleighCO2(w, cd)
    print(np.sum(r.optical_depth, axis=0))

    bar = rayleigh_co2_optical_depth(cd, wavenumber(w))
    print(np.sum(bar, axis=0))
