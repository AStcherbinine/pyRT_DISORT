"""

column.py
class Column



Examples
- plot_phase_function_decomposition.py
- plot_fit_hg_phase_function.py
- plot_rayleigh.py


class ForwardScatteringProperties:
    def __init__(scattering_cross_section, extinction_cross_section, reff_grid, wav_grid)
        self.scs
        self.ecs
        self.reff_grid
        self.wav_grid

    def make_extinction(wave_ref):


class Aerosol:
    def __init__(fsp: FSP, phase_function: _PhaseFunction, name: str = None)
        self.fsp = fsp
        self.pf = phase_function
        self.name = name
    # warn if reff and/or wavs in fsp and pf are not the same


class Column:
    od = OpticalDepth()
    ssa = SingleScatteringAlbedo()
    pmom = PhaseFunctionDecomposition()
    def __init__(od, ssa, pmom, name=None):
        self.od = od
        self.ssa = ssa
        self.pmom = pmom
    # These can get gotten + set but this obj. checks for array shapes

    def __add__():
        # define methods for adding od, ssa, pmom

def rayleigh_ssa(nlyr, nwavs) -> np.ndarray(nlyr, nwavs)
def rayleigh_pmom(nlyr, nwavs) -> np.ndarray(nlyr, nwavs)
def rayleigh_co2(colden, wavs) -> Column


class AerosolColumn(Column):
    def __init__(aerosol: Aerosol, q_profile, colden, integrated od, wave_ref):
        self.aerosol = aerosol
        self.q_prof = q_profile
        self. ... = ....
        od = self.make_od()
        ssa =
        pmom =

        super().__init__(od, ssa, pmom)

    def remake_properties():
        self.od = self.make_od()
        self.ssa = self.make_ssa()
        self.pmom = self.make_pmom()

    @property
    def q_prof():
        return self.q_prof

    @remake_properties
    @q_prof.setter
    def q_prof(self, val):
        self.q_prof = val


class VerticalProfile:
    def __init__(profile: np.ndarray, altitudes: np.ndarray):
        self.profile = profile
        self.altitudes = altitudes


consider an Atmosphere class that holds eos variables, and then requiring Atmosphere instead of colden


~~~~~~~~~~~~~~Structure~~~~~~~~~~~~~~~~~~~~~~
grid.py
class _ParticleSizeWavelengthGriddedArray
2 descriptors

phase_function.py
class PhaseFunction
class HenyeyGreenstein
class LegendreCoefficients
many descriptors

fsp.py
class ForwardScatteringProperties

column.py
class Column
class AersolColumn

aerosol.py
class Aerosol

rayleigh.py
def rayleigh_ssa()
def rayleigh_pmom()
def rayleigh_co2()

vertical_profile.py
class VerticalProfile
def conrath()
def uniform()


"""
import numpy as np
from astropy.io import fits


if __name__ == '__main__':
    hdul = fits.open('/home/kyle/Downloads/mars045i_all_v01.fits')
    g = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/asymmetry_parameter.npy')
    #cext = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/extinction_cross_section.npy')
    #csca = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_cross_section.npy')
    reff = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/particle_sizes.npy')
    wavs = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/wavelengths.npy')
    #pmom = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/legendre_coefficients.npy')
    #phsfn = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')
    #phsfnrexp = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function_reexpanded.npy')
    sa = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')

