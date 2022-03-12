
"""
# Thought: PMOM is just some abstract number derived from the much more real phase function

Descriptors:
class _ParticleSize()
class _Wavelengths()
class _PositiveArray()
class _ScatteringAngles()

# Idk about this, the array may not exist and for HG it can be less than 0
class _ParticleSizeWavelengthGriddedArray:
    reff_grid = _ParticleSize()
    wav_grid = _Wavelengths()
    reff = _ParticleSize()
    wavs = _Wavelengths()
    array = _PositiveArray()
    def __init__(array, reff_grid, wav_grid, reff, wavs)
        self.array = array
        self.reff_grid = reff_grid
        self.wav_grid = wav_grid
        self.reff = reff
        self.wavs = wavs

    def nn_selector() -> np.ndarray

class _PhaseFunction(_ParticleSizeWavelengthGriddedArray)
    sa = _ScatteringAngles()
    def __init__(scattering_angles, reff_grid, wav_grid, reff, wavs)
        super().__init__(reff_grid, ...)
        self.sa = scattering_angles

    def _make_pf_col():
        raise NotImplementedError('not implemented yet')

    def decompose(nmoments: int)
        raise NotImplementedError('not implemented yet')

class Empirical(_PhaseFunction)
    def __init__(phase_function, scattering_angles, reff_grid, wav_grid, reff, wavs)
        self.pf = pf
        super().__init__(sa, reff_grid, wav_grid, reff, wavs)
        self.pf_col = _make_pf_col()

    def _make_pf_col() -> np.ndarray
        uses nn selector
    def resample() -> None
    def normalize() -> None
    def decompose() -> LegendreCoefficients

class HenyeyGreenstein(_PhaseFunction)
    def __init__(asymmetry_parameter, reff_grid, wavs_grid, reff, wavs, nsa=181, nmom: int)
        sa = np.linspace(0, 180, num=nsa)
        super().__init__(sa, reff_grid, wav_grid, reff, wavs)
        self.g = asymmetry_parameter
        self.pf_col = _make_pf_col

    def _make_pf_col() -> np.ndarray
        uses nn selector
    def construct(nscattering_angles: int) -> Empirical
    def decompose() -> LegendreCoefficients

class LegendreCoefficients()
    coeff = _Coefficients()
    sa = _ScatteringAngles()
    def __init__(coeff, scattering_angles)
        self.coeff = coeff
        self.sa = sa

    def set_negative_coeff_to_0() -> None
    def reconstruct() -> np.ndarray


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

def conrath() -> VerticalProfile
def uniform() -> VerticalProfile

def wavenumber(wavelengths) -> np.ndarray
def azimuth(ia, ea, pa) -> np.ndarray

"""
