from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simps
from scipy.interpolate import interp1d


class _ScatteringAngles(np.ndarray):
    def __new__(cls, array: ArrayLike):
        obj = cls._make_array(array).view(cls)
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return

    @staticmethod
    def _make_array(value):
        try:
            array = np.asarray(value)
            array.astype(float)
        except TypeError as te:
            message = 'The scattering angles must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The scattering angles must be numeric.'
            raise ValueError(message) from ve
        return array

    @staticmethod
    def _validate(array):
        if not np.all((0 <= array) & (array <= 180)):
            message = 'The scattering angles must be between 0 and 180 degrees.'
            raise ValueError(message)
        if not np.ndim(array) == 1:
            message = 'The scattering angles must be 1-dimensional.'
            raise ValueError(message)


class _FiniteNumericArray(np.ndarray):
    def __new__(cls, array: ArrayLike):
        obj = cls._make_array(array).view(cls)
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return

    @staticmethod
    def _make_array(value):
        try:
            array = np.asarray(value)
            array.astype(float)
        except TypeError as te:
            message = 'The array must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The array must be numeric.'
            raise ValueError(message) from ve
        return array

    @staticmethod
    def _validate(array):
        if not np.all(np.isfinite(array)):
            message = 'The array must be finite.'
            raise ValueError(message)


class _PhaseFunctionND(_FiniteNumericArray):
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.all(array >= 0):
            message = 'The phase function must be non-negative'
            raise ValueError(message)


class _PhaseFunction1D(_PhaseFunctionND):
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.ndim(array) == 1:
            message = 'The phase function must be 1-dimensional.'
            raise ValueError(message)


def _validate_samples(samples: int) -> int:
    try:
        return int(samples)
    except TypeError as te:
        message = 'samples must be an int.'
        raise TypeError(message) from te
    except ValueError as ve:
        message = 'samples cannot be converted to an int.'
        raise ValueError(message) from ve


def _validate_moments(scattering_angles, moments: int):
    samples = len(scattering_angles)
    try:
        moments = int(moments)
    except TypeError as te:
        message = 'moments must be an int.'
        raise TypeError(message) from te
    except ValueError as ve:
        message = 'moments cannot be converted to an int.'
        raise ValueError(message) from ve
    if moments > samples:
        message = 'moments cannot be larger than the number of samples.'
        raise ValueError(message)
    return moments


def _validate_scattering_angle_dimension(phase_function: np.ndarray, scattering_angles: np.ndarray) -> None:
    if phase_function.shape[0] != scattering_angles.shape[0]:
        message = f'Axis 0 of phase_function ({phase_function.shape[0]},) ' \
                  f'must have the same length as ' \
                  f'scattering_angles {scattering_angles.shape}.'
        raise ValueError(message)


def resample_pf(phase_function: ArrayLike,
                scattering_angles: ArrayLike,
                samples: int) \
        -> tuple[np.ndarray, np.ndarray]:
    """Resample a phase function to an input a number of points.

    Parameters
    ----------
    phase_function: ArrayLike
        The phase function array. Can be N-dimensional but axis 0 must be the
        same shape as :code:`scattering_angles`.
    scattering_angles: ArrayLike
        The scattering angles [degrees] of the phase function. Must be
        1-dimensional.
    samples: int
        The number of samples to regrid the phase function onto.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The regridded phase function and scattering angles.

    Examples
    --------
    Resample a 3D phase function to have 361 points (every 0.5 degrees).

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> phsfn.shape
    (181, 24, 317)
    >>> phsfn, sa = pyrt.resample_pf(phsfn, ang, 361)
    >>> phsfn.shape
    (361, 24, 317)
    >>> sa.shape
    (361,)

    """
    pf = _PhaseFunctionND(phase_function)
    sa = _ScatteringAngles(scattering_angles)
    samples = _validate_samples(samples)
    _validate_scattering_angle_dimension(pf, sa)
    f = interp1d(sa, pf, axis=0)
    angles = np.linspace(sa[0], sa[-1], num=samples)
    return f(angles), angles


def normalize_pf(phase_function: ArrayLike,
                 scattering_angles: ArrayLike) \
        -> np.ndarray:
    """Normalize an input phase function.

    Parameters
    ----------
    phase_function: ArrayLike
        The phase function array. Can be N-dimensional but axis 0 must be the
        same shape as :code:`scattering_angles`.
    scattering_angles: ArrayLike
        The scattering angles [degrees] of the phase function. Must be
        1-dimensional.

    Returns
    -------
    np.ndarray
        The normalized phase function along the scattering angle axis.

    Notes
    -----
    This algorithm uses Simpson's rule for the integration and normalizes
    the phase function such that

    .. math::
        \int p(\mu) d \mu = 2

    where :math:`p` is the phase function and :math:`\mu` is the cosine of
    the scattering angles.

    Examples
    --------
    Normalize a 3D phase function along the scattering angle axis and verify
    the first element integrated to 2.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> phsfn = pyrt.normalize_pf(phsfn, ang)
    >>> mean_pf = (phsfn[:-1, 0, 0] + phsfn[1:, 0, 0]) / 2
    >>> angle_diff = np.abs(np.diff(np.cos(np.radians(ang))))
    >>> np.sum(mean_pf.T * angle_diff.T)
    2.0000527

    """
    pf = _PhaseFunctionND(phase_function)
    sa = _ScatteringAngles(scattering_angles)
    norm = np.abs(simps(pf, np.cos(np.radians(sa)), axis=0))
    return np.array(2 * pf / norm)


def decompose(phase_function: ArrayLike,
              scattering_angles: ArrayLike,
              moments: int) -> np.ndarray:
    """Decompose the phase function into its Legendre coefficients.

    Parameters
    ----------
    phase_function: ArrayLike
        The phase function array. Must be 1-dimensional.
    scattering_angles: ArrayLike
        The scattering angle array. Must be 1-dimensional and the same shape
        as :code:`phase_function`.
    moments: int
        The number of moments to decompose the phase function into. This
        value must be smaller than the number of points in the phase
        function.

    Returns
    -------
    np.ndarray
        The phase function decomposed into its Legendre coefficients.

    Notes
    -----
    This method normalizes the phase function, as the moments only really
    make sense if that's the case.

    Examples
    --------
    Get the first 5 moments Legendre moments from a phase function's
    decomposition.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')[:, 0, 0]
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> pyrt.decompose(phsfn, ang, 129)[:5]
    array([1.        , 0.17778457, 0.50944022, 0.03520301, 0.00162705])

    """
    pf = _PhaseFunction1D(phase_function)
    sa = _ScatteringAngles(scattering_angles)
    pf = normalize_pf(pf, sa)
    sa = np.radians(sa)
    # Subtract 1 since I'm forcing c0 = 1 in the equation
    # P(x) = c0 + c1*L1(x) + ... for DISORT
    pf -= 1
    n_moments = _validate_moments(sa, moments)
    lpoly = _make_legendre_polynomials(sa, n_moments)
    normal_matrix = _make_normal_matrix(pf, lpoly)
    normal_vector = _make_normal_vector(pf, lpoly)
    cholesky = np.linalg.cholesky(normal_matrix)
    first_solution = np.linalg.solve(cholesky, normal_vector)
    second_solution = np.linalg.solve(cholesky.T, first_solution)
    coeff = np.concatenate((np.array([1]), second_solution))
    return coeff


def _make_legendre_polynomials(scattering_angles, n_moments) -> np.ndarray:
    """Make an array of Legendre polynomials at the scattering angles.

    Notes
    -----
    This returns a 2D array. The 0th index is the i+1 polynomial and the
    1st index is the angle. So index [2, 6] will be the 3rd Legendre
    polynomial (L3) evaluated at the 6th angle

    """
    ones = np.ones((n_moments, scattering_angles.shape[0]))

    # This creates an MxN array with 1s on the diagonal and 0s elsewhere
    diag_mask = np.triu(ones) + np.tril(ones) - 1

    # Evaluate the polynomials at the input angles. I don't know why
    return np.polynomial.legendre.legval(
        np.cos(scattering_angles), diag_mask)[1:n_moments, :]


def _make_normal_matrix(phase_function, lpoly: np.ndarray) -> np.ndarray:
    return np.sum(
        lpoly[:, None, :] * lpoly[None, :, :] / phase_function ** 2,
        axis=-1)


def _make_normal_vector(phase_function, lpoly: np.ndarray) -> np.ndarray:
    return np.sum(lpoly / phase_function, axis=-1)


def set_negative_coefficients_to_0(coefficients: ArrayLike) -> np.ndarray:
    """Set the Legendre coefficients to 0 after the first coefficient is
    negative.

    Parameters
    ----------
    coefficients: ArrayLike
        The Legendre coefficients. Can be N-dimensional but axis 0 is assumed
        to be the coefficient dimension.

    Returns
    -------
    np.ndarray
        Array of the zeroed coefficients.

    Examples
    --------
    Zero the coefficients of a 1-dimensional phase function.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')[:, 0, 0]
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> coeff = pyrt.decompose(phsfn, ang, 129)[:10]
    >>> coeff
    array([ 1.00000000e+00,  1.77784574e-01,  5.09440222e-01,  3.52030055e-02,
            1.62704765e-03,  6.26912942e-05,  8.40628501e-06, -6.12456095e-07,
           -4.97888637e-06, -1.45066047e-06])
    >>> pyrt.set_negative_coefficients_to_0(coeff)
    array([1.00000000e+00, 1.77784574e-01, 5.09440222e-01, 3.52030055e-02,
           1.62704765e-03, 6.26912942e-05, 8.40628501e-06, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00])

    """
    coeff = _FiniteNumericArray(coefficients)
    argmax = np.argmax(coeff < 0, axis=0)
    c = np.indices(coeff.shape)[0, ...]
    cond = c >= argmax
    coeff[cond] = 0
    return np.array(coeff)


def reconstruct_phase_function(coefficients: ArrayLike, scattering_angles: ArrayLike) -> np.ndarray:
    """Reconstruct a phase function from an array of Legendre coefficients.

    Parameters
    ----------
    coefficients: ArrayLike
        Array of Legendre coefficients. Can be N-dimensional but axis 0 is
        assumed to be the phase function / Legendre coefficient dimension.
    scattering_angles: ArrayLike
        The scattering angles. Must be 1-dimensional.

    Returns
    -------
    np.ndarray
        Phase function(s) reconstructed from the input coefficients.

    Examples
    --------
    Deconstruct and re-construct a phase function.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')[:, 0, 0]
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> coeff = pyrt.decompose(phsfn, ang, 129)[:10]
    >>> reconst_pf = pyrt.reconstruct_phase_function(coeff, ang)
    >>> reconst_pf.shape
    (181,)

    """
    coeff = _FiniteNumericArray(coefficients)
    sa = _ScatteringAngles(scattering_angles)
    pfs = np.moveaxis(np.polynomial.legendre.legval(np.cos(np.radians(sa)), coeff), -1, 0)
    return np.array(pfs)


if __name__ == '__main__':
    g = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/asymmetry_parameter.npy')
    #cext = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/extinction_cross_section.npy')
    #csca = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_cross_section.npy')
    reff = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/particle_sizes.npy')
    wavs = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/wavelengths.npy')
    #pmom = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/legendre_coefficients.npy')
    phsfn = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')
    #phsfnrexp = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function_reexpanded.npy')
    sa = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')

    d = decompose(phsfn[:, 0, 0], sa, 129)
    e = reconstruct_phase_function(d, sa)
    print(type(e), e.shape)
