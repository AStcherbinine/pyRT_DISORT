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
            message = 'scattering_angles must be between 0 and 180 degrees.'
            raise ValueError(message)
        if not np.ndim(array) == 1:
            message = 'The scattering angles must be 1-dimensional.'
            raise ValueError(message)


class _FiniteNumericArray(np.ndarray):
    def __new__(cls, array: ArrayLike):
        obj = cls._make_array(array).view(cls)
        cls._validate(obj)
        return obj

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


class _AsymmetryParameter(_FiniteNumericArray):
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.all((array >= -1) & (array <= 1)):
            message = 'The asymmetry parameter must be between -1 and 1.'
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


def _validate_moments(moments: int, scattering_angles: np.ndarray = None) \
        -> int:
    try:
        moments = int(moments)
    except TypeError as te:
        message = 'moments must be an int.'
        raise TypeError(message) from te
    except ValueError as ve:
        message = 'moments cannot be converted to an int.'
        raise ValueError(message) from ve
    if moments <= 0:
        message = 'moments must be positive.'
        raise ValueError(message)
    if scattering_angles is not None:
        samples = len(scattering_angles)
        if moments > samples:
            message = f'moments ({moments}) cannot be larger than the ' \
                      f'number of samples ({samples}).'
            raise ValueError(message)
    return moments


def _validate_scattering_angle_dimension(
        phase_function: np.ndarray, scattering_angles: np.ndarray) -> None:
    if phase_function.shape[0] != scattering_angles.shape[0]:
        message = f'Axis 0 of phase_function ({phase_function.shape[0]},) ' \
                  f'must have the same length as ' \
                  f'scattering_angles {scattering_angles.shape}.'
        raise ValueError(message)


# TODO: this one is the only one that makes assumptions
#  (linear interpolation, sa is a uniform grid). Consider deletion.
def resample(phase_function: ArrayLike,
             scattering_angles: ArrayLike,
             samples: int) \
        -> tuple[np.ndarray, np.ndarray]:
    """Resample a phase function to an input a number of points.

    Parameters
    ----------
    phase_function: ArrayLike
        N-dimensional array of phase functions. Axis 0 is assumed to be the
        scattering angle axis and must have the same shape as
        ``scattering_angles``.
    scattering_angles: ArrayLike
        1-dimensional array of the scattering angles [degrees] associated with
        axis 0 of ``phase_function``.
    samples: int
        The number of samples to regrid the phase function onto.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The regridded phase function and scattering angles.

    Examples
    --------
    Take a 3-dimensional phase function sampled every 1 degree and resample it
    every 0.5 degree.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> phsfn.shape
    (181, 24, 317)
    >>> phsfn, sa = pyrt.resample(phsfn, ang, 361)
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


def normalize(phase_function: ArrayLike,
              scattering_angles: ArrayLike) \
        -> np.ndarray:
    """Normalize a phase function such that :math:`\int p(\mu) d \mu = 2`,
    where :math:`p` is the phase function and :math:`\mu` is the cosine of
    the scattering angles.

    Parameters
    ----------
    phase_function: ArrayLike
        N-dimensional array of phase functions. Axis 0 is assumed to be the
        scattering angle axis and must have the same shape as
        ``scattering_angles``.
    scattering_angles: ArrayLike
        1-dimensional array of the scattering angles [degrees] associated with
        axis 0 of ``phase_function``.

    Returns
    -------
    np.ndarray
        N-dimensional array of normalized phase functions. This array has a
        shape of ``phase_function.shape``.

    Notes
    -----
    This algorithm uses Simpson's rule for the integration.

    Examples
    --------
    Normalize a 3-dimensional phase function. Note that this  example will not
    do much of anything because the Martian dust phase function included with
    pyRT_DISORT is already normalized.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> phsfn = pyrt.normalize(phsfn, ang)

    Verify the first element along the scattering angle axis integrated to 2.

    >>> mean_pf = (phsfn[:-1, 0, 0] + phsfn[1:, 0, 0]) / 2
    >>> angle_diff = np.abs(np.diff(np.cos(np.radians(ang))))
    >>> np.sum(mean_pf.T * angle_diff.T)
    2.0000527

    """
    pf = _PhaseFunctionND(phase_function)
    sa = _ScatteringAngles(scattering_angles)
    _validate_scattering_angle_dimension(pf, sa)
    norm = np.abs(simps(pf, np.cos(np.radians(sa)), axis=0))
    return np.array(2 * pf / norm)


def decompose(phase_function: ArrayLike,
              scattering_angles: ArrayLike,
              n_moments: int) -> np.ndarray:
    """Decompose a phase function into Legendre coefficients.

    .. warning::
       This is not vectorized and can only handle 1-dimensional arrays!

    Parameters
    ----------
    phase_function: ArrayLike
        1-dimensional array of phase functions. Must have the same shape as
        ``scattering_angles``.
    scattering_angles: ArrayLike
        1-dimensional array of scattering angles associated with
        ``phase_function``.
    n_moments: int
        The number of moments to decompose the phase function into. This
        value must be smaller than the number of points in the phase
        function.

    Returns
    -------
    np.ndarray
        1-dimensional array of Legendre coefficients of the decomposed phase
        function. This array has a shape of ``(moments,)``.

    Notes
    -----
    This method normalizes the phase function, as the moments only really
    make sense if that's the case.

    Examples
    --------
    Decompose a phase function into Legendre coefficients and get the first
    5 of these coefficients.

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
    _validate_scattering_angle_dimension(pf, sa)
    pf = normalize(pf, sa)
    sa = np.radians(sa)
    # Subtract 1 since I'm forcing c0 = 1 in the equation
    # P(x) = c0 + c1*L1(x) + ... for DISORT
    pf -= 1
    n_moments = _validate_moments(n_moments, sa)
    lpoly = _make_legendre_polynomials(sa, n_moments)
    normal_matrix = _make_normal_matrix(pf, lpoly)
    normal_vector = _make_normal_vector(pf, lpoly)
    cholesky = np.linalg.cholesky(normal_matrix)
    first_solution = np.linalg.solve(cholesky, normal_vector)
    second_solution = np.linalg.solve(cholesky.T, first_solution)
    coeff = np.concatenate((np.array([1]), second_solution))
    return coeff


def fit_asymmetry_parameter(phase_function: ArrayLike,
                            scattering_angles: ArrayLike) \
        -> np.ndarray:
    """Fit asymmetry parameters to an array of phase functions.

    .. warning::
       This function assumes a uniformly spaced scattering angles.

    Parameters
    ----------
    phase_function: ArrayLike
        N-dimensional array of phase functions. Axis 0 is assumed to be the
        scattering angle axis and must have the same shape has
        ``scattering_angles``.
    scattering_angles: ArrayLike
        1-dimensional array of scattering angles [degrees] associated with axis
        0 of ``phase_function``.

    Returns
    -------
    N-dimensional array of asymmetry parameters. This array will have a shape
    of ``phase_function.shape[1:]``.

    Examples
    --------
    Fit asymmetry parameters to an array of phase functions.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')
    >>> phsfn.shape
    (181, 24, 317)
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> g = pyrt.fit_asymmetry_parameter(phsfn, ang)
    >>> g.shape
    (24, 317)

    """
    pf = _PhaseFunctionND(phase_function)
    sa = _ScatteringAngles(scattering_angles)
    _validate_scattering_angle_dimension(pf, sa)
    mean_pf = (pf[1:] + pf[:-1]) / 2
    cos_sa = np.cos(np.radians(sa))
    median_angle_difference = np.median(np.abs(np.diff(cos_sa)))
    mean_sa = np.linspace(sa[0] + median_angle_difference, sa[-1] - median_angle_difference, num=len(sa)-1)
    expectation_pf = mean_pf.T * np.cos(np.radians(mean_sa))
    # Divide by 2 because g = 1/(4*pi) * integral but the azimuth angle
    # integral = 2*pi so the factor becomes 1/2
    return np.sum((expectation_pf * np.abs(np.diff(cos_sa))).T / 2, axis=0)


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


def construct_hg(asymmetry_parameter: ArrayLike,
                 scattering_angles: ArrayLike) \
        -> np.ndarray:
    r"""Construct a Henyey-Greenstein phase function from asymmetry parameters.

    Parameters
    ----------
    asymmetry_parameter: ArrayLike
        N-dimensional array of asymmetry paramters. All values must be between
        -1 and 1.
    scattering_angles: ArrayLike
        1-dimensional array of scattering angles [degrees].

    Returns
    -------
    np.ndarray
        N-dimensional arrray of phase functions. This array has a shape of
        ``scattering_angles.shape + asymmetry_parameter.shape``.

    Notes
    -----
    The Henyey-Greenstein phase function is defined as

    .. math::

       p(\theta) = \frac{1}{4\pi} \frac{1 - g^2}
                    {[1 + g^2 - 2g \cos(\theta)]^\frac{3}{2}}

    where :math:`p` is the phase function, :math:`\theta` is the scattering
    angle, and :math:`g` is the asymemtry parameter.

    .. warning::
       The normalization for the Henyey-Greenstein phase function is not the
       same as for a regular phase function. For this phase function,

       .. math::
          \int_{4\pi} p(\theta) = 1

       *not* 4 :math:`\pi`! To normalize it, either call
       :py:func:`~pyrt.normalize` or simply multiply the output by
       4 :math:`\pi`.

    Examples
    --------
    Construct phase functions having 181 scattering angles from an array of
    asymmetry parameters.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> g = np.load(dust_dir / 'asymmetry_parameter.npy')
    >>> g.shape
    (24, 317)
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> ang.shape
    (181,)
    >>> pyrt.construct_hg(g, ang).shape
    (181, 24, 317)

    """
    g = _AsymmetryParameter(asymmetry_parameter)
    sa = _ScatteringAngles(scattering_angles)
    cos_sa = np.cos(np.radians(sa))
    denominator = (1 + g**2 - 2 * np.multiply.outer(cos_sa, g))**(3/2)
    return 1 / (4 * np.pi) * (1 - g**2) / denominator


def decompose_hg(asymmetry_parameter: ArrayLike,
                 n_moments: int) \
        -> np.ndarray:
    r"""Decompose a Henyey-Greenstein phase function into Legendre
    coefficients.

    Parameters
    ----------
    asymmetry_parameter: ArrayLike
        N-dimensional array of asymmetry parameters. All values must be between
        -1 and 1.
    n_moments: int
        The number of moments to decompose the phase function into.

    Returns
    -------
    np.ndarray
        N-dimensional arrray of Legendre coefficients. This array has a shape
        of ``(n_moments,) + asymmetry_parameter.shape``.

    Notes
    -----
    The Henyey-Greenstein phase function can be decomposed as follows:

    .. math::
       p(\mu) = \sum_{n=0}^{\infty} (2n + 1)g^n P_n(\mu)

    where :math:`p` is the phase function, :math:`\mu` is the cosine of the
    scattering angle, :math:`n` is the moment number, :math:`g` is the
    asymmetry parameter, and :math:`P_n(\mu)` is the :math:`n`:sup:`th`
    Legendre polynomial.

    Examples
    --------
    Decompose an N-dimensional array of asymmetry parameters into 129 moments.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> g = np.load(dust_dir / 'asymmetry_parameter.npy')
    >>> g.shape
    (24, 317)
    >>> coeff = pyrt.decompose_hg(g, 129)
    >>> coeff.shape
    (129, 24, 317)

    Construct a Henyey-Greenstein phase function, decompose it, and see how
    this result compares to the analytic decomposition performed above.

    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> pf = pyrt.construct_hg(g[0, 0], ang)
    >>> lc = pyrt.decompose(pf, ang, 129)
    >>> np.amax(np.abs(lc - coeff[:, 0, 0]))
    3.6771765655231856e-07

    """
    g = _AsymmetryParameter(asymmetry_parameter)
    n_moments = _validate_moments(n_moments)
    moments = np.linspace(0, n_moments-1, num=n_moments)
    coeff = (2 * moments + 1) * np.power.outer(g, moments)
    return np.array(np.moveaxis(coeff, -1, 0))


def set_negative_coefficients_to_0(coefficients: ArrayLike) \
        -> np.ndarray:
    """Set an array of Legendre coefficients to 0 after the first coefficient
    is negative.

    Parameters
    ----------
    coefficients: ArrayLike
        N-dimensional array of Legendre coefficients. Axis 0 is assumed to be
        the phase function decomposition dimension.

    Returns
    -------
    np.ndarray
        N-dimensional array of the zeroed coefficients. This array has a shape
        of ``coefficients.shape``.

    Examples
    --------
    Decompose a 1-dimensional phase function.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')[:, 0, 0]
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> coeff = pyrt.decompose(phsfn, ang, 129)[:12]
    >>> coeff
    array([ 1.00000000e+00,  1.77784574e-01,  5.09440222e-01,  3.52030055e-02,
            1.62704765e-03,  6.26912942e-05,  8.40628501e-06, -6.12456095e-07,
           -4.97888637e-06, -1.45066047e-06,  8.79039649e-06, -1.34314968e-06])

    The eighth coefficient is negative. Set all coefficients after this value
    ---both positive and negative---to 0.

    >>> trimmed_coeff = pyrt.set_negative_coefficients_to_0(coeff)
    >>> trimmed_coeff
    array([1.00000000e+00, 1.77784574e-01, 5.09440222e-01, 3.52030055e-02,
           1.62704765e-03, 6.26912942e-05, 8.40628501e-06, 0.00000000e+00,
           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])

    This function will usually (always?) result in a worse fit, but it
    ensures you are not fitting physically meaningless aspects of the phase
    function.

    >>> reconst_pf = pyrt.reconstruct(pyrt.decompose(phsfn, ang, 129), ang)
    >>> np.sum((phsfn - reconst_pf)**2)
    1.5541640584143724e-08
    >>> reconst_pf = pyrt.reconstruct(trimmed_coeff, ang)
    >>> np.sum((phsfn - reconst_pf)**2)
    5.211496292356077e-08

    """
    coeff = np.copy(_FiniteNumericArray(coefficients))
    argmax = np.argmax(coeff < 0, axis=0)
    c = np.indices(coeff.shape)[0, ...]
    cond = c >= argmax
    coeff[cond] = 0
    return np.array(coeff)


def reconstruct(coefficients: ArrayLike,
                scattering_angles: ArrayLike) \
        -> np.ndarray:
    """Reconstruct a phase function from an array of Legendre coefficients.

    Parameters
    ----------
    coefficients: ArrayLike
        N-dimensional array of Legendre coefficients. Axis 0 is assumed to be
        the phase function dimension.
    scattering_angles: ArrayLike
        1-dimensional array of the scattering angles [degress] associated with
        axis 0 of ``coefficients``.

    Returns
    -------
    np.ndarray
        N-dimensional array of reconstructed phase functions. If
        ``coefficients`` is 1-dimensional this array will have a shape of
        ``scattering_angles.shape``, otherwise this array will have a shape of
        ``scattering_angles.shape + coefficients.shape[1:]``.

    Examples
    --------
    Deconstruct and re-construct a phase function.

    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pyrt
    >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
    >>> phsfn = np.load(dust_dir / 'phase_function.npy')[:, 0, 0]
    >>> ang = np.load(dust_dir / 'scattering_angles.npy')
    >>> coeff = pyrt.decompose(phsfn, ang, 129)
    >>> reconst_pf = pyrt.reconstruct(coeff, ang)
    >>> reconst_pf.shape
    (181,)

    Check how well the reconstructed phase function matches the original.

    >>> np.sum((phsfn - reconst_pf)**2)
    1.5541640584143724e-08

    """
    coeff = np.copy(_FiniteNumericArray(coefficients))
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

    print(np.where(g == np.amax(g)))
