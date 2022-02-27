from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

'''

class _PhaseFunction(_ScatteringArray)
   def normalize

class PhaseFunction
   def resample
   def normalize
   def decompose -> LC

   getter can return ndarray
   setter can set with _PhaseFunction

class LegendreCoefficients
   def zero()
   
'''


class _NSamples(int):
    """Designate that a number represents the number of samples.

    Parameters
    ----------
    value
        The number of samples to use.

    Raises
    ------
    TypeError
        Raised if the input cannot be converted into an int.
    ValueError
        Raised if the number of samples is not positive.

    """

    def __new__(cls, value: int, *args, **kwargs):
        return super().__new__(cls, value)

    def __init__(self, value: int, *args, **kwargs):
        self._raise_value_error_if_moments_is_not_positive()

    def _raise_value_error_if_moments_is_not_positive(self):
        if self <= 0:
            message = 'The number of samples must be positive.'
            raise ValueError(message)


class _NMoments(int):
    """Designate that a number represents the number of moments.

    Parameters
    ----------
    value
        The number of moments to use.

    Raises
    ------
    TypeError
        Raised if the input cannot be converted into an int.
    ValueError
        Raised if the number of moments is not positive.

    """

    def __new__(cls, value: int, *args, **kwargs):
        return super().__new__(cls, value)

    def __init__(self, value: int, *args, **kwargs):
        self._raise_value_error_if_moments_is_not_positive()

    def _raise_value_error_if_moments_is_not_positive(self):
        if self <= 0:
            message = 'The number of moments must be positive.'
            raise ValueError(message)


class _ScatteringArray(np.ndarray):
    def __new__(cls, array: np.ndarray, name: str = 'scattering_array'):
        obj = np.asarray(array).view(cls)
        obj.name = name
        cls._raise_value_error_if_array_is_not_positive_finite(obj)
        cls._raise_value_error_if_array_is_not_1d(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        self.name = getattr(obj, 'name', None)

    @staticmethod
    def _raise_value_error_if_array_is_not_positive_finite(obj) -> None:
        if ((obj < 0) | ~np.isfinite(obj)).any():
            message = f'{obj.name} must be non-negative and finite.'
            raise ValueError(message)

    @staticmethod
    def _raise_value_error_if_array_is_not_1d(obj) -> None:
        if obj.ndim != 1:
            message = f'{obj.name} must be 1-dimensional.'
            raise ValueError(message)

    def resample(self, samples: int):
        samples = _NSamples(samples)
        old = np.arange(len(self))
        new = np.linspace(0, old[-1], num=samples)
        return self.__new__(type(self), np.interp(new, old, self))


class _PhaseFunction(_ScatteringArray):
    """Designate an array as representing a phase function.

    Parameters
    ----------
    array
        Any phase function.

    Raises
    ------
    ValueError
        Raised if the input array is not 1-dimensional or if it contains values
        that are negative or not finite.

    """
    def __new__(cls, array: np.ndarray):
        obj = super().__new__(cls, array, 'phase_function')
        return obj


class _ScatteringAngles(_ScatteringArray):
    """Create an array that represents the angles where the phase function is
    defined.

    Parameters
    ----------
    array
        The scattering angles where the phase function is defined.

    Raises
    ------
    ValueError
        Raised if the input array is not 1-dimensional, if it contains values
        that are not between 0 and pi, or if it is not monotonically
        increasing.

    """
    def __new__(cls, array: np.ndarray):
        obj = super().__new__(cls, array, 'scattering_angles')
        cls._raise_value_error_if_array_is_not_between_0_and_pi(obj)
        cls._raise_value_error_if_array_is_not_monotonically_inc(obj)
        return obj

    @staticmethod
    def _raise_value_error_if_array_is_not_between_0_and_pi(obj):
        # Add some leeway for interpolation failures
        if ((obj < 0) | (obj > np.pi + 0.01)).any():
            message = f'{obj.name} must be between 0 and pi.'
            raise ValueError(message)

    @staticmethod
    def _raise_value_error_if_array_is_not_monotonically_inc(obj):
        if ~np.all(np.diff(obj) > 0):
            message = f'{obj.name} must be monotonically increasing.'
            raise ValueError(message)


# TODO: fit HG g method
class PhaseFunction:
    """Create an object representing a phase function and its scattering
    angles.

    Parameters
    ----------
    phase_function
        The phase function.
    scattering_angles
        The scattering angles of the phase function.

    Raises
    ------
    ValueError
        Raised if :code:`phase_function` is not 1-dimensional or if it contains
        values that are negative or not finite; if :code:`scattering_angles` is
        not 1-dimensional, if it contains values that are not between 0 and pi,
        or if it is not monotonically increasing; or if either
        :code:`phase_function` or :code:`angles` do not have the same shape.

    """
    def __init__(self, phase_function: ArrayLike,
                 scattering_angles: ArrayLike):
        self._phase_function = _PhaseFunction(phase_function)
        self.scattering_angles = _ScatteringAngles(scattering_angles)

        self._raise_value_error_if_inputs_are_not_same_shape()

    def _raise_value_error_if_inputs_are_not_same_shape(self):
        if self.phase_function.shape != self.scattering_angles.shape:
            message = f'The phase function and the scattering angles must ' \
                      f'have the same shape.'
            raise ValueError(message)

    @property
    def phase_function(self) -> np.ndarray:
        """Get the current state of the phase function

        Returns
        -------
        np.ndarray
            The phase function.

        """
        return np.array(self._phase_function)

    @phase_function.setter
    def phase_function(self, value):
        self._phase_function = _PhaseFunction(value)

    @property
    def scattering_angles(self) -> np.ndarray:
        """Get the current state of the scattering angles.

        Returns
        -------
        np.ndarray
            The scattering angles.

        """
        return np.array(self._scattering_angles)

    @scattering_angles.setter
    def scattering_angles(self, value):
        self._scattering_angles = _ScatteringAngles(value)

    def resample(self, samples: int) -> None:
        """Resample the phase function to a set number of samples.

        Parameters
        ----------
        samples: int
            The number of samples to resample the phase function (and angles).

        Returns
        -------
        None

        Notes
        -----
        Due to a bug with np.interp() there's a little jitter in the values
        at the endpoints at the 5th decimal point.

        Examples
        --------
        Resample the phase function.

        >>> import numpy as np
        >>> import pyrt
        >>> phsfn = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')[:, 0, 0]
        >>> ang = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')
        >>> f'The input phase function has shape: {phsfn.shape}.'
        'The input phase function has shape: (181,).'
        >>> pf = pyrt.PhaseFunction(phsfn, np.radians(ang))
        >>> pf.resample(360)
        >>> f'The resampled phase function has shape: {pf.phase_function.shape}.'
        'The resampled phase function has shape: (360,).'

        """
        samples = _NSamples(samples)
        self._phase_function = self._phase_function.resample(samples)
        self._scattering_angles = self._scattering_angles.resample(samples)

    def normalize(self) -> None:
        """Normalize the phase function.

        Returns
        -------
        None

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
        Normalize a phase function and verify it integrates to 2.

        >>> import numpy as np
        >>> import pyrt
        >>> phsfn = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')[:, 0, 0]
        >>> ang = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')
        >>> pf = pyrt.PhaseFunction(phsfn, np.radians(ang))
        >>> pf.normalize()
        >>> mean_phase_function = (pf.phase_function[:-1] + pf.phase_function[1:]) / 2
        >>> np.sum(mean_phase_function * np.abs(np.diff(np.cos(pf.scattering_angles))))
        2.0000522

        """
        norm = np.abs(integrate.simps(self._phase_function,
                                      np.cos(self._scattering_angles)))
        self._phase_function = 2 * self._phase_function / norm

    def decompose(self, moments: int) -> LegendreCoefficients:
        """Decompose the phase function into its Legendre coefficients.

        Parameters
        ----------
        moments: int
            The number of moments to decompose the phase function into. This
            value should be smaller than the number of points in the phase
            function, but there are no checks that this is satisfied.

        Returns
        -------
        LegendreCoefficients
            The decomposed Legendre coefficients.

        Notes
        -----
        This method normalizes the phase function, as the moments only really
        make sense if that's the case.

        Examples
        --------
        Decompose a phase function.

        >>> import numpy as np
        >>> import pyrt
        >>> phsfn = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')[:, 0, 0]
        >>> ang = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')
        >>> pf = pyrt.PhaseFunction(phsfn, np.radians(ang))
        >>> moments = pf.decompose(129)
        >>> print(moments[:5])
        [1.         0.17778457 0.50944022 0.03520301 0.00162705]

        """
        self.normalize()
        # Subtract 1 since I'm forcing c0 = 1 in the equation
        # P(x) = c0 + c1*L1(x) + ... for DISORT
        self._phase_function -= 1
        n_moments = _NMoments(moments)
        lpoly = self._make_legendre_polynomials(n_moments)
        normal_matrix = self._make_normal_matrix(lpoly)
        normal_vector = self._make_normal_vector(lpoly)
        cholesky = np.linalg.cholesky(normal_matrix)
        first_solution = np.linalg.solve(cholesky, normal_vector)
        second_solution = np.linalg.solve(cholesky.T, first_solution)
        coeff = np.concatenate((np.array([1]), second_solution))
        # Re-add 1 for the reason listed above
        self._phase_function += 1
        return LegendreCoefficients(coeff)

    def _make_legendre_polynomials(self, n_moments) -> np.ndarray:
        """Make an array of Legendre polynomials at the scattering angles.

        Notes
        -----
        This returns a 2D array. The 0th index is the i+1 polynomial and the
        1st index is the angle. So index [2, 6] will be the 3rd Legendre
        polynomial (L3) evaluated at the 6th angle

        """
        ones = np.ones((n_moments, self.scattering_angles.shape[0]))

        # This creates an MxN array with 1s on the diagonal and 0s elsewhere
        diag_mask = np.triu(ones) + np.tril(ones) - 1

        # Evaluate the polynomials at the input angles. I don't know why
        return np.polynomial.legendre.legval(
            np.cos(self.scattering_angles), diag_mask)[1:n_moments, :]

    def _make_normal_matrix(self, lpoly: np.ndarray) -> np.ndarray:
        return np.sum(
            lpoly[:, None, :] * lpoly[None, :, :] / self.phase_function ** 2,
            axis=-1)

    def _make_normal_vector(self, lpoly: np.ndarray) -> np.ndarray:
        return np.sum(lpoly / self.phase_function, axis=-1)


class LegendreCoefficients(np.ndarray):
    """Create an array of Legendre coefficients of the phase function.

    This class extends an ndarray and thus acts just like one, with additional
    methods shown below.

    Parameters
    ----------
    coefficients: np.ndarray
        The Legendre coefficients.

    Raises
    ------
    ValueError
        Raised if the coefficients are not 1-dimensional.

    """
    def __new__(cls, coefficients: np.ndarray):
        obj = np.asarray(coefficients).view(cls)
        cls._raise_value_error_if_array_is_not_1d(obj)
        return obj

    @staticmethod
    def _raise_value_error_if_array_is_not_1d(obj) -> None:
        if obj.ndim != 1:
            message = f'{obj.name} must be 1-dimensional.'
            raise ValueError(message)

    def set_negative_coefficients_to_0(self) -> None:
        """Set all coefficients to 0 starting from the first negative
        coefficient.

        Returns
        -------
        None

        Examples
        --------
        Set negative coefficients to 0.

        >>> import numpy as np
        >>> import pyrt
        >>> coeff = np.linspace(1, -2, num=5)
        >>> lc = pyrt.LegendreCoefficients(coeff)
        >>> f'The original coefficients are: {lc}'
        'The original coefficients are: [ 1.    0.25 -0.5  -1.25 -2.  ]'
        >>> lc.set_negative_coefficients_to_0()
        >>> f'The new coefficients are: {lc}'
        'The new coefficients are: [1.   0.25 0.   0.   0.  ]'

        """
        first = self._get_first_negative_coefficient_index()
        if first:
            self[first:] = 0

    def _get_first_negative_coefficient_index(self):
        return np.argmax(self < 0)

    def reconstruct_phase_function(self) -> PhaseFunction:
        """Reconstruct the phase function from the Legendre coefficients.

        Returns
        -------
        PhaseFunction
            The phase function.

        """
        angles = np.radians(np.arange(180))
        pf = np.polynomial.legendre.legval(np.cos(angles), self)
        return PhaseFunction(pf, angles)
