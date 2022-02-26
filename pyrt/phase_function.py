"""The legendre module provides functions for making Legendre decompositions."""
import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate


# TODO: n_samples should be at least n_moments


# TODO: this
class _LegendreDecomposer:
    """ A collection of methods for decomposing a phase function into
    polynomials.

    This class can decompose a phase function into Legendre polynomials. In
    principle it can decompose any function into those polynomials but it's set
    up specifically for that task.

    Parameters
    ----------
    phase_function
    angles
    n_moments
    n_samples

    Raises
    ------
    TypeError
        Raised if any of the inputs cannot be cast to the correct shape.
    ValueError
        Raised in any of the inputs are unphysical.

    """

    def __init__(self, phase_function: np.ndarray, angles: np.ndarray,
                 n_moments: int, n_samples: int):
        # self.bundle = _PhaseFunctionBundle(phase_function, angles)
        self.n_moments = _NMoments(n_moments)
        self.n_samples = _NSamples(n_samples)

        # Fit P(x) = c0 + c1*L1(x) + ... where I force c0 = 1 for DISORT
        self.resamp_norm_pf = \
            self.bundle.normalize_resampled_phase_function(self.n_samples) - 1
        self.lpoly = self._make_legendre_polynomials()

    def _make_legendre_polynomials(self) -> np.ndarray:
        """Make an array of Legendre polynomials at the input angles.

        Notes
        -----
        This returns a 2D array. The 0th index is the i+1 polynomial and the
        1st index is the angle. So index [2, 6] will be the 3rd Legendre
        polynomial (L3) evaluated at the 6th angle

        """
        resampled_theta = self.bundle.resample_angles(self.n_samples)
        ones = np.ones((self.n_moments, self.n_samples))

        # This creates an MxN array with 1s on the diagonal and 0s elsewhere
        diag_mask = np.triu(ones) + np.tril(ones) - 1

        # Evaluate the polynomials at the input angles. I don't know why
        return np.polynomial.legendre.legval(
            np.cos(resampled_theta), diag_mask)[1:self.n_moments, :]

    def decompose(self) -> np.ndarray:
        """Decompose the phase function into its Legendre moments.

        """
        normal_matrix = self._make_normal_matrix()
        normal_vector = self.__make_normal_vector()
        cholesky = np.linalg.cholesky(normal_matrix)
        first_solution = np.linalg.solve(cholesky, normal_vector)
        second_solution = np.linalg.solve(cholesky.T, first_solution)
        coeff = np.concatenate((np.array([1]), second_solution))
        self._warn_if_negative_coefficients(coeff)
        return coeff

    def _make_normal_matrix(self) -> np.ndarray:
        return np.sum(self.lpoly[:, None, :] * self.lpoly[None, :, :] /
                      self.resamp_norm_pf ** 2, axis=-1)

    def __make_normal_vector(self) -> np.ndarray:
        return np.sum(self.lpoly / self.resamp_norm_pf, axis=-1)

    def _warn_if_negative_coefficients(self, coeff):
        first_negative_index = self._get_first_negative_coefficient_index(
            coeff)
        if first_negative_index:
            message = f'Coefficient {first_negative_index} is negative.'

    @staticmethod
    def _get_first_negative_coefficient_index(coeff: np.ndarray) -> np.ndarray:
        return np.argmax(coeff < 0)

    def filter_negative_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        """Set all coefficients at the first negative one to 0.

        Parameters
        ----------
        coeff
            The Legendre coefficients.

        """
        first_negative_index = self._get_first_negative_coefficient_index(
            coeff)
        if first_negative_index:
            coeff[first_negative_index:] = 0
        return coeff


class _PhaseFunction(np.ndarray):
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
        return np.asarray(array).view(cls)

    def __array_finalize__(self, obj: np.ndarray):
        self._raise_value_error_if_array_is_not_positive_finite(obj)
        self._raise_value_error_if_array_is_not_1d(obj)

    @staticmethod
    def _raise_value_error_if_array_is_not_positive_finite(obj: np.ndarray):
        if ((obj < 0) | ~np.isfinite(obj)).any():
            message = 'The phase function must be non-negative and finite.'
            raise ValueError(message)

    @staticmethod
    def _raise_value_error_if_array_is_not_1d(obj: np.ndarray):
        if obj.ndim != 1:
            message = 'The phase function must be 1-dimensional.'
            raise ValueError(message)


class _ScatteringAngles(np.ndarray):
    """Designate an array as representing the angles where the phase function is
    defined.

    Parameters
    ----------
    array
        The scattering angles where the phase function is defined.

    Raises
    ------
    ValueError
        Raised if the input array is not 1-dimensional, if it contains values
        that are not between 0 and 180, or if it is not monotonically
        increasing.

    """

    def __new__(cls, array: np.ndarray):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        # self._raise_value_error_if_array_is_not_between_0_and_180(obj)
        self._raise_value_error_if_array_is_not_1d(obj)
        self._raise_value_error_if_array_is_not_monotonically_inc(obj)

    @staticmethod
    def _raise_value_error_if_array_is_not_between_0_and_180(obj: np.ndarray):
        if ((obj < 0) | (obj > 180)).any():
            message = 'The scattering angles must be between 0 and 180.'
            raise ValueError(message)

    @staticmethod
    def _raise_value_error_if_array_is_not_1d(obj: np.ndarray):
        if obj.ndim != 1:
            message = 'The scattering angles must be 1-dimensional.'
            raise ValueError(message)

    @staticmethod
    def _raise_value_error_if_array_is_not_monotonically_inc(obj: np.ndarray):
        if ~np.all(np.diff(obj) > 0):
            message = 'The scattering angles must be monotonically increasing.'
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


# TODO: fit HG g method
class PhaseFunction:
    """Create a phase function.

    Parameters
    ----------
    phase_function
        The phase function.
    scattering_angles
        The scattering angles of the phase function.

    Raises
    ------
    ValueError
        Raised if either :code:`phase_function` or :code:`angles` do not have
        the same shape.

    """

    def __init__(self, phase_function: ArrayLike,
                 scattering_angles: ArrayLike):
        self.phase_function = _PhaseFunction(phase_function)
        self.scattering_angles = _ScatteringAngles(scattering_angles)

        self._raise_value_error_if_inputs_are_not_same_shape()

    def _raise_value_error_if_inputs_are_not_same_shape(self):
        if self.phase_function.shape != self.scattering_angles.shape:
            message = 'The phase function and the scattering angles must have' \
                      'the same shape.'
            raise ValueError(message)

    @property
    def phase_function(self):
        return self._phase_function

    @phase_function.setter
    def phase_function(self, value):
        self._phase_function = np.array(value)

    @property
    def scattering_angles(self):
        return self._scattering_angles

    @scattering_angles.setter
    def scattering_angles(self, value):
        self._scattering_angles = np.array(value)

    def resample(self, samples: int) -> None:
        samples = _NSamples(samples)
        resampled_scattering_angles = np.linspace(
            self.scattering_angles[0], self.scattering_angles[-1], num=samples)
        self.phase_function = np.interp(
            resampled_scattering_angles, self.scattering_angles,
            self.phase_function)
        self.scattering_angles = resampled_scattering_angles

    def normalize(self) -> None:
        norm = np.abs(integrate.simps(self.phase_function,
                                      np.cos(self.scattering_angles)))
        self.phase_function = 2 * self.phase_function / norm

    def decompose(self, moments: int) -> np.ndarray:
        self.normalize()
        self.phase_function -= 1
        n_moments = _NMoments(moments)
        lpoly = self._make_legendre_polynomials(n_moments)
        normal_matrix = self._make_normal_matrix(lpoly)
        normal_vector = self._make_normal_vector(lpoly)
        cholesky = np.linalg.cholesky(normal_matrix)
        first_solution = np.linalg.solve(cholesky, normal_vector)
        second_solution = np.linalg.solve(cholesky.T, first_solution)
        coeff = np.concatenate((np.array([1]), second_solution))
        self.phase_function += 1
        return coeff

    def _make_legendre_polynomials(self, n_moments) -> np.ndarray:
        """Make an array of Legendre polynomials at the input angles.

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

    def _make_normal_matrix(self, lpoly) -> np.ndarray:
        return np.sum(
            lpoly[:, None, :] * lpoly[None, :, :] / self.phase_function ** 2,
            axis=-1)

    def _make_normal_vector(self, lpoly) -> np.ndarray:
        return np.sum(lpoly / self.phase_function, axis=-1)


class LegendreCoefficients(np.ndarray):
    def __new__(cls, array: np.ndarray):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        pass

    def set_negative_coefficients_to_0(self):
        first = self._get_first_negative_coefficient_index()
        if first:
            self[first:] = 0

    def _get_first_negative_coefficient_index(self):
        return np.argmax(self < 0)


if __name__ == '__main__':
    pmom = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/legendre_coefficients.npy')[:, 0, 0]
    phsfn = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function.npy')[:, 0, 0]
    exp = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/phase_function_reexpanded.npy')[:, 0, 0]
    ang = np.load('/home/kyle/repos/pyRT_DISORT/anc/mars_dust/scattering_angles.npy')

    pf = PhaseFunction(phsfn, np.radians(ang))
    pf.resample(360)
    pm = pf.decompose(129)

    lc = LegendreCoefficients(pm)
    lc.set_negative_coefficients_to_0()
    print(lc)
