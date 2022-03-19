from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class _Angles(np.ndarray):
    """A base class for designating that an input represents angles.

    This object otherwise acts like an ndarray.

    Parameters
    ----------
    array
        Any array of angles.
    name
        The name of the angular array.
    low
        The lowest value any value in the array is allowed to be.
    high
        The highest value any value in the array is allowed to be.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are outside the input range.

    """

    def __new__(cls, array: ArrayLike, name: str, low: float, high: float):
        obj = cls._make_array(array, name).view(cls)
        obj.name = name
        obj.low = low
        obj.high = high
        cls._validate(obj)
        return obj

    @staticmethod
    def _make_array(value, name: str):
        try:
            array = np.asarray(value)
            array.astype(float)
        except TypeError as te:
            message = f'{name} must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = f'{name} must be numeric.'
            raise ValueError(message) from ve
        return array

    @staticmethod
    def _validate(array):
        if not np.all(((array.low <= array) & (array <= array.high))):
            message = f'All values in {array.name} must be between ' \
                      f'{array.low} and {array.high} degrees.'
            raise ValueError(message)


def azimuth(incidence: ArrayLike,
            emission: ArrayLike,
            phase: ArrayLike) \
        -> np.ndarray:
    r"""Construct azimuth angles from a set of incidence, emission, and phase
    angles.

    Parameters
    ----------
    incidence: ArrayLike
        N-dimensional array of incidence (solar zenith) angles [degrees]. All
        values must be between 0 and 90.
    emission: ArrayLike
        N-dimensional array of emission (emergence) angles [degrees]. All
        values must be between 0 and 90.
    phase: ArrayLike
        N-dimensional array of phase angles [degrees]. All values must be
        between 0 and 180.

    Returns
    -------
    np.ndarray
        N-dimensional array of azimuth angles.

    Raises
    ------
    TypeError
        Raised if any of the inputs cannot be cast into an ndarray.
    ValueError
        Raised if any of the values in the inputs are not within their
        mathematically valid range or if the inputs do not have compatible
        shapes.

    Notes
    -----
    While emission angles can mathematically be up to 180 degrees, I cannot
    envision a situation where this function would be useful for emission
    angles above 90 degrees---hence the cutoffs.

    Examples
    --------
    Create the azimuth angles from an array of angles, where (for simplicity)
    the incidence, emission, and phase angles are all equal.

    >>> import numpy as np
    >>> import pyrt
    >>> angles = np.array([[10, 20, 30], [15, 25, 35]])
    >>> pyrt.azimuth(angles, angles, angles)
    array([[119.74712028, 118.97673223, 117.65209561],
           [119.42828679, 118.38707342, 116.7625057 ]])

    """
    incidence = _Angles(incidence, 'incidence', 0, 90)
    emission = _Angles(emission, 'emission', 0, 90)
    phase = _Angles(phase, 'phase', 0, 180)
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp_arg = np.true_divide(
                np.cos(np.radians(phase)) - np.cos(np.radians(emission)) *
                np.cos(np.radians(incidence)),
                np.sin(np.radians(emission)) * np.sin(np.radians(incidence)))
            tmp_arg[~np.isfinite(tmp_arg)] = -1
            d_phi = np.arccos(np.clip(tmp_arg, -1, 1))
        return np.array(180 - np.degrees(d_phi))
    except ValueError as ve:
        message = f'The input arrays must have compatible shapes. They are' \
                  f'incidence: {incidence.shape}, ' \
                  f'emission: {emission.shape}, and ' \
                  f'phase: {phase.shape}.'
        raise ValueError(message) from ve
