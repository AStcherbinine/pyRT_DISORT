from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class _Wavelengths(np.ndarray):
    """A base class for designating that an input represents wavelengths.

    This object otherwise acts like an ndarray.

    Parameters
    ----------
    array: ArrayLike
        Any array of wavelengths.
    name: str
        The name of the array.
    low: float
        The lowest value any value in the array is allowed to be.
    high: float
        The highest value any value in the array is allowed to be.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are outside the input range.

    """

    def __new__(cls, array: ArrayLike, name: str, low: float = 0.1,
                high: float = 50):
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
        if not np.all((array.low <= array) & (array <= array.high)):
            message = f'All values in {array.name} must be between ' \
                      f'{array.low} and {array.high} microns.'
            raise ValueError(message)


def wavenumber(wavelength: ArrayLike) -> np.ndarray:
    r"""Convert wavelengths [microns] to wavenumber
    [:math:`\frac{1}{\text{cm}}`].

    Parameters
    ----------
    wavelength: ArrayLike
        N-dimensional array of wavelengths. All values must be between 0.1
        and 50 microns (I assume this is the valid range to do radiative
        transfer).

    Returns
    -------
    np.ndarray
        N-dimensional array of wavenumbers with shape ``wavelength.shape``.

    Raises
    ------
    TypeError
        Raised if the input cannot be cast into an ndarray.
    ValueError
        Raised if the input contains values outside the valid range.

    Examples
    --------
    Convert wavelengths to wavenumbers.

    >>> import numpy as np
    >>> import pyrt
    >>> wavs = [1, 2, 3]
    >>> pyrt.wavenumber(wavs)
    array([10000.        ,  5000.        ,  3333.33333333])

    This function can handle arrays of any shape.

    >>> wavs = np.ones((10, 20, 30))
    >>> pyrt.wavenumber(wavs).shape
    (10, 20, 30)

    """
    wavelength = _Wavelengths(wavelength, 'wavelength')
    return np.array(10 ** 4 / wavelength)
