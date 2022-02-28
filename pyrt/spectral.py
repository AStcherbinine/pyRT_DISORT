import numpy as np
from numpy.typing import ArrayLike


class _Wavelength(np.ndarray):
    """Designate that an input array represents wavelengths.

    Parameters
    ----------
    array
        Array of wavelengths. Must be between 0.1 and 50 microns.
    name
        Name of the wavelength.

    Raises
    ------
    ValueError
        Raised if any values in :code`array` are not between 0.1 and 50
        microns (I assume this is the valid range to do retrievals).

    """
    def __new__(cls, array: ArrayLike, name: str):
        obj = np.asarray(array).view(cls)
        obj.name = name
        cls.__raise_value_error_if_array_is_not_in_range(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    @staticmethod
    def __raise_value_error_if_array_is_not_in_range(obj) -> None:
        if not np.all((0.1 <= obj) & (obj <= 50)):
            message = f'All values in {obj.name} must be between 0.1 and 50 ' \
                      'microns.'
            raise ValueError(message)


def to_wavenumber(wavelength: ArrayLike) -> np.ndarray:
    """Convert wavelengths [microns] to wavenumber [1 / cm].

    Parameters
    ----------
    wavelength: ArrayLike
        Wavelengths.

    Returns
    -------
    np.ndarray
        The wavenumbers of each wavelength.

    Raises
    ------
    ValueError
        Raised if any values of the input wavelengths are not between 0.1 and
        50 (I assume this is the range of wavelengths where you can do
        radiative transfer).

    Examples
    --------
    Convert wavelengths to wavenumbers.

    >>> import pyrt
    >>> wavs = [1, 2, 3]
    >>> pyrt.to_wavenumber(wavs)
    array([10000.        ,  5000.        ,  3333.33333333])

    """
    wavelength = _Wavelength(wavelength, 'wavelength')
    return np.array(10 ** 4 / wavelength)
