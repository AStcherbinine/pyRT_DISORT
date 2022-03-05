from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class _Angles(np.ndarray):
    """An abstract base class for designating that an input represents angles.

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
        obj = np.asarray(array).view(cls)
        obj.name = name
        obj.low = low
        obj.high = high
        obj = cls.__add_dimension_if_array_is_shapeless(obj)
        cls.__raise_value_error_if_array_is_not_in_input_range(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.low = getattr(obj, 'low', None)
        self.high = getattr(obj, 'high', None)

    @staticmethod
    def __add_dimension_if_array_is_shapeless(obj):
        if obj.shape == ():
            obj = obj[None]
        return obj

    @staticmethod
    def __raise_value_error_if_array_is_not_in_input_range(obj) -> None:
        if not np.all(((obj.low <= obj) & (obj <= obj.high))):
            message = f'All values in {obj.name} must be between ' \
                      f'{obj.low} and {obj.high} degrees.'
            raise ValueError(message)


class IncidenceAngles(_Angles):
    """Designate that an array represents incidence (solar zenith) angles.

    This object otherwise acts like an ndarray.

    Parameters
    ----------
    array
        Any array of incidence angles [degrees]. Must be between 0 and 90.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 90.

    Examples
    --------
    Designate an array-like object as representing incidence angles.

    >>> import numpy as np
    >>> import pyrt
    >>> pyrt.IncidenceAngles([0, 10, 20])
    IncidenceAngles([ 0, 10, 20])

    This array can have any shape.

    >>> inc = pyrt.IncidenceAngles(np.ones((2, 3)) * np.array([30, 60, 90]))
    >>> inc.shape
    (2, 3)

    Compute MU0 for a slice of this array.

    >>> np.cos(np.radians(inc[0]))
    IncidenceAngles([8.66025404e-01, 5.00000000e-01, 6.12323400e-17])

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'IncidenceAngles', 0, 90)
        return obj


class EmissionAngles(_Angles):
    """Designate that an array represents emission (emergence) angles.

    This object otherwise acts like an ndarray.

    Parameters
    ----------
    array
        Any array of emission angles [degrees]. Must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 180.

    Examples
    --------
    Designate an array-like object as representing emission angles.

    >>> import numpy as np
    >>> import pyrt
    >>> pyrt.EmissionAngles([0, 10, 20])
    EmissionAngles([ 0, 10, 20])

    This array can have any shape.

    >>> inc = pyrt.EmissionAngles(np.ones((2, 3)) * np.array([30, 60, 90]))
    >>> inc.shape
    (2, 3)

    Compute MU for a slice of this array.

    >>> np.cos(np.radians(inc[0]))
    EmissionAngles([8.66025404e-01, 5.00000000e-01, 6.12323400e-17])

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'EmissionAngles', 0, 180)
        return obj


class PhaseAngles(_Angles):
    """Designate that an array represents phase angles.

    This object otherwise acts like an ndarray.

    Parameters
    ----------
    array
        Any array of phase angles [degrees]. Must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 180.

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'PhaseAngles', 0, 180)
        return obj


class AzimuthAngles(_Angles):
    """Designate that an array represents azimuth angles.

    This object otherwise acts like an ndarray.

    Parameters
    ----------
    array
        Any array of phase angles [degrees]. Must be between 0 and 360.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if any values in the input array are not between 0 and 360.

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'AzimuthAngles', 0, 360)
        return obj


def azimuth(incidence: IncidenceAngles, emission: EmissionAngles,
            phase: PhaseAngles) -> AzimuthAngles:
    r"""Construct azimuth angles from a set of incidence, emission, and phase
    angles.

    Parameters
    ----------
    incidence: IncidenceAngles
        Incidence (solar zenith) angles [degrees].
    emission: EmissionAngles
        Emission (emergence) angles [degrees].
    phase: PhaseAngles
        Phase angles [degrees].

    Raises
    ------
    ValueError
        Raised if the input arrays do not have compatible shapes.

    Notes
    -----
    The inputs can be ndarrays but then there's no angle validation.
    Consequently, the results may not accurate in this scenario. I recommend
    using the correctly typed inputs.

    Examples
    --------
    Create the azimuth angles from an assortment of angles.

    >>> import numpy as np
    >>> import pyrt
    >>> incidence_angles = pyrt.IncidenceAngles(np.array([20, 30, 40]))
    >>> emission_angles = pyrt.EmissionAngles(np.array([30, 40, 50]))
    >>> phase_angles = pyrt.PhaseAngles(np.array([25, 30, 35]))
    >>> pyrt.azimuth(incidence_angles, emission_angles, phase_angles)
    AzimuthAngles([122.74921226, 129.08074256, 131.57329276])

    """
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp_arg = np.true_divide(
                np.cos(np.radians(phase)) - np.cos(np.radians(emission)) *
                np.cos(np.radians(incidence)),
                np.sin(np.radians(emission)) * np.sin(np.radians(incidence)))
            tmp_arg[~np.isfinite(tmp_arg)] = -1
            d_phi = np.arccos(np.clip(tmp_arg, -1, 1))
        return AzimuthAngles(180 - np.degrees(d_phi))
    except ValueError as ve:
        message = f'The input arrays must have compatible shapes. They are' \
                  f'incidence: {incidence.shape}, ' \
                  f'emission: {emission.shape}, and ' \
                  f'phase: {phase.shape}.'
        raise ValueError(message) from ve
