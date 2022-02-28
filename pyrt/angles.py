from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def make_azimuth(incidence: ArrayLike, emission: ArrayLike,
                 phase: ArrayLike) -> np.ndarray:
    r"""Construct azimuth angles from a set of incidence, emission, and phase
    angles.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 90.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180.
    phase
        Phase angles [degrees]. All values must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range or if the input arrays do not have the same
        shapes.

    Examples
    --------
    Create the azimuth angles from an assortment of angles.

    >>> import numpy as np
    >>> import pyrt
    >>> incidence_angles = np.array([20, 30, 40])
    >>> emission_angles = np.array([30, 40, 50])
    >>> phase_angles = np.array([25, 30, 35])
    >>> pyrt.make_azimuth(incidence_angles, emission_angles, phase_angles)
    array([122.74921226, 129.08074256, 131.57329276])

    """
    bundle = _OrbiterAngleBundle(incidence, emission, phase)

    with np.errstate(divide='ignore', invalid='ignore'):
        tmp_arg = (np.true_divide(
            bundle.phase.cos() - bundle.emission.cos() * bundle.incidence.cos(),
            bundle.emission.sin() * bundle.incidence.sin())).to_ndarray()
        tmp_arg[~np.isfinite(tmp_arg)] = -1
        d_phi = np.arccos(np.clip(tmp_arg, -1, 1))

    return 180 - np.degrees(d_phi)


class _Angles(np.ndarray):
    """An abstract base class for designating that an input represents angles.

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

    def sin(self) -> _Angles:
        """Compute the sine of the input angles.

        """
        return np.sin(np.radians(self))

    def cos(self) -> _Angles:
        """Compute the cosine of the input angles.

        """
        return np.cos(np.radians(self))

    def to_ndarray(self) -> np.ndarray:
        """Turn this object into a generic ndarray.

        """
        return np.array(self)


class _IncidenceAngles(_Angles):
    """Designate that an input array represents incidence (solar zenith) angles.

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

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'incidence', 0, 90)
        return obj


class _EmissionAngles(_Angles):
    """Designate that an input array represents emission (emergence) angles.

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

    """
    def __new__(cls, array: ArrayLike):
        obj = super().__new__(cls, array, 'emission', 0, 180)
        return obj


class _PhaseAngles(_Angles):
    """Designate that an input array represents phase angles.

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
        obj = super().__new__(cls, array, 'phase', 0, 180)
        return obj


class _OrbiterAngleBundle:
    """Designate a collection of angles that represent those found in an
    orbiter observation as being linked.

    Parameters
    ----------
    incidence
        Incidence (solar zenith) angles [degrees]. All values must be between 0
        and 90.
    emission
        Emission (emergence) angles [degrees]. All values must be between 0 and
        180.
    phase
        Phase angles [degrees]. All values must be between 0 and 180.

    Raises
    ------
    TypeError
        Raised if any values in the input arrays are nonnumerical.
    ValueError
        Raised if any values of the input arrays are outside their
        mathematically valid range, or if the input arrays do not have the same
        shapes.

    """
    def __init__(self, incidence: ArrayLike, emission: ArrayLike,
                 phase: ArrayLike):
        self.incidence = _IncidenceAngles(incidence)
        self.emission = _EmissionAngles(emission)
        self.phase = _PhaseAngles(phase)

        self._raise_value_error_if_angle_shapes_do_not_match()

    def _raise_value_error_if_angle_shapes_do_not_match(self) -> None:
        if not (self.incidence.shape == self.emission.shape ==
                self.phase.shape):
            message = f'The shapes of the arrays must match. They are ' \
                      f'{self.incidence.shape}, {self.emission.shape}, and ' \
                      f'{self.phase.shape}'
            raise ValueError(message)
