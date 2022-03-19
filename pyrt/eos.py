"""The :code:`eos` module contains data structures to compute and hold equation
of state variables used throughout pyRT_DISORT.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import Boltzmann
from scipy.integrate import quadrature as quad


# TODO: Fix the Raises docstring formatting
class Hydrostatic:
    """A data structure that computes a hydrostatic equation of state.

    Hydrostatic accepts equation of state variables and regrids them to a
    user-specified altitude grid using linear interpolation. Then, it computes
    number density and scale height at the new boundaries, and the
    column density within the new boundaries, assuming the atmosphere is in
    hydrostatic equilibrium.

    """

    def __init__(self, altitude_grid: np.ndarray, pressure_grid: np.ndarray,
                 temperature_grid: np.ndarray, altitude_boundaries: np.ndarray,
                 particle_mass: float, gravity: float) -> None:
        r"""
        Parameters
        ----------
        altitude_grid
            The altitude grid [km] over which the equation of state variables
            are defined. See the note below for additional conditions.
        pressure_grid
            The pressure [Pa] at all values in ``altitude_grid``.
        temperature_grid
            The temperature [K] at all values in ``altitude_grid``.
        altitude_boundaries
            The desired boundary altitude [km]. See the note below for
            additional conditions.
        particle_mass
            The average mass [kg] of atmospheric particles.
        gravity
            The gravitational acceleration
            [:math:`\frac{\text{kg m}}{\text{s}^2}`] of the atmosphere.

        Raises
        ------
        TypeError
            Raised if ``altitude_grid``, ``pressure_grid``,
            ``temperature_grid``, or ``altitude_boundaries`` are not all
            numpy.ndarrays; or if ``particle_mass`` or ``gravity`` are not
            floats.
        ValueError
            Raised if:
               * ``altitude_grid``, ``pressure_grid``, or
                  ``temperature_grid`` do not have the same shapes
               * ``altitude_grid`` or ``altitude_boundaries`` have
                 incompatible pixel dimensions
               * ``altitude_grid`` or ``altitude_boundaries`` are not
                 monotonically decreasing along the 0 :sup:`th` axis;
               * ``pressure_grid``, or ``temperature_grid`` contain
                 non-positive, finite values
               * ``altitude_boundaries`` does not contain at least 2
                 boundaries
               * ``particle_mass`` or ``gravity`` are not positive, finite

        Notes
        -----
        This class assumes the atmosphere follows the equation

        .. math::
           :label: hydrostatic_equation

           P = n k_B T

        where :math:`P` is the pressure, :math:`n` is the number density,
        :math:`k_B` is Boltzmann's constant, and :math:`T` is the temperature.

        The inputs can be ND arrays, as long as they have compatible shapes. In
        this scenario, :code:`altitude_grid`, :code:`pressure_grid`, and
        :code:`temperature_grid` must be of shape Mx(pixels) whereas
        :code:`altitude_boundaries` must be of shape Nx(pixels), as long as
        N > 1 to ensure that the model has at least 1 layer.

        To keep with DISORT's convention, :code:`altitude_grid` and
        :code:`altitude_boundaries` must be monotonically decreasing. If these
        are ND arrays, this condition only applies to the 0 :sup:`th` axis.

        Also, scipy's Gaussian quadrature routine becomes less accurate the
        smaller the atmosphere's scale height is. I'm working to reduce the
        errors. In the meantime the column density is fairly close to analytical
        results but should be improved.

        """
        self.__altitude_grid = _Altitude(altitude_grid, 'altitude_grid')
        self.__pressure_grid = _EoSVar(pressure_grid, 'pressure_grid')
        self.__temperature_grid = _EoSVar(temperature_grid, 'temperature_grid')
        self.__mass = _ScaleHeightVar(particle_mass, 'particle_mass')
        self.__gravity = _ScaleHeightVar(gravity, 'gravity')
        self.__altitude = _Altitude(altitude_boundaries, 'altitude_boundaries')

        self.__raise_error_if_inputs_are_bad()

        self.__n_layers = self.__extract_n_layers()

        self.__pressure = \
            self.__interpolate_to_boundary_alts(pressure_grid)
        self.__temperature = \
            self.__interpolate_to_boundary_alts(temperature_grid)
        self.__number_density = \
            self.__compute_number_density(self.__pressure, self.__temperature)
        self.__column_density = \
            self.__compute_column_density()
        self.__scale_height = \
            self.__compute_scale_height(particle_mass, gravity)

    def __raise_error_if_inputs_are_bad(self) -> None:
        self.__raise_value_error_if_eos_vars_are_not_all_same_shape()
        self.__raise_value_error_if_altitudes_do_not_match_pixel_dim()
        self.__raise_value_error_if_model_has_too_few_boundaries()

    def __raise_value_error_if_eos_vars_are_not_all_same_shape(self) -> None:
        if not self.__altitude_grid.shape == self.__pressure_grid.shape == \
               self.__temperature_grid.shape:
            message = 'altitude_grid, pressure_grid, and temperature_grid ' \
                      'must have the same shapes.'
            raise ValueError(message)

    def __raise_value_error_if_altitudes_do_not_match_pixel_dim(self) -> None:
        if self.__altitude_grid.shape[1:] != self.__altitude.shape[1:]:
            message = 'altitude_grid and altitude_boundaries can have ' \
                      'different shapes along the 0th axis but must have' \
                      'the same shape along all subsequent axes.'
            raise ValueError(message)

    def __raise_value_error_if_model_has_too_few_boundaries(self) -> None:
        if self.__altitude.shape[0] < 2:
            message = 'altitude_boundaries must contain at least 2 boundaries.'
            raise ValueError(message)

    def __extract_n_layers(self) -> int:
        return self.__altitude.shape[0] - 1

    # TODO: Ideally I'd like to vectorize this
    def __interpolate_to_boundary_alts(self, grid: np.ndarray) -> np.ndarray:
        flattened_altitude_grid = \
            self.__flatten_along_pixel_dimension(self.__altitude_grid.val)
        flattened_boundaries = \
            self.__flatten_along_pixel_dimension(self.__altitude.val)
        flattened_quantity_grid = self.__flatten_along_pixel_dimension(grid)
        interpolated_quantity = np.zeros(flattened_boundaries.shape)
        for pixel in range(flattened_boundaries.shape[1]):
            interpolated_quantity[:, pixel] = \
                np.interp(flattened_boundaries[:, pixel],
                          np.flip(flattened_altitude_grid[:, pixel]),
                          np.flip(flattened_quantity_grid[:, pixel]))
        return interpolated_quantity.reshape(self.__altitude.shape)

    @staticmethod
    def __flatten_along_pixel_dimension(grid: np.ndarray) -> np.ndarray:
        return grid.reshape(grid.shape[0], int(grid.size / grid.shape[0]))

    @staticmethod
    def __compute_number_density(pressure: np.ndarray,
                                 temperature: np.ndarray) -> np.ndarray:
        return pressure / temperature / Boltzmann

    # TODO: Ideally I'd like to vectorize this
    # TODO: Mike said to do this in log(z) space. Is this still necessary?
    # TODO: (Related) This introduces some errors. I reckon using log(z) space
    #  will fix them.
    def __compute_column_density(self) -> np.ndarray:
        flattened_boundaries = \
            np.flipud(self.__flatten_along_pixel_dimension(self.__altitude.val)
                      * 1000)
        flattened_pressure = \
            np.flipud(self.__flatten_along_pixel_dimension(self.__pressure))
        flattened_temperature = \
            np.flipud(self.__flatten_along_pixel_dimension(self.__temperature))
        column_density = np.zeros((self.__n_layers,
                                   flattened_boundaries.shape[1]))
        for pixel in range(flattened_boundaries.shape[1]):
            colden = [quad(self.__make_number_density_at_altitude,
                           flattened_boundaries[i, pixel],
                           flattened_boundaries[i+1, pixel],
                           args=(flattened_boundaries[:, pixel],
                                 flattened_pressure[:, pixel],
                                 flattened_temperature[:, pixel]))[0]
                      for i in range(self.__n_layers)]
            column_density[:, pixel] = np.flip(np.array(colden))
        if np.ndim(self.__altitude) == 1:
            return np.squeeze(column_density)
        else:
            return column_density.reshape((column_density.shape[0],) +
                                          self.__altitude.shape[1:])

    def __make_number_density_at_altitude(
            self, z: float, alt_grid: np.ndarray, pressure: np.ndarray,
            temperature: np.ndarray) -> np.ndarray:
        p = np.interp(z, alt_grid, pressure)
        t = np.interp(z, alt_grid, temperature)
        return self.__compute_number_density(p, t)

    def __compute_scale_height(self, particle_mass: float,
                               gravity: float) -> np.ndarray:
        return Boltzmann * self.__temperature / (particle_mass * gravity)

    @property
    def n_layers(self) -> int:
        """Get the number of layers in the model.

        Notes
        -----
        This value is inferred from the 0 :sup:`th` axis of
        ``altitude_boundaries``.

        In DISORT, this variable is named ``MAXCLY`` (though in the ``disort``
        package, this variable is optional).

        """
        return self.__n_layers

    @property
    def altitude(self) -> np.ndarray:
        """Get the input boundary altitude [km].

        """
        return self.__altitude.val

    @property
    def pressure(self) -> np.ndarray:
        """Get the pressure [Pa] at the boundary altitude.

        Notes
        -----
        This variable is obtained by linearly interpolating the input pressure
        onto ``altitude_boundaries``.

        """
        return self.__pressure

    @property
    def temperature(self) -> np.ndarray:
        """Get the temperature [K] at the boundary altitude.

        Notes
        -----
        This variable is obtained by linearly interpolating the input
        temperature onto ``altitude_boundaries``.

        In DISORT, this variable is named ``TEMPER``. It is only needed by
        DISORT if :py:attr:`~radiation.ThermalEmission.thermal_emission` is set
        to ``True``.

        """
        return self.__temperature

    @property
    def number_density(self) -> np.ndarray:
        r"""Get the number density [:math:`\frac{\text{particles}}{\text{m}^3}`]
        at the boundary altitude.

        Notes
        -----
        This variable is obtained by getting the pressure and temperature at the
        boundary altitude, then solving :eq:`hydrostatic_equation`.

        """
        return self.__number_density

    @property
    def column_density(self) -> np.ndarray:
        r"""Get the column density [:math:`\frac{\text{particles}}{\text{m}^2}`]
        of the boundary *layers*.

        Notes
        -----
        This is obtained by getting the number density at the boundary altitude,
        then integrating (using Gaussian quadrature) between the boundary
        altitude such that

        .. math::
           N = \int n(z) dz

        is satisfied, where :math:`N` is the column density and :math:`n(z)` is
        the number density.

        """
        return self.__column_density

    @property
    def scale_height(self) -> np.ndarray:
        r"""Get the scale height [km] at the boundary altitude.

        Notes
        -----
        For a hydrostatic atmosphere, the scale height is defined as

        .. math::
           H = \frac{k_B T}{mg}

        where :math:`H` is the scale height, :math:`k_B` is Boltzmann's
        constant, :math:`T` is the temperature, :math:`m` is the average mass
        of an atmospheric particle, and :math:`g` is the planetary gravity.

        In DISORT, this variable is named ``H_LYR``. Despite the name, this
        variable should have length of ``n_layers + 1``. It is only used if
        :py:attr:`~controller.ModelBehavior.do_pseudo_sphere` is set to
        ``True``.

        """
        return self.__scale_height


class _Altitude:
    """Perform checks that a given altitude is plausible.

    _Altitude accepts altitudes and ensures they're monotonically decreasing.

    """

    def __init__(self, altitude: np.ndarray, name: str) -> None:
        """
        Parameters
        ----------
        altitude
            Array of altitude.
        name
            Name of the altitude array.

        Raises
        ------
        TypeError
            Raised if `altitude`` is not a numpy.ndarray
        ValueError
            Raised if ``altitude`` is not monotonically decreasing along
            the 0th axis.

        """
        self.__altitude = altitude
        self.__name = name

        self.__raise_error_if_input_is_bad()

    def __raise_error_if_input_is_bad(self) -> None:
        self.__raise_type_error_if_not_ndarray()
        self.__raise_value_error_if_not_monotonically_decreasing()

    def __raise_type_error_if_not_ndarray(self) -> None:
        if not isinstance(self.__altitude, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_not_monotonically_decreasing(self) -> None:
        if not np.all(np.diff(self.__altitude, axis=0) < 0):
            message = f'{self.__name} must be monotonically decreasing along ' \
                      f'the 0th axis.'
            raise ValueError(message)

    def __getattr__(self, method):
        return getattr(self.val, method)

    @property
    def val(self) -> np.ndarray:
        return self.__altitude




class _EoSVar:
    """Perform checks that a given equation of state variable is plausible.

    _EoSVar accepts equation of state variables to ensure they're physically
    allowable.

    """

    def __init__(self, variable: np.ndarray, name: str) -> None:
        """
        Parameters
        ----------
        variable
            Array of an equation of state variable.
        name
            The name of the variable.

        Raises
        ------
        TypeError
            Raised if ``variable`` is not a numpy.ndarray.
        ValueError
            Raised if ``variable`` contains negative values.

        """
        self.__var = variable
        self.__name = name

        self.__raise_error_if_input_is_bad()

    def __raise_error_if_input_is_bad(self) -> None:
        self.__raise_type_error_if_not_ndarray()
        self.__raise_value_error_if_contains_negative_values()

    def __raise_type_error_if_not_ndarray(self) -> None:
        if not isinstance(self.__var, np.ndarray):
            message = f'{self.__name} must be a numpy.ndarray.'
            raise TypeError(message)

    def __raise_value_error_if_contains_negative_values(self) -> None:
        if np.any(self.__var < 0) or np.any(np.isinf(self.__var)):
            message = f'{self.__name} must only contain positive finite values.'
            raise ValueError(message)

    def __getattr__(self, method):
        return getattr(self.val, method)

    @property
    def val(self) -> np.ndarray:
        """Get the input variable.

        """
        return self.__var


class _ScaleHeightVar:
    """Perform checks that a given scale height variable is plausible.

    _ScaleHeightVar accepts scale height variables to ensure they're physically
    allowable.

    """

    def __init__(self, variable: float, name: str) -> None:
        """
        Parameters
        ----------
        variable
            Array of an equation of state variable.
        name
            The name of the variable.

        Raises
        ------
        TypeError
            Raised if ``variable`` is not a float.
        ValueError
            Raised if ``variable`` contains negative values.

        """
        self.__var = variable
        self.__name = name

        self.__raise_error_if_input_is_bad()

    def __raise_error_if_input_is_bad(self) -> None:
        self.__raise_type_error_if_not_float()
        self.__raise_value_error_if_contains_negative_values()

    def __raise_type_error_if_not_float(self) -> None:
        if not isinstance(self.__var, float):
            message = f'{self.__name} must be a float.'
            raise TypeError(message)

    def __raise_value_error_if_contains_negative_values(self) -> None:
        if np.any(self.__var < 0) or np.any(np.isinf(self.__var)):
            message = f'{self.__name} must only contain positive finite values.'
            raise ValueError(message)

    def __getattr__(self, method):
        return getattr(self.val, method)

    @property
    def val(self) -> float:
        """Get the input variable.

        """
        return self.__var


class _Altitudes(np.ndarray):
    """A base class for designating that an input represents altitudes.

    This object otherwise acts like an ndarray.

    Parameters
    ----------
    array
        Any array of altitudes.
    name
        The name of the altitude array.

    Raises
    ------
    TypeError
        Raised if any values in the input array are nonnumerical.
    ValueError
        Raised if the input array is not 1-dimensional or if the values are
        not monotonically decreasing.

    """

    def __new__(cls, array: ArrayLike, name: str):
        obj = cls._make_array(array, name).view(cls)
        obj.name = name
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
        if array.ndim != 1:
            message = f'{array.name} must be 1-dimensional.'
            raise ValueError(message)
        if not np.all(np.diff(array) < 0):
            message = f'{array.name} must be monotonically decreasing.'
            raise ValueError(message)


def _validate_profile_value(value: float):
    try:
        val = float(value)
    except TypeError as te:
        message = 'value could not be converted to a float.'
        raise TypeError(message) from te
    if val <= 0:
        message = 'value must be positive.'
        raise ValueError(message)
    return val


def constant_profile(altitude: ArrayLike, value: float) -> np.ndarray:
    """Make a constant profile.

    Parameters
    ----------
    altitude: ArrayLike
        The altitudes at which to make the vertical profile. Must be
        1-dimensional and monotonically decreasing.
    value: float
        The value of the profile. Must be positive.

    Returns
    -------
    np.ndarray
        1-dimensional array of a constant value with shape ``altitude.shape``.

    Raises
    ------
    TypeError
        Raised if either of the inputs cannot be cast to the correct type.
    ValueError
        Raised if either of the inputs does not have the aforementioned
        desired properties.

    Examples
    --------
    Make a constant temperature profile of 150 K.

    >>> import numpy as np
    >>> import pyrt
    >>> z = np.linspace(100, 0, num=15)
    >>> pyrt.constant_profile(z, 150)
    array([150., 150., 150., 150., 150., 150., 150., 150., 150., 150., 150.,
           150., 150., 150., 150.])
    """
    alt = _Altitudes(altitude, 'altitude')
    val = _validate_profile_value(value)
    return np.zeros(len(alt)) + val


def exponential_profile(altitude: ArrayLike, surface: float, scale_height: ArrayLike):
    """Make an exponential profile.

    Parameters
    ----------
    altitude
    surface
    scale_height

    Returns
    -------

    """
    return surface * np.exp(-altitude / scale_height)





def linear_profile(altitude: ArrayLike, top: float, bottom: float):
    """Make a linear profile.

    Parameters
    ----------
    altitude
    top
    bottom

    Returns
    -------

    """
    return np.linspace(top, bottom, num=len(altitude))


def linear_grid_profile(altitude: ArrayLike, altitude_grid: ArrayLike, profile_grid: ArrayLike):
    """Make a profile with linear interpolation between grid points.

    Parameters
    ----------
    altitude
    altitude_grid
    profile_grid

    Returns
    -------

    """
    return np.interp(altitude, altitude_grid, profile_grid)


def log_grid_profile(altitude: ArrayLike, altitude_grid: ArrayLike, profile_grid: ArrayLike):
    """Make a profile with linear interpolation between grid points in log space.

    Parameters
    ----------
    altitude
    altitude_grid
    profile_grid

    Returns
    -------

    """
    return np.exp(np.interp(np.log(altitude), np.log(altitude_grid), profile_grid))


def scale_height(mass: float, gravity: float, temperature: ArrayLike) -> np.ndarray:
    """Compute the scale height.

    Parameters
    ----------
    mass
    gravity
    temperature

    Returns
    -------

    """
    return Boltzmann * temperature / mass / gravity


def column_density(pressure_profile, temperature_profile, altitude: ArrayLike, profargs, tempargs):
    """Make the column density.

    Parameters
    ----------
    pressure_profile
    temperature_profile
    altitude
    profargs
    tempargs

    Returns
    -------

    """
    def hydrostatic_profile(alts):
        return pressure_profile(alts, *profargs) / temperature_profile(alts, *tempargs) / Boltzmann

    n = [quad(hydrostatic_profile, altitude[i + 1], altitude[i])[0] for i in range(len(altitude) - 1)]
    return np.array(n) * 1000


if __name__ == '__main__':
    z = np.linspace(80, 0, num=15)
    #print(z)
    #p = hydrostatic_profile(z, 610, 10)

    a = column_density(exponential_profile, linear_profile, z, (610, 10), (150, 200))
    print(a)
    a = column_density(exponential_profile, linear_profile, z, (610, 10), (150, 150))
    print(a)
    b = column_density(exponential_profile, constant_profile, z, (610, 10), (150,))
    print(b)

    #n = p / t / Boltzmann
    #foo = np.array([quad(hydrostatic_profile, z[i+1], z[i], args=(n[-1], 10))[0] for i in range(len(z)-1)]) * 1000

    #print(np.sum(foo))
    #print(n[-1] * 10000 * (1-np.exp(-8)))
