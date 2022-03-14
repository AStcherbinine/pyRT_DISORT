from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simps
from scipy.interpolate import interp1d


class _ParticleSizes(np.ndarray):
    def __new__(cls, array: ArrayLike, name: str):
        obj = cls._make_array(array).view(cls)
        obj.name = name
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    @staticmethod
    def _make_array(obj: np.ndarray):
        try:
            obj = np.asarray(obj)
            obj.astype(float)
        except TypeError as te:
            message = 'The particle size must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The particle sizes must be numeric.'
            raise ValueError(message) from ve
        return obj

    @staticmethod
    def _validate(array):
        if not np.all(array > 0):
            message = 'The particle sizes must be positive.'
            raise ValueError(message)
        if not np.ndim(array) == 1:
            message = 'The particle sizes must be 1-dimensional.'
            raise ValueError(message)


class _Wavelengths(np.ndarray):
    def __new__(cls, array: ArrayLike, name: str):
        obj = cls._make_array(array).view(cls)
        obj.name = name
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    @staticmethod
    def _make_array(value):
        try:
            array = np.asarray(value)
            array.astype(float)
        except TypeError as te:
            message = 'The wavelengths must be ArrayLike.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'The wavelengths must be numeric.'
            raise ValueError(message) from ve
        return array

    @staticmethod
    def _validate(array):
        if not np.all((0.1 <= array) & (array <= 50)):
            message = 'The wavelengths must be between 0.1 and 50 microns.'
            raise ValueError(message)
        if not np.ndim(array) == 1:
            message = 'The wavelengths must be 1-dimensional.'
            raise ValueError(message)


class _FiniteNumericArray(np.ndarray):
    def __new__(cls, array: ArrayLike, name: str):
        obj = cls._make_array(array).view(cls)
        obj.name = name
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

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


class _ParticleSizeWavelengthGriddedArray(ABC):
    def __init__(self,
                 array: ArrayLike,
                 particle_size_grid: ArrayLike,
                 wavelength_grid: ArrayLike):
        self._arr = _FiniteNumericArray(array, 'array')
        self._reff_grid = _ParticleSizes(particle_size_grid, 'particle_size_grid')
        self._wav_grid = _Wavelengths(wavelength_grid, 'wavelength_grid')
        self._raise_value_error_if_array_dims_do_not_match()

    def _raise_value_error_if_array_dims_do_not_match(self):
        if self._arr.shape[-2] != self._reff_grid.shape[0]:
            message = 'The particle size dimension does not match the input array.'
            raise ValueError(message)
        if self._arr.shape[-1] != self._wav_grid.shape[0]:
            message = 'The wavelength dimension does not match the input array.'
            raise ValueError(message)

    def _regrid(self,
                particle_sizes: ArrayLike,
                wavelengths: ArrayLike) \
            -> np.ndarray:
        """Regrid the input array onto a new particle size and wavelength grid
        using nearest neighbor 'interpolation'.

        Parameters
        ----------
        particle_sizes: ArrayLike
            The particle sizes to regrid the array on to.
        wavelengths: ArrayLike
            The wavelengths to regrid the array on to.

        Returns
        -------
        np.ndarray
            Regridded array of shape (..., particle_sizes, wavelengths)

        """
        psizes = _ParticleSizes(particle_sizes, 'particle_sizes')
        wavs = _Wavelengths(wavelengths, 'wavelengths')
        reff_indices = self._get_nearest_indices(self._reff_grid, psizes)
        wav_indices = self._get_nearest_indices(self._wav_grid, wavs)
        return np.take(np.take(self._arr, reff_indices, axis=-2),
                       wav_indices, axis=-1)

    @staticmethod
    def _get_nearest_indices(grid: np.ndarray, values: np.ndarray) \
            -> np.ndarray:
        # grid should be 1D; values can be ND
        return np.abs(np.subtract.outer(grid, values)).argmin(0)


class _ScatteringAngles(np.ndarray):
    def __new__(cls, array: ArrayLike, name: str):
        obj = cls._make_array(array).view(cls)
        obj.name = name
        cls._validate(obj)
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

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


class _PhaseFunctionND(_FiniteNumericArray):
    def __new__(cls, array: ArrayLike, name: str):
        obj = super().__new__(cls, array, name)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.all(array >= 0):
            message = 'The phase function must be non-negative'
            raise ValueError(message)


class _PhaseFunction1D(_PhaseFunctionND):
    def __new__(cls, array: ArrayLike, name: str):
        obj = super().__new__(cls, array, name)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.ndim(array) == 1:
            message = 'The phase function must be 1-dimensional.'
            raise ValueError(message)


class _PhaseFunction3D(_PhaseFunctionND):
    def __new__(cls, array: ArrayLike, name: str):
        obj = super().__new__(cls, array, name)
        cls._validate(obj)
        return obj

    @staticmethod
    def _validate(array):
        if not np.ndim(array) == 3:
            message = 'The phase function must be 3-dimensional.'
            raise ValueError(message)


class _PhaseFunction(ABC):
    def __init__(self, phase_function, scattering_angles):
        self._pf = _PhaseFunctionND(phase_function, 'phase_function')
        self._sa = _ScatteringAngles(scattering_angles, 'scattering_angles')

    def _resample(self, samples: int) -> None:
        samples = self._validate_samples(samples)
        f = interp1d(self._sa, self._pf, axis=0)
        angles = np.linspace(self._sa[0], self._sa[-1], num=samples)
        self._pf = f(angles)
        self._sa = angles

    def _normalize(self) -> None:
        norm = np.abs(simps(self._pf, np.cos(np.radians(self._sa)), axis=0))
        self._pf = 2 * self._pf / norm

    @abstractmethod
    def decompose(self, moments: int):
        raise NotImplementedError('Not implemented yet!')

    @staticmethod
    def _validate_samples(samples: int):
        try:
            return int(samples)
        except TypeError as te:
            message = 'samples must be an int.'
            raise TypeError(message) from te
        except ValueError as ve:
            message = 'samples cannot be converted to an int.'
            raise ValueError(message) from ve

    def _validate_moments(self, moments: int):
        samples = len(self._sa)
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


class PhaseFunction(_PhaseFunction):
    def __init__(self, phase_function, scattering_angles):
        super().__init__(phase_function, scattering_angles)
        self._pf = _PhaseFunction1D(phase_function, 'phase_function')

    @property
    def phase_function(self):
        return self._pf

    @property
    def scattering_angles(self):
        return self._sa

    def resample(self, samples: int) -> None:
        return self._resample(samples)

    def normalize(self) -> None:
        return self._normalize()

    def _decompose(self, moments: int):
        """Decompose the phase function into its Legendre coefficients.

        Parameters
        ----------
        moments: int
            The number of moments to decompose the phase function into.

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

        >>> from pathlib import Path
        >>> import numpy as np
        >>> import pyrt
        >>> dust_dir = Path(__file__).parent.parent / 'anc' / 'mars_dust'
        >>> phsfn = np.load(dust_dir / 'phase_function.npy')[:, 0, 0]
        >>> ang = np.load(dust_dir / 'scattering_angles.npy')
        >>> pf = pyrt.PhaseFunction(phsfn, ang)
        >>> moments = pf.decompose(129)
        >>> print(moments[:5])
        [1.         0.17778457 0.50944022 0.03520301 0.00162705]

        """
        n_moments = self._validate_moments(moments)
        self.normalize()
        self._sa = np.radians(self._sa)
        # Subtract 1 since I'm forcing c0 = 1 in the equation
        # P(x) = c0 + c1*L1(x) + ... for DISORT
        self._pf -= 1
        lpoly = self._make_legendre_polynomials(n_moments)
        normal_matrix = self._make_normal_matrix(lpoly)
        normal_vector = self._make_normal_vector(lpoly)
        cholesky = np.linalg.cholesky(normal_matrix)
        first_solution = np.linalg.solve(cholesky, normal_vector)
        second_solution = np.linalg.solve(cholesky.T, first_solution)
        coeff = np.concatenate((np.array([1]), second_solution))
        # Re-add 1 for the reason listed above
        self._pf += 1
        self._sa = np.degrees(self._sa)
        return coeff

    def decompose(self, moments):
        return LegendreCoefficients(self._decompose(moments), self._sa)

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


class PhaseFunctionGrid(_PhaseFunction, _ParticleSizeWavelengthGriddedArray):
    def __init__(self, phase_function, scattering_angles, particle_size_grid, wavelength_grid):
        _PhaseFunction.__init__(self, phase_function, scattering_angles)
        _ParticleSizeWavelengthGriddedArray.__init__(self, phase_function, particle_size_grid, wavelength_grid)
        self._pf = _PhaseFunction3D(phase_function, 'phase_function')

    @property
    def phase_function(self):
        return self._pf

    @property
    def scattering_angles(self):
        return self._sa

    @property
    def particle_size_grid(self):
        return self._reff_grid

    @property
    def wavelength_grid(self):
        return self._wav_grid

    def resample(self, samples: int) -> None:
        self._resample(samples)

    def normalize(self) -> None:
        self._normalize()

    def regrid(self, particle_sizes: ArrayLike, wavelengths: ArrayLike):
        regrid = self._regrid(particle_sizes, wavelengths)
        return PhaseFunctionGradient(regrid, self._sa, particle_sizes, wavelengths)

    def decompose(self, moments: int):
        # NOTE: this is slow cause of the for loop
        moments = self._validate_moments(moments)
        arr = np.zeros((moments, len(self._reff_grid), len(self._wav_grid)))
        for i in range(len(self._reff_grid)):
            for j in range(len(self._wav_grid)):
                pf = PhaseFunction(self.phase_function[:, i, j], self.scattering_angles)
                arr[:, i, j] = pf.decompose(moments)
        return arr


class PhaseFunctionGradient(_PhaseFunction, _ParticleSizeWavelengthGriddedArray):
    def __init__(self, phase_function, scattering_angles, particle_size_grad, wavelength_grad):
        _PhaseFunction.__init__(self, phase_function, scattering_angles)
        _ParticleSizeWavelengthGriddedArray.__init__(self, phase_function, particle_size_grad, wavelength_grad)
        self._pf = _PhaseFunction3D(phase_function, 'phase_function')

    @property
    def phase_function(self):
        return self._pf

    @property
    def scattering_angles(self):
        return self._sa

    @property
    def particle_size_gradient(self):
        return self._reff_grid

    @property
    def wavelength_grid(self):
        return self._wav_grid

    def resample(self, samples: int) -> None:
        self._resample(samples)

    def normalize(self) -> None:
        self._normalize()

    def decompose(self, moments: int):
        # NOTE: this is slow cause of the for loop
        moments = self._validate_moments(moments)
        arr = np.zeros((moments, len(self._reff_grid), len(self._wav_grid)))
        for i in range(len(self._reff_grid)):
            for j in range(len(self._wav_grid)):
                pf = PhaseFunction(self.phase_function[:, i, j], self.scattering_angles)
                arr[:, i, j] = pf._decompose(moments)
        return LegendreCoefficientsGradient(arr, self.scattering_angles, self.particle_size_gradient, self.wavelength_grid)


class _LegendreCoefficients(ABC):
    def __init__(self, coefficients: ArrayLike, scattering_angles: ArrayLike):
        self._coef = _FiniteNumericArray(coefficients, 'coefficient')
        self._sa = _ScatteringAngles(scattering_angles, 'scattering_angles')

    def _set_negative_coefficients_to_0(self) -> None:
        argmax = np.argmax(self._coef < 0, axis=0)
        c = np.indices(self._coef.shape)[0, ...]
        cond = c >= argmax
        self._coef[cond] = 0

    def _reconstruct_phase_function(self):
        pfs = np.moveaxis(np.polynomial.legendre.legval(np.cos(np.radians(self._sa)), self._coef), -1, 0)
        #angles = np.radians(np.arange(180))
        #pf = np.polynomial.legendre.legval(np.cos(angles), self)
        return pfs


class LegendreCoefficients(_LegendreCoefficients):
    def __init__(self, coefficients: ArrayLike, scattering_angles: ArrayLike):
        super().__init__(coefficients, scattering_angles)

    def set_negative_coefficients_to_0(self) -> None:
        self._set_negative_coefficients_to_0()

    def reconstruct(self):
        pf = self._reconstruct_phase_function()
        return PhaseFunction(pf, self._sa)


class LegendreCoefficientsGradient(_LegendreCoefficients):
    def __init__(self, coefficients: ArrayLike, scattering_angles: ArrayLike, particle_sizes, wavelengths):
        super().__init__(coefficients, scattering_angles)
        self._reff = _ParticleSizes(particle_sizes, 'particle_sizes')
        self._wavs = _Wavelengths(wavelengths, 'wavelengths')

    def set_negative_coefficients_to_0(self) -> None:
        self._set_negative_coefficients_to_0()

    def reconstruct(self):
        pf = self._reconstruct_phase_function()
        return PhaseFunctionGradient(pf, self._sa, self._reff, self._wavs)

'''






class PhaseFunctionGradient(_PhaseFunction, _ParticleSizeWavelengthGriddedArray):
    _pf = _PhaseFunction3D()

    def __init__(self, phase_function, scattering_angles, particle_size_grid, wavelength_grid):
        _PhaseFunction.__init__(self, phase_function, scattering_angles)
        _ParticleSizeWavelengthGriddedArray.__init__(self, phase_function, particle_size_grid, wavelength_grid)

    def resample(self, samples: int) -> None:
        return self._resample(samples)

    def normalize(self) -> None:
        return self._normalize()

    def decompose(self, moments: int):
        pass
        ''''''arr = np.zeros((moments, len(particle_sizes), len(wavelengths)))
        for i in range(len(particle_sizes)):
            for j in range(len(wavelengths)):
                pf = PhaseFunction(self._pf[:, i, j], self.scattering_angles)
                arr[:, i, j] = pf.decompose(moments)
        return arr'''''''''


'''class LegendreCoefficients(np.ndarray):
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
    def __new__(cls, coefficients: ArrayLike):
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
        return PhaseFunction(pf, angles)'''


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

    pf = PhaseFunction(phsfn[:, 0, 0], sa)
    a = pf.decompose(129)
    a.set_negative_coefficients_to_0()
    b = a.reconstruct()
    print(np.amax((phsfn[:, 0, 0] - b.phase_function)**2))


    foo = PhaseFunctionGrid(phsfn, sa, reff, wavs)
    bar = foo.regrid([1, 2], [3, 4, 5])
    lc = bar.decompose(129)
    lc.set_negative_coefficients_to_0()
    print(lc._coef.shape)

    raise SystemExit(9)

    pf = PhaseFunctionGrid(phsfn, sa, reff, wavs)
    oldpfs = pf.regrid([1, 2], [3, 4, 5])
    print(type(oldpfs))
    print(oldpfs._reff_grid)
    angles = np.radians(np.arange(181))
    #pfs = np.moveaxis(np.polynomial.legendre.legval(np.cos(angles), a), -1, 0)
    #b = (oldpfs - pfs)**2
    # print(np.amax(b))

    #pf = PhaseFunction(phsfn, sa, reff, wavs)
    #a = pf.decompose(129, [1, 2], [3, 4, 5])
    #print(a.shape)
    #print(pf.phase_function.shape)

    #mpf = (pf.phase_function[:-1, ...] + pf.phase_function[1:, ...]) / 2
    #print(np.sum((mpf.T * np.abs(np.diff(np.cos(np.radians(pf.scattering_angles))))).T, axis=0))
    #a = pf.regrid([1, 2], [3, 4, 5])
    #print(a.shape)
