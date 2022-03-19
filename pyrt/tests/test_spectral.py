import numpy as np
import pytest
from pyrt import wavenumber


class TestWavenumber:
    @pytest.fixture
    def known_wavelengths(self) -> np.ndarray:
        yield [1, 10, 40]

    @pytest.fixture
    def known_wavenumbers(self) -> np.ndarray:
        yield np.array([10000, 1000, 250])

    @pytest.fixture
    def largest_small_wavelength(self) -> float:
        yield np.nextafter(0.1, 0)

    @pytest.fixture
    def smallest_large_wavelength(self) -> float:
        yield np.nextafter(50, 51)

    @pytest.fixture
    def non_numeric_wavelength(self) -> float:
        yield np.array([0, 10, {'a': 1}])

    def test_non_numeric_wavelength_raises_type_error(
            self, non_numeric_wavelength):
        with pytest.raises(TypeError):
            wavenumber(non_numeric_wavelength)

    def test_too_small_wavelength_raises_value_error(
            self, largest_small_wavelength):
        wavenumber(0.1)
        with pytest.raises(ValueError):
            wavenumber(largest_small_wavelength)

    def test_too_large_wavelength_raises_value_error(
            self, smallest_large_wavelength):
        wavenumber(50)
        with pytest.raises(ValueError):
            wavenumber(smallest_large_wavelength)

    def test_known_wavelengths_returns_known_wavenumbers(
            self, known_wavelengths, known_wavenumbers):
        np.testing.assert_equal(wavenumber(known_wavelengths),
                                known_wavenumbers)
