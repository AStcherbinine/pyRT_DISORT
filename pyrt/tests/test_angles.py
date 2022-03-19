import numpy as np
import pytest
from pyrt import azimuth


class TestAzimuth:
    @pytest.fixture
    def angles(self) -> np.ndarray:
        yield np.array([0, 10, 70])

    @pytest.fixture
    def known_azimuth(self) -> np.ndarray:
        yield np.array([0, 119.74712028, 104.76497572])

    @pytest.fixture
    def largest_negative(self) -> float:
        yield np.nextafter(0, -1)

    @pytest.fixture
    def non_numeric_angles(self) -> float:
        yield np.array([0, 10, {'a': 1}])

    def test_non_numeric_incidence_angles_raises_type_error(self, non_numeric_angles):
        with pytest.raises(TypeError):
            azimuth(non_numeric_angles, 10, 10)

    def test_incidence_angles_less_than_0_raises_value_error(self, largest_negative):
        azimuth(0, 10, 10)
        with pytest.raises(ValueError):
            azimuth(largest_negative, 10, 10)

    def test_indicence_angles_greater_than_90_raises_value_error(self):
        azimuth(90, 10, 10)
        larger = np.nextafter(90, 91)
        with pytest.raises(ValueError):
            azimuth(larger, 10, 10)

    def test_non_numeric_emission_angles_raises_type_error(self, non_numeric_angles):
        with pytest.raises(TypeError):
            azimuth(10, non_numeric_angles, 10)

    def test_emission_angles_less_than_0_raises_value_error(self, largest_negative):
        azimuth(10, 0, 10)
        with pytest.raises(ValueError):
            azimuth(10, largest_negative, 10)

    def test_emission_angles_greater_than_90_raises_value_error(self):
        azimuth(10, 90, 10)
        larger = np.nextafter(90, 91)
        with pytest.raises(ValueError):
            azimuth(10, larger, 10)

    def test_non_numeric_phase_angles_raises_type_error(self, non_numeric_angles):
        with pytest.raises(TypeError):
            azimuth(10, 10, non_numeric_angles)

    def test_phase_angles_less_than_0_raises_value_error(self, largest_negative):
        azimuth(10, 10, 0)
        with pytest.raises(ValueError):
            azimuth(10, 10, largest_negative)

    def test_phase_angles_greater_than_180_raises_value_error(self):
        azimuth(10, 10, 180)
        larger = np.nextafter(180, 181)
        with pytest.raises(ValueError):
            azimuth(10, 10, larger)

    def test_known_angles_yield_known_azimuth(self, angles, known_azimuth):
        np.testing.assert_almost_equal(azimuth(angles, angles, angles), known_azimuth)
