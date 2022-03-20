import numpy as np
import pytest
from pyrt import constant_profile


class TestConstantProfile:
    @pytest.fixture
    def altitude(self):
        yield np.linspace(100, 0, num=15)

    @pytest.fixture
    def non_numeric_altitude(self) -> float:
        yield np.array([0, 10, {'a': 1}])

    def test_non_numeric_altitude_raises_type_error(
            self, non_numeric_altitude):
        with pytest.raises(TypeError):
            constant_profile(non_numeric_altitude, 100)

    def test_monotonically_increasing_altitude_raises_value_error(self):
        alt = np.linspace(0, 100, num=50)
        with pytest.raises(ValueError):
            constant_profile(alt, 100)

    def test_2_dimensional_altitude_raises_value_error(self):
        alt = np.ones((10, 10))
        with pytest.raises(ValueError):
            constant_profile(alt, 100)

    def test_ndarray_value_raises_type_error(self, altitude):
        arr = np.ones((10,))
        with pytest.raises(TypeError):
            constant_profile(altitude, arr)

    def test_function_gives_known_result(self, altitude):
        assert np.all(constant_profile(altitude, 150) == 150)
