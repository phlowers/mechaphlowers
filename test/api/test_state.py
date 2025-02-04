import numpy as np
import pytest
from mechaphlowers.api.state import StateAccessor


class MockPhysics:
	def L_ref(self, current_temperature):
		return current_temperature

class MockSection:
	def __init__(self, data_shape):
		self.section = self
		self.data = np.zeros(data_shape)
		self.physics = MockPhysics()

@pytest.fixture
def section_dataframe():
	return MockSection((5,))

#---------Tests---------

def test_L_ref_with_float(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	current_temperature = 25.0
	result = state_accessor.L_ref(current_temperature)
	expected = np.full(5, current_temperature)
	np.testing.assert_array_equal(result, expected)

def test_L_ref_with_correct_array(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	current_temperature = np.array([25.0, 26.0, 27.0, 28.0, 29.0])
	result = state_accessor.L_ref(current_temperature)
	np.testing.assert_array_equal(result, current_temperature)

def test_L_ref_with_incorrect_array_length(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	current_temperature = np.array([25.0, 26.0])
	with pytest.raises(ValueError, match="Current temperature should have the same length as the section"):
		state_accessor.L_ref(current_temperature)