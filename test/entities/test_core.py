import numpy as np

from mechaphlowers.config import options
from mechaphlowers.core.models.balance.interfaces import VhlStrength
from mechaphlowers.entities.core import QuantityArray


def test_vhl_strength():
    save_option = options.output_units.force
    options.output_units.force = "daN"

    arr = np.array(
        [[1000, 2000, 3000], [4000, 5000, 6000], [7000, 8000, 9000]]
    )
    vhl = VhlStrength(arr, input_unit="N")

    V, H, L = vhl.vhl
    np.testing.assert_array_equal(V.value, np.array([1000, 2000, 3000]) / 10)
    np.testing.assert_array_equal(H.value, np.array([4000, 5000, 6000]) / 10)
    np.testing.assert_array_equal(L.value, np.array([7000, 8000, 9000]) / 10)

    np.testing.assert_array_equal(V.value, vhl.V.value)
    np.testing.assert_array_equal(H.value, vhl.H.value)
    np.testing.assert_array_equal(L.value, vhl.L.value)

    options.output_units.force = save_option


def test_vhl_strength_str_repr():
    save_option = options.output_units.force
    options.output_units.force = "daN"

    arr = np.array([[1000, 2000], [3000, 4000], [5000, 6000]])
    vhl = VhlStrength(arr, input_unit="N")
    vhl_str = str(vhl)
    expected_str = (
        "V: [100. 200.] daN\nH: [300. 400.] daN\nL: [500. 600.] daN\n"
    )
    assert vhl_str == expected_str
    vhl_repr = repr(vhl)
    expected_repr = f"VhlStrength\n{expected_str}"
    assert vhl_repr == expected_repr

    options.output_units.force = save_option


def test_quantity_array_creation():
    arr = np.array([1, 2, 3, 4, 5])
    expected_unit = "m"
    quantity_arr = QuantityArray(arr, output_unit="m", input_unit="mm")
    assert quantity_arr.unit == "meter"
    assert quantity_arr.symbol == expected_unit

    out_arr, out_unit = quantity_arr.to_tuple()
    np.testing.assert_array_equal(out_arr, quantity_arr.value)
    np.testing.assert_array_equal(quantity_arr.value, arr / 1000)
    assert out_unit == expected_unit
