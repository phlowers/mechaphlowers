import numpy as np

from mechaphlowers.config import options
from mechaphlowers.core.models.balance.interfaces import VhlStrength


def test_vhl_strength():
    save_option = options.units.force
    options.units.force = "daN"

    arr = np.array(
        [[1000, 2000, 3000], [4000, 5000, 6000], [7000, 8000, 9000]]
    )
    vhl = VhlStrength(arr, input_unit="N")

    V, H, L = vhl.vhl
    np.testing.assert_array_equal(V.array, np.array([1000, 2000, 3000]) / 10)
    np.testing.assert_array_equal(H.array, np.array([4000, 5000, 6000]) / 10)
    np.testing.assert_array_equal(L.array, np.array([7000, 8000, 9000]) / 10)

    np.testing.assert_array_equal(V.array, vhl.V.array)
    np.testing.assert_array_equal(H.array, vhl.H.array)
    np.testing.assert_array_equal(L.array, vhl.L.array)

    options.units.force = save_option


def test_vhl_strength_str_repr():
    save_option = options.units.force
    options.units.force = "daN"

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

    options.units.force = save_option
