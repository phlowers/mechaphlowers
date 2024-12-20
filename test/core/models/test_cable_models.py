import numpy as np

from mechaphlowers.core.models.cable_models import CatenaryCableModel


def test_catenary_cable_model() -> None:
    cable_model = CatenaryCableModel(501.3, -23.2, 2_112.2)
    x = np.linspace(-223.2, 245.2, 250)

    assert isinstance(cable_model.z(x), np.ndarray)

    assert isinstance(cable_model.x_m(), float)

    assert isinstance(cable_model.x_n(), float)


def test_catenary_cable_model__x_m__if_no_elevation_difference() -> None:
    a = 100
    b = 0
    p = 2_000

    cable_model = CatenaryCableModel(a, b, p)
    assert abs(cable_model.x_m() + 50.) < 0.01
