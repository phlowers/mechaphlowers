from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.data.catalog.catalog import sample_cable_catalog


@pytest.fixture
def study_4span_with_load() -> SectionStudy:
    cable_array = sample_cable_catalog.get_as_object(["ASTER600"])
    section_array = SectionArray(
        pd.DataFrame({
            "name": ["1", "2", "3", "4", "5"],
            "suspension": [False, True, True, True, False],
            "conductor_attachment_altitude": [50.0, 50.0, 50.0, 50.0, 50.0],
            "crossarm_length": [0.0, 5.0, 5.0, 5.0, 0.0],
            "line_angle": [0.0, 0.0, 0.0, 0.0, 0.0],
            "insulator_length": [3.0, 3.0, 3.0, 3.0, 3.0],
            "span_length": [400.0, 400.0, 400.0, 400.0, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 500.0, 1000.0],
            "load_mass": [0.0, 0.0, 0.0, 0.0, 0.0],
            "load_position": [0.0, 0.0, 0.0, 0.0, 0.0],
        }),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    study = SectionStudy(cable_array=cable_array, section_array=section_array)
    study.set_loads(
        load_position_distance=[200.0, 0.0, 0.0, 0.0],
        load_mass=[500.0, 0.0, 0.0, 0.0],
    )
    return study


@pytest.mark.integration
def test_repeated_solve_idempotent_loads(study_4span_with_load: SectionStudy) -> None:
    study_4span_with_load.solve_adjustment()
    study_4span_with_load.solve_change_state(new_temperature=15)
    data_first = study_4span_with_load.get_data_spans()
    L0_first = np.array(data_first["L0"])
    displacement_first = np.array(data_first["horizontal_distance"])
    print("\n")
    print(L0_first)
    print("------------")
    for i in range(4):
        # study_4span_with_load.solve_adjustment()
        study_4span_with_load.solve_change_state(new_temperature=60)
        data = study_4span_with_load.get_data_spans()
        print("L0: ", data["L0"])
        # np.testing.assert_allclose(data["L0"], L0_first,
        #     err_msg=f"L0 drifted at iteration {i + 2}")
        # np.testing.assert_allclose(data["horizontal_distance"], displacement_first,
        #     err_msg=f"horizontal_distance drifted at iteration {i + 2}")




@pytest.mark.integration
def test_repeated_solve_idempotent_no_loads(study_4span_no_load: SectionStudy) -> None:
    study_4span_no_load.solve_adjustment()
    initial_balance_engine = deepcopy(study_4span_no_load.balance_engine)
    study_4span_no_load.solve_change_state(new_temperature=15)
    data_first = study_4span_no_load.get_data_spans()
    L0_first = np.array(data_first["L0"])
    displacement_first = np.array(data_first["horizontal_distance"])
    print("\n")
    print(L0_first)
    print("------------")
    for i in range(4):
        study_4span_no_load.solve_adjustment()
        current_balance_engine = study_4span_no_load.balance_engine
        # data = study_4span_no_load.get_data_spans()
        # print("L0 adjustment : ", data["L0"])
        study_4span_no_load.solve_change_state(ice_thickness=0.1)
        data = study_4span_no_load.get_data_spans()
        print("L0: ", data["L0"])
        # study_4span_no_load.solve_change_state(ice_thickness=0)
        # data = study_4span_no_load.get_data_spans()
        # print("L0: ", data["L0"])
        # np.testing.assert_allclose(data["L0"], L0_first,
        #     err_msg=f"L0 drifted at iteration {i + 2}")
        # np.testing.assert_allclose(data["horizontal_distance"], displacement_first,
        #     err_msg=f"horizontal_distance drifted at iteration {i + 2}")



@pytest.mark.integration
def test_L0_not_same(study_4span_no_load: SectionStudy) -> None:
    study_4span_no_load.solve_adjustment()
    initial_balance_engine = deepcopy(study_4span_no_load.balance_engine)
    study_4span_no_load.solve_change_state(new_temperature=15)
    data_first = study_4span_no_load.get_data_spans()
    L0_first = np.array(data_first["L0"])
    displacement_first = np.array(data_first["horizontal_distance"])
    print("\n")
    print(L0_first)
    print("------------")
    study_4span_no_load.solve_adjustment()
    current_balance_engine = deepcopy(study_4span_no_load.balance_engine)
    study_4span_no_load.solve_change_state(new_temperature=60)
    data = study_4span_no_load.get_data_spans()
    print("L0: ", data["L0"])
    
    initial_dict=initial_balance_engine.balance_model.__dict__
    current_dict=current_balance_engine.balance_model.__dict__
    for (key, value) in initial_dict.items():
        if isinstance(value, np.ndarray):
            if (value != current_dict[key]).all():
                print("OOOOOOOOOOOOOOOOO")
                print(key)
                print("initial", initial_dict[key])
                print("current", current_dict[key])
        else:
            try:
                if value != current_dict[key]:
                    print("OOOOOOOOOOOOOOOOO")
                    print(key)
                    # print("initial", initial_dict[key])
                    # print("current", current_dict[key])
            except Exception as e:
                print(e)




@pytest.fixture
def study_4span_no_load() -> SectionStudy:
    cable_array = sample_cable_catalog.get_as_object(["ASTER600"])
    section_array = SectionArray(
        pd.DataFrame({
            "name": ["1", "2", "3", "4", "5"],
            "suspension": [False, True, True, True, False],
            "conductor_attachment_altitude": [50.0, 50.0, 50.0, 50.0, 50.0],
            "crossarm_length": [0.0, 5.0, 5.0, 5.0, 0.0],
            "line_angle": [0.0, 0.0, 0.0, 0.0, 0.0],
            "insulator_length": [3.0, 3.0, 3.0, 3.0, 3.0],
            "span_length": [400.0, 400.0, 400.0, 400.0, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 500.0, 1000.0],
            "load_mass": [0.0, 0.0, 0.0, 0.0, 0.0],
            "load_position": [0.0, 0.0, 0.0, 0.0, 0.0],
        }),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    study = SectionStudy(cable_array=cable_array, section_array=section_array)
    return study

def test_sandbox(study_4span_no_load: SectionStudy):
    study_4span_no_load.solve_adjustment()
    print("\n")
    data = study_4span_no_load.get_data_spans()
    print("L0: ", data["L0"])
    study_4span_no_load.set_loads(
        load_position_distance=[200.0, 0.0, 0.0, 0.0],
        load_mass=[500.0, 0.0, 0.0, 0.0],
    )
    print("---Add loads 500-----")
    # study_4span_no_load.solve_adjustment()
    data = study_4span_no_load.get_data_spans()
    print("L0: ", data["L0"])
    study_4span_no_load.solve_change_state(new_temperature=15)

    # Second cycle: 2 loaded spans — stale cache expects 1 slot, crashes in build_merged
    study_4span_no_load.set_loads(
        load_position_distance=[300.0, 0.0, 0.0, 0.0],
        load_mass=[1500.0, 0.0, 0.0, 0.0],
    )
    print("---Add loads 1000-----")

    # study_4span_no_load.solve_adjustment()
    data = study_4span_no_load.get_data_spans()
    print("L0: ", data["L0"])
    study_4span_no_load.solve_change_state(new_temperature=15)



@pytest.mark.integration
def test_sandbox_1(study_4span_no_load: SectionStudy) -> None:
    study_4span_no_load.solve_adjustment()
    study_4span_no_load.solve_change_state(new_temperature=15)
    print("\n")
    data = study_4span_no_load.get_data_spans()
    print("L0 adjustment : ", data["L0"])    
    print("------------")
    
    # study_4span_no_load._section_array.sagging_parameter=2000
    # study_4span_no_load._section_array.sagging_temperature=15
    
    # study_4span_no_load.balance_engine.deformation_model_type.temp_ref=15
    # study_4span_no_load.balance_engine.deformation_model_type.current_temperature=15
    # study_4span_no_load.balance_engine.balance_model.sagging_temperature=15
    study_4span_no_load.solve_adjustment()
    data = study_4span_no_load.get_data_spans()
    print("L0 adjustment : ", data["L0"])
    study_4span_no_load.solve_change_state(ice_thickness=0.1)
    
    # study_4span_no_load._section_array.sagging_parameter=2000
    # study_4span_no_load._section_array.sagging_temperature=15
    # zeros_vector = np.zeros(5)
    # study_4span_no_load.balance_engine.cable_loads.ice_thickness = np.zeros(zeros_vector)
    # study_4span_no_load.balance_engine.cable_loads.wind_pressure = np.zeros(zeros_vector)
    
    # study_4span_no_load.solve_change_state()
    study_4span_no_load.solve_adjustment()
    data = study_4span_no_load.get_data_spans()
    print("L0 adjustment : ", data["L0"])
    study_4span_no_load.solve_change_state(ice_thickness=0.1)




@pytest.mark.integration
def test_set_loads_change_number_of_loaded_spans_raises(
    study_4span_no_load: SectionStudy,
) -> None:
    # First cycle: 1 loaded span — _precompute_merge_indices cached for 1 slot
    study_4span_no_load.set_loads(
        load_position_distance=[200.0, 0.0, 0.0, 0.0],
        load_mass=[500.0, 0.0, 0.0, 0.0],
    )
    study_4span_no_load.solve_adjustment()
    study_4span_no_load.solve_change_state(new_temperature=40)

    # Second cycle: 2 loaded spans — stale cache expects 1 slot, crashes in build_merged
    study_4span_no_load.set_loads(
        load_position_distance=[200.0, 200.0, 0.0, 0.0],
        load_mass=[500.0, 300.0, 0.0, 0.0],
    )
    # study_4span_no_load.solve_adjustment()
    study_4span_no_load.solve_change_state(new_temperature=40)
    