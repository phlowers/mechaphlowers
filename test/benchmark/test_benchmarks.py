"""this module contains tests for functional benchmark"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.catalog.catalog import build_catalog_from_yaml
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import SectionArray
from test.utils import generate_html_from_json

try:
    projet_dir: Path = Path(__file__).resolve().parents[2]

    EXTERNAL_DATA_DIR = projet_dir / "input_data"
    EXTERNAL_DATA_DIR.mkdir(exist_ok=True)

    test_output_dir = Path(projet_dir / "output_test")
    test_output_dir.mkdir(exist_ok=True)

    # objects for benchmark tests
    cable_bench = build_catalog_from_yaml(
        "sample_cable_database.yaml",
        user_filepath=EXTERNAL_DATA_DIR,
        separator=";",
        decimal=",",
    )
    cable_bench.clean_catalog()


except Exception as e:
    warnings.warn(f"Error during benchmark setup: {str(e)}")
    warnings.warn(
        "you should add the path input_data with your data and the yaml"
    )
    from mechaphlowers.data.catalog import sample_cable_catalog as cable_bench

cable_list = cable_bench.keys()


def test_import_csv():
    build_catalog_from_yaml(
        "sample_cable_database.yaml",
        user_filepath=EXTERNAL_DATA_DIR,
        separator=";",
        decimal=",",
    )


@pytest.fixture(scope="session")
def benchmark_report():
    """Fixture to collect and report functional benchmark results."""
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
    }

    yield report_data

    # Write report to JSON after all tests
    report_path = test_output_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n✓ Benchmark report saved to {report_path}")
    generate_html_from_json(report_path)


@pytest.fixture
def record_benchmark(benchmark_report, request):
    """Fixture to record individual benchmark results.

    Usage in tests:
        record_benchmark("name", {"key": "value"})
    The fixture will update recorded entries' status after the test finishes
    (using pytest's test report) so the final JSON contains pass/fail info.
    """

    def _record(test_name: str, result: dict):
        entry = {**result, "name": test_name, "status": "pending"}
        idx = len(benchmark_report["tests"])
        benchmark_report["tests"].append(entry)
        # Store the index so the hook knows which entry to update
        request.node._benchmark_entry_idx = idx

    return _record


@pytest.fixture
def section_array_angles() -> SectionArray:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": convert_weight_to_mass(
                    [1000, 500, 500, 1000]
                ),
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return section_array


# @pytest.mark.xfail(reason="Functional benchmark tests - not a real unit test")
# @pytest.mark.benchmark
@pytest.mark.parametrize("cable_name", cable_list)
def test_element_sandbox(
    section_array_angles: SectionArray, cable_name: str, record_benchmark
):
    if not cable_name:
        raise ValueError(
            "Cable name is empty / check your folder configuration"
        )
    report_content = {}

    try:
        cable_array = cable_bench.get_as_object([cable_name])
        balance_engine = BalanceEngine(
            cable_array=cable_array,
            section_array=section_array_angles,
        )
        balance_engine.solve_adjustment()

        # section.sagging_temperature = 30
        # section.cable_loads.ice_thickness = np.array([1,1,1,1]) * 1e-2
        balance_engine.balance_model.cable_loads.wind_pressure = np.array(
            [200] * 4
        )
        balance_engine.solve_change_state()

    except Exception as e:
        report_content["error"] = str(e)
        assert False

    finally:
        record_benchmark(
            "test_element_sandbox :: " + str(cable_name), report_content
        )
