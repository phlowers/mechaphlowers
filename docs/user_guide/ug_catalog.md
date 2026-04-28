# Data catalog

## How to use the catalogs

### Catalogs available

Mechaphlowers comes with a few catalogs that you can use to instantiate entities.  
You can find the following catalogs with sample data inside:

- `sample_cable_catalog`
- `sample_support_catalog`
- `section_factory_sample_data`


### Get a support as an object

```python
from mechaphlowers.data.catalog import sample_support_catalog

# you can see the keys in the existing catalog
sample_support_catalog.keys()

# or see object directly
sample_support_catalog

# get data as the appropriate mechaphlowers object
sample_support_catalog.get_as_object(["support_1", "support_5"])

```

### sample_cable_catalog

Four fictive cables representative of common overhead line conductor families:

```python
from mechaphlowers.data.catalog import sample_cable_catalog

sample_cable_catalog.keys()
cable_array = sample_cable_catalog.get_as_object(["ASTER600"])
```

| Name | Section (mm²) | Diameter (mm) | Linear mass (kg/m) | Notes |
|------|--------------|---------------|--------------------|-------|
| ASTER600 | 600.4 | 31.86 | 1.8 | Linear stress-strain model |
| CROCUS400 | 400.9 | 26.16 | 1.6 | Linear stress-strain model |
| NARCISSE600G | 600.4 | 29.56 | 2.2 | Polynomial stress-strain model |
| PETUNIA600 | 599.7 | 31.66 | 2.3 | Linear model, magnetic heart |

Columns and their input units (before SI conversion):

| Column (CSV → internal) | Input unit | Description |
|-------------------------|-----------|-------------|
| `section` | mm² | Total cross-section |
| `diameter` | mm | External diameter |
| `young_modulus` | MPa | Young's modulus |
| `linear_mass` | kg/m | Linear mass |
| `dilatation_coefficient` | 1/K | Thermal expansion coefficient |
| `temperature_reference` | °C | Reference temperature |
| `stress_strain_a0`…`a4` → `a0`…`a4` | MPa | Conductor polynomial stress-strain coefficients |
| `stress_strain_b0`…`b4` → `b0`…`b4` | MPa | Heart polynomial stress-strain coefficients |
| `diameter_heart` | mm | Heart diameter |
| `section_conductor` | mm² | Conductor cross-section |
| `section_heart` | mm² | Heart cross-section |
| `solar_absorption` | — | Solar absorption coefficient |
| `emissivity` | — | Emissivity coefficient |
| `electric_resistance_20` | Ω/km | Electric resistance at 20 °C |
| `linear_resistance_temperature_coef` | 1/K | Temperature coefficient of resistance |
| `is_polynomial` | — | `True` if polynomial stress-strain model is used |
| `radial_thermal_conductivity` | W·m⁻¹·K⁻¹ | Radial thermal conductivity |
| `has_magnetic_heart` | — | `True` if the heart is magnetic (e.g. steel) |
| `rts_cable` | N | Rated Tensile Strength of the intact cable |
| `rts_layer_1`…`rts_layer_8` | N | Unit RTS per strand per layer (`0` = unused layer) |
| `safety_coefficient` | — | Safety coefficient for utilization rate (default 1.5) |
| `nb_strand_layer_1`…`nb_strand_layer_8` | — | Number of strands per layer (`0` = unused layer) |

!!! note "Stress-strain model"
    For linear cables (`is_polynomial = False`), only `a1` and `b1` are non-zero (conductor and heart Young's modulus). Polynomial cables use all five coefficients `a0`…`a4` / `b0`…`b4`.

!!! note "RRTS columns"
    `rts_cable`, `rts_layer_*`, `safety_coefficient` and `nb_strand_layer_*` are optional. They are only required for [RRTS and utilization rate calculations](ug_rrts.md).
### Build a balance engine from scratch

```python
# Get section factory data as a dataframe
from mechaphlowers.data.catalog import section_factory_sample_data

data = section_factory_sample_data(size_section = 5, seed = 42)

# Get cable array object from the cable catalog
from mechaphlowers.data.catalog import sample_cable_catalog

cable_array = sample_cable_catalog.get_as_object(["ASTER600"])

# Build a balance engine with the section array and the cable array
import mechaphlowers as mph
import pandas as pd

section_array = mph.SectionArray(pd.DataFrame(data))
balance_engine = mph.BalanceEngine(section_array=section_array, cable_array=cable_array)
# ... do some computations with balance_engine
```

## Custom catalogs

### Loading your data in the catalogs

You can add new catalogs in `.csv` format to a user folder.

To configure how Mechaphlowers will read them, you need to provide a corresponding `.yaml` file that references the CSV catalog, also placed in the same folder.

To get and customize the existing catalogs, use the `write_yaml_catalog_template()` function.

To instantiate this new catalog in the code, define it in the folder of your choice and load it using the function `build_catalog_from_yaml()`:

```python
write_yaml_catalog_template("my/path/where/I/want/to/write","support_catalog")

# ...modify the yaml for your data
# and then load it !

file_path_yaml = Pathlib("my/path/where/my/yaml/and/csv/file/are/located")

my_catalog = build_catalog_from_yaml("new_catalog_config_file.yaml", user_filepath=filepath)
```


### yaml file format

Here is an example of yaml file:

```yaml
csv_name: "pokemon.csv"
catalog_type: default_catalog
key_column_name: Name

columns:
  - Type 1: str
  - Type 2: str
  - Total: int
  - HP: int
  - Attack: int
  - Defense: int
  - Sp. Atk: int
  - Sp. Def: int
  - Speed: int
  - Generation: int
  - Legendary: bool
  - Size: float

columns_renaming:
  - Type 1: First Type
  - Type 2: Second Type
  - Size: Height

columns_units:
  - Height: m
```

- `csv_name`: the name of the csv located in `src/mechaphlowers/data/`
- `catalog_type`: used for extracting catalog into mechaphlowers object. Types currently allowed: 'default_catalog', 'cable_catalog'
- `key_column_name`: name of the column considered as key index 
- `columns`: list of all columns with their type
- `columns_renaming`: a optional dictionary that maps original column names to their new names using the format `original_name: new_name`. Type checking is performed before renaming, so "columns" should list the original column names.
- `columns_units`: an optional dictionary to store the units of the columns. If left empty for a column, there is usually a default unit if necessary ([see arrays.py](../docstring/entities/arrays.md)). The name of the columns should after the renamed columns, if concerned by `columns_renaming`


!!! note "Key column"
    You don't have to put the key column in the field "columns"; it is always considered as a string.

!!! warning "Booleans"
    To avoid issues when empty value in boolean columns, booleans columns are not type checked.

### Augmenting your catalog

For developers who wants to directly instantiate objects from custom catalogs, there are two ways: 
- use the existing catalogs and add your own data. You will take advantage of the existing object facilities with the `get_as_object()` method. For example support catalog will provide a list of `SupportShape` objects.
- Implement in a pull request your own object to handle a new type of data.

### Unit conversion

You can specify the unit of the data in the csv. This way, they will be automatically converted into SI units for computations.

This is done using the python package [pint](https://pint.readthedocs.io/en/stable/){:target="_blank"}. The syntax for units covers all usual notations.

For example, every following notation works: `m/s^2`, `m/s**2` `m*s^(-2)`, `meter.seconds^(-2)`


