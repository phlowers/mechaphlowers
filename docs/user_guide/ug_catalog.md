# Data catalog

## Catalogs available

Mechaphlowers comes with a few catalogs that you can use to instantiate entities.  
You can find the following catalogs with sample data inside:
- `sample_cable_catalog`
- `sample_support_catalog`

```python
from mechaphlowers.data.catalog import sample_support_catalog

# you can see the keys in the existing catalog
sample_support_catalog.keys()

# or see object directly
sample_support_catalog

# get data as the appropriate mechaphlowers object
sample_support_catalog.get_as_object(["support_1", "support_5"])

```



## Loading your data in the catalogs

You can add new catalogs in `.csv` format to the a user folder.

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


## yaml file format

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

## Augmenting your catalog

For developers who wants to directly instanciate objects from custom catalogs, there are two ways: 
- use the existing catalogs and add your own data. You will take advantage of the existing object facilities with the `get_as_object()` method. For example support catalog will provide a list of `SupportShape` objects.
- Implement in a pull request your own object to handle a new type of data.

## Unit conversion

You can specify the unit of the data in the csv. This way, they will be automatically converted into SI units for computations.

This is done using the python package [pint](https://pint.readthedocs.io/en/stable/){:target="_blank"}. The syntax for units covers all usual notations.

For example, every following notation works: `m/s^2`, `m/s**2` `m*s^(-2)`, `meter.seconds^(-2)`


