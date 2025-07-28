# Data catalog

## Adding new catalogs

You can add new catalogs in `.csv` format to the folder `src/mechaphlowers/data/`.

To configure how Mechaphlowers will read them, you need to provide a corresponding `.yaml` file that references the CSV catalog, also placed in the `src/mechaphlowers/data/` folder.

To instantiate this new catalog in the code, define it in `src/mechaphlowers/data/catalog.py` using the function `build_catalog_from_yaml()`:

```python
new_catalog = build_catalog_from_yaml("new_catalog_config_file.yaml")
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

columns_renaming:
  - Type 1: First Type
  - Type 2: Second Type
```

- `csv_name`: the name of the csv located in `src/mechaphlowers/data/`
- `catalog_type`: used for extracting catalog into mechaphlowers object. Types currently allowed: 'default_catalog', 'cable_catalog'
- `key_column_name`: name of the column considered as key index 
- `columns`: list of all columns with their type
- `columns_renaming`: a dictionary that maps original column names to their new names using the format `original_name: new_name`. Type checking is performed before renaming, so "columns" should list the original column names.


!!! note "Key column"
  You don't have to put the key column in the field "columns"; it is always considered as a string.

!!! warning "Booleans"
	To avoid issues when empty value in boolean columns, booleans columns are not type checked.
