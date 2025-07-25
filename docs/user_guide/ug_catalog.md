# Data catalog

## Adding new catalogs

You can put new catalogs in .csv format in folder `src/mechaphlowers/data/`

In order to configure how mechaphlowers will read them, you will need to join a .yaml file that refers the csv catalog, also in folder `src/mechaphlowers/data/`.

Then, to instantiate this new catalog in the code, you can define it in `src/mechaphlowers/data/catalog.py` using function `build_catalog_from_yaml()`

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
- `columns_renaming`: dictionnary to rename columns `original_name: new_name`. Type checking is made before renaming, so "columns" contains the original columns names.


!!! note "Key column"
	You don't have to put the key column in field "column", it is always considered as a string.

!!! warning "Booleans"
	To avoid issues when empty value in boolean columns, booleans columns are not validated.

## 