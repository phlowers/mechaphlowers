# Data catalog

## Adding new catalogs

You can put new catalogs in .csv format in folder `src/mechaphlowers/data/`

In order to configure how mechaphlowers will read them, you will need to join a .yaml file that refers the csv catalog.

Then, to instantiate this new catalog in the code, you can define it in `src/mechaphlowers/data/catalog.py` using function `build_catalog_from_yaml()`

```python
new_catalog = build_catalog_from_yaml("new_catalog_config_file.yaml")
```

## yaml file format

Key column name is always considered as a string.

Type checking is made before renaming, so "columns" contains the original columns names


## 