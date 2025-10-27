# How to publish a new version of the package ?

## Last tests
- checkout main and run `make all`

## Local tests

- update the lock file

```bash
uv lock --upgrade
```

- test locally

```bash
# tag with the version you want
git tag v0.3.0b0
# push tag
git push --tags
# delete tag with the version you want
git tag -d v0.3.0b0

# or use make command
make version=v0.3.0b0 build-local
```

!!! note
    Don't forget to create a release/vX.Y.Z/from-main branch for the last changes

- install and test

```bash
uv venv .venv_release
source .venv_release/bin/activate
uv pip install dist/mechaphlowers-0.3.0b0-py3-none-any.whl
```

## Push to github

- merge your release branch
```bash
# tag with the version you want
git tag v0.3.0b0
# push tag
git push --tags
```
- On github run Action 🚀 Build upload on test-pypi
- If success you can replay the tests on the wheel delpoyed
- make a release on github

