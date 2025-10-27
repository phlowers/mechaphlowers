# How to publish a new version of the package ?

## Last tests
- checkout main and run make all

## Local tests

- update the lock file

uv lock --upgrade





```bash
# tag with the version you want
git tag v0.3.0b0
# push tag
git push --tags
```


## Push to github
```bash
# tag with the version you want
git tag v0.3.0b0
# push tag
git push --tags
```