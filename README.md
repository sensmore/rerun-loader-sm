# Rerun Sensmore Data Loader
This data loader allows to display the files used by sensmore in rerun. It also allows to just doubleclick 
or open the file in the file browser/file explorer/Finder or Finder to load the folder in rerun.

## Installtion of Rerun

See https://rerun.io/docs/getting-started/installing-viewer

## Installation or Usage of Sensmore Data Loader

```bash
# TODO: maybe do not install complete python but use uv run ...
uv tool install .
# uninstall with
uv tool uninstall rerun-loader-sm
```

After that check that the following works

```bash
rerun-loader-sm
```

## Installation for Development

```bash
# Install with
uv tool install -e .
# After changes in TOML or large changes in packgese run
uv tool install --reinstall -e . 
```
After that you can run in globally, but still modify it by editing the files int this folder.