# Notebooks

Phase 4 (v0.5.0) starter notebooks for new users. Each notebook is
written in plain JSON (nbformat v4) and can be opened in Jupyter
Lab/Notebook or VS Code.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [quick_start](01_quick_start.ipynb) | List experiments, run exploration, inspect output |
| 02 | [custom_experiment](02_custom_experiment.ipynb) | Register a new strategy end-to-end |
| 03 | [interpreting_results](03_interpreting_results.ipynb) | SPA / PSR / FDR / risk-allocation deep-dive |

## Regenerating

Each notebook has a corresponding `.py` source file with a
`make_cell` / `make_md` / `make_code` helper and a final
`json.dump`. Regenerate any notebook by running:

```bash
python notebooks/01_quick_start.py
python notebooks/02_custom_experiment.py
python notebooks/03_interpreting_results.py
```

This avoids the need for `nbformat` as a build-time dependency.

## Running

```bash
pip install jupyterlab  # or use VS Code Jupyter extension
jupyter lab notebooks/
```
