# Case 5: The Average Day Doesn't Exist

A complete Explainable Data Science teaching case on permutation feature importance (PFI), partial dependence (PDP), individual conditional expectation (ICE), and accumulated local effects (ALE).

## Open it

- **Read immediately:** open `05_bike_rentals_global_explanations.html` in a browser. The executed figures are embedded; a Python installation is not needed to read it.
- **Run and edit:** open `05_bike_rentals_global_explanations.ipynb` in VS Code with the Python and Jupyter extensions, or in JupyterLab. Keep the extracted folder intact, especially `data/hour.csv`.
- **Teach:** see `TA_Notes.md` for the suggested pacing, interpretation guide, short answer key, and assessment rubric.

The notebook computes the explanations in its cells. It does not depend on the earlier research package or previously saved result files. The data is included. Once dependencies are installed, the calculations run offline.

## Local setup

Tested computational environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, matplotlib 3.10.8. Use Python 3.12 and the included `requirements.txt` for the closest reproduction. No GPU is needed.

From a terminal in the extracted `case5_teaching` folder:

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name eds-case5 --display-name "EDS Case 5 (Python 3.12)"
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name eds-case5 --display-name "EDS Case 5 (Python 3.12)"
```

In VS Code, select the **EDS Case 5 (Python 3.12)** kernel, then **Restart Kernel and Run All**. If the data path check fails, open the extracted case folder as your workspace and use that folder as the notebook working directory.

For a standalone JupyterLab interface, activate this environment, install `jupyterlab`, and launch it from this folder:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

If using an existing EDS environment, compare its package versions before installing the pinned requirements. The commands above create an isolated environment for this case. Do not run installation cells inside the teaching notebook; setup is separate from the lesson.

## What runs

The notebook contains 47 cells, including 20 code cells and nine figures. It fits a fixed Random Forest, recomputes all core explanations and the support-cutoff check, evaluates the final period, and writes fresh tables/figures to `outputs/`. Outputs with the same names are overwritten on rerun. Preserve a separate copy if you change settings for an experiment.

Execution time is machine-dependent; this host's measured in-notebook run was approximately 12 seconds after imports/setup, excluding package installation. Allow additional time on a laptop. The model uses two CPU workers.

For conventional headless execution after setup:

```bash
python -m jupyter nbconvert --to notebook --execute 05_bike_rentals_global_explanations.ipynb --output 05_bike_rentals_global_explanations_rerun.ipynb --ExecutePreprocessor.timeout=600
```

## Scope and final evaluation

One row is a system-wide hourly observation. The target is recorded total rentals; recorded weather is used. This is an offline model/explanation audit, not demonstrated prospective forecasting or a station-allocation tool.

The model trains on January 2011–June 2012. Explanation development uses July–September 2012. October–December 2012 was scored after the analysis design was fixed. The completed notebook includes those final results. Rerunning it reproduces an already-observed test; it does not create a new holdout.

| Model | Development MAE | Final-period MAE |
|---|---:|---:|
| Training mean | 197.83 | 156.57 |
| Hour/working-day mean | 125.54 | 88.54 |
| Random Forest | 74.44 | 67.29 |

MAE units are rentals/hour. Final model R² is 0.756 (development 0.805). A smaller absolute error alongside a lower R² is possible because the outcome distribution differs across periods. These scores do not validate causal claims or explanation correctness.

## Data provenance

Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894. Licensed CC BY 4.0. The raw hourly data and archive README are included and unmodified. Source archive: https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip.

The CSV has 17,379 rows. The archive README and repository webpage disagree on some metadata, including temperature scaling and season labels. This notebook retains stored normalized temperature values, omits season, and does not fill unrepresented timestamps with zero rentals. The original README is preserved as `data/UCI_Readme.txt` so these differences remain visible.

## Validation

All code cells were executed sequentially in a fresh IPython process with real rich outputs captured, and the notebook schema was validated. Prior development metrics and support-cutoff results were reproduced; the additive-function ALE check passed; all nine figures were visually reviewed. The host does not permit the sockets needed to launch a separate Jupyter kernel, so local VS Code/Jupyter GUI startup was not tested here. See `VALIDATION.md` for the exact checks. No unexecuted or fabricated result is presented as an execution result.
