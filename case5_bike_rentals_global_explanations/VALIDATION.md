# Case 5 validation record

Completed 4 October 2026.

## Execution

- Notebook format: v4, validated with nbformat 5.11.1.
- 47 cells total, including 20 code cells. Every code cell executed in order in a fresh IPython process.
- Execution counts are consecutive from 1 to 20. No cell errors or stderr warnings were captured.
- Actual HTML/table/Markdown/image outputs were captured into the notebook. All nine PNG figure outputs are embedded. The HTML preview was exported from that executed notebook.
- The host prohibits kernel sockets; attempts to launch a separate TCP/IPC Jupyter kernel failed before code execution. The completed calculation used a fresh IPython session with the inline matplotlib backend. VS Code/Jupyter GUI startup was not independently tested. This environment limitation did not prevent execution of any notebook cell.

## Meaningful checks

- Raw hourly CSV hash matches the preserved UCI data: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.
- Timestamps are ordered and unique; rental components sum to the total and are excluded from predictors.
- Training/development/test partition boundaries and reference-sample indices are explicit.
- Development Random Forest MAE and grouped permutation results match the earlier feasibility results within numerical tolerance.
- PDP is the average of the computed ICE matrix; centered ICE is zero at its reference hour.
- First-order ALE recovers a known centered component of a separate additive linear function to floating-point precision. All resulting bins are populated; actual versus requested bin counts are displayed.
- The support-distance cutoff results reproduce the earlier sensitivity study at every tested radius. The notebook calculates them independently of the saved research tables.
- The configuration is written before final-period predictions. The same fitted model/baselines are scored without post-test tuning.
- All nine figures were visually reviewed together for readable labels, clipping, and consistency with their interpretations.
- The package includes the dataset, original source README, setup requirements, executed notebook, HTML preview, TA guide, and computed results. Core notebook calculations do not load any precomputed research result.

## Final predictive results

| Predictor | Final MAE | Final RMSE | Final R² |
|---|---:|---:|---:|
| Training mean | 156.565 | 208.056 | −0.065 |
| Hour/working-day mean | 88.543 | 129.202 | 0.589 |
| Random Forest | 67.291 | 99.562 | 0.756 |

The final period is October–December 2012, with 2,168 rows. These metrics evaluate recorded-condition prediction in that period, not prospective forecast availability, causal effects, or explanation correctness. All final-period outputs are now observed; later experimentation requires an appropriate new evaluation design.

## Files recording execution

- `outputs/execution_log.json`: executed cell indices, counts, runtimes, and error status.
- `outputs/analysis_config.json`: fixed model, split, reference indices, versions, and data hash.
- `outputs/notebook_results.json`: computed development/final metrics and explanation checks.
- `outputs/final_test_predictions.csv`: actual outcomes and fixed-model/baseline predictions.
