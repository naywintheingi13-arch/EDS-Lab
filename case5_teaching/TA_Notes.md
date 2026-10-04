# Case 5: TA notes and discussion guide

## Purpose

Teach students to move from a feature ranking to feature-response curves, then inspect what those curves average over and what synthetic inputs they query. The lesson should remain one investigation led by an operations manager's questions.

**Core sequence:** task and baseline → PFI → hour PDP → ICE/centered ICE → temperature dependence and support → ALE → interpretation report.

**Compact extensions:** the constant-year PFI puzzle, grouped temperature permutation, and cutoff sensitivity. Keep the main discussion centered on what each method represents rather than every implementation setting.

Suggested use: two 50–60 minute sessions, or a shorter guided demonstration with implementation details as preparation. This is a pacing suggestion, not a measured completion time. Students need Python, regression, train/test splits, and basic evaluation knowledge. No specialized ALE package is required.

## Before teaching

Run setup and restart/run all locally. Open the HTML export as a read-only backup. Decide whether students pause at prompts or work in pairs. Keep the executed output visible only after students have made a prediction. The TA notes contain answer anchors and should be distributed separately if desired.

## Section-by-section answer anchors

1. **Prediction contract.** `casual` and `registered` sum exactly to the outcome, so including them would make the task tautological. Recorded weather does not establish weather availability in advance. The data is system-wide, not station-specific.
2. **Baselines.** The hour/working-day baseline is a stronger comparison than the unconditional mean because it captures a relevant calendar pattern. Development model MAE is 74.44 versus 125.54. MAE is in rentals/hour, not percent accuracy.
3. **PFI.** A score measures performance reliance under the permutation protocol. It gives no direction or response shape. Shuffling can generate inconsistent inputs. Shuffle SD is conditional on the fixed data and model; it excludes model-fit and population uncertainty.
4. **Year puzzle.** Every development row has year code 1. Permuting a constant cannot change a prediction, even if the fitted model uses that feature elsewhere. Do not substitute “the model ignores year” for this explanation.
5. **PDP.** Substituting hour into all reference rows and averaging predictions differs from averaging observed rentals grouped by original hour. A PDP does not require outcome labels once the model is fitted.
6. **ICE.** Averaging hides variability of predicted responses. Centering removes baseline level differences but retains response-shape variation. An ICE row represents one altered hourly record; it is neither a rider nor an observed longitudinal day. Group differences mix model response differences with differences in reference-feature composition.
7. **Support.** Temperature and perceived temperature correlate around 0.992 in training. Individually ordinary numbers can produce atypical pairs. “Far from observed pairs under a chosen rule” is not the same as “physically impossible” or “wrong prediction.”
8. **ALE.** Local endpoint prediction differences are averaged within bins, accumulated, interpolated, and empirically centered. Zero is a centering reference, not zero rentals. Requested bins collapse at repeated edges: 10, 15, and 20 requested produce 10, 14, and 16 actual bins in this sample. The separate additive-function check verifies the implementation, not the real model.
9. **Similar curves.** PDP/ALE broadly agree in shape here. That does not imply identical estimands, sampling assumptions, or query support. Agreement is informative but is not proof of validity.
10. **Cutoff sensitivity.** At half/original/double radius, PDP flag percentages are 82.43/72.19/57.62; ALE 18.25/9.50/3.63. The ordering survives these changes. The test does not establish robustness to other distance metrics, features, periods, or models.
11. **Grouped permutation.** Jointly shuffling temperature and perceived temperature preserves their internal pairing but breaks associations with remaining inputs. Group PFI is 15.14, compared with 0.88 and 11.89 individually. No additive identity or unique causal credit allocation is implied.
12. **Final-period prediction.** MAE is 67.29 versus calendar-baseline 88.54; R² is 0.756. Lower MAE than development does not mean universally better generalization: R² is lower and the target distribution changed. Do not retune on this period and still call it a fresh test.

## Why the case is credible without a dramatic reveal

The hour investigation provides readily visible heterogeneous responses. The temperature investigation exposes a methodological issue even though PDP and ALE shapes are broadly similar. Students should not learn that one method must disagree with another for an investigation to be worthwhile.

Unrestricted permutation and synthetic hour substitutions are themselves assumption-sensitive. Keep those limits beside the figures rather than suggesting only temperature PDP can create questionable inputs.

## Suggested answer to the manager

Our fixed model predicts hourly rentals more accurately than the simple calendar baseline in both evaluated periods. On October–December 2012, its mean absolute error is 67.29 rentals/hour, compared with 88.54 for that baseline. In the July–September development period, shuffling hour produces the largest increase in error, so hour is an important input under this test. However, individual hour-response curves differ, and a single average hides that variation. Temperature and perceived temperature are strongly related; independently changing temperature creates many queries distant from observed training pairs. ALE reduces those distant-pair queries across the tested cutoffs, although its curve broadly resembles the PDP. These findings describe this model's behavior under specified comparisons. They do not establish why riders rent bikes, and the pairwise support flags do not measure explanation accuracy. Before prospective operational use, we would need a prediction-time definition, weather inputs available at that time, and evaluation aligned with the intended planning decision.

## A simple 10-point report rubric

| Criterion | Points |
|---|---:|
| Correctly states the task and evaluation population | 2 |
| Distinguishes PFI reliance from feature-effect shape | 2 |
| Uses ICE and PDP/ALE findings accurately | 2 |
| Gives specific limits without unsupported causal claims | 2 |
| Uses actual evidence and proposes a justified follow-up | 2 |

## Optional investigations after the main case

Only introduce these as new questions, not missing prerequisites for this completed lesson: alternative seasonal reference populations, calendar-aware permutation protocols, or comparison of temperature measurement representations. Once final-test results have been observed, further model selection requires a new evaluation design; it cannot reuse the same period as an untouched final test.
