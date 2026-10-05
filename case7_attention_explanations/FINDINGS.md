# Case 7 — Release findings

These are measured outputs of the included fixed experiment, not expected answers invented before training.

## Prediction results

| model                      |   accuracy |   balanced_accuracy |   log_loss |
|:---------------------------|-----------:|--------------------:|-----------:|
| majority / class prior     |     0.5092 |              0.5000 |     0.6977 |
| TF-IDF logistic regression |     0.7901 |              0.7892 |     0.4647 |
| BiLSTM attention seed 2026 |     0.7764 |              0.7753 |     0.6469 |
| BiLSTM attention seed 2027 |     0.7901 |              0.7894 |     0.5682 |
| BiLSTM attention seed 2028 |     0.7787 |              0.7775 |     0.6454 |

## Explanation audit: means over the same 120 examples

|      seed |   attention_gradient_rho |   attention_replacement_rho |   gradient_replacement_rho |   uniform_tv |   uniform_abs_dp |   uniform_abs_dm |   uniform_flip |   shuffle_flip_rate |
|----------:|-------------------------:|----------------------------:|---------------------------:|-------------:|-----------------:|-----------------:|---------------:|--------------------:|
| 2026.0000 |                   0.4930 |                      0.3889 |                     0.4265 |       0.7409 |           0.1151 |           1.5671 |         0.1167 |              0.1375 |
| 2027.0000 |                   0.5894 |                      0.4575 |                     0.4503 |       0.7022 |           0.0859 |           1.3985 |         0.0583 |              0.1042 |
| 2028.0000 |                   0.5082 |                      0.2993 |                     0.4283 |       0.7201 |           0.1087 |           1.6120 |         0.0667 |              0.1211 |

TV measures attention change; dp is absolute probability change; dm is absolute target-logit-margin change. Uniform flip is a sentence-level rate. Shuffle flip is the mean within-sentence flip rate across 30 permutations. These are descriptive results, not independent trials for a significance test.

## Interpretation

Attention has measurable predictive influence in this setup, while attention, gradients, and token replacement do not give identical rankings. A changed internal distribution can change output scores, so the results do not justify saying that attention is irrelevant. Nor does sensitivity to attention prove that a highlighted visible word is the unique cause: the hidden states contain context. Inspect the distributions and model errors in the notebook.

The simple baseline is a substantive comparison. A neural model need not outperform it for the audit to teach something; its performance also does not justify deployment. Probability confidence is not calibrated confidence.

## Data and limits

{
  "original_train": 67349,
  "original_validation": 872,
  "train_duplicate_rows": 371,
  "conflicting_train_keys": 5,
  "cross_split_exact_keys_removed": 0,
  "validation_duplicate_rows": 0,
  "clean_training_pool": 66973,
  "selected_training_rows": 20000,
  "vocab_size": 9649,
  "validation_unknown_rate": 0.0701579935918683,
  "train_truncation_rate": 0.0,
  "validation_truncation_rate": 0.0
}

One fixed 20,000-row training sample, one labeled validation set, three seeds, one small BiLSTM architecture. No confirmatory holdout after case development. Training fragments can share source reviews; exact checks do not resolve near duplication. UNK replacement can create unnatural inputs and cannot further change an already unknown ID. The cohort is modest, and the detailed disagreement/error cases are intentionally extreme. No human plausibility study, calibration study, production audit, transformer claim, or cross-dataset generalization.

## Verification

Targeted checks cover one-ID replacement, padding invariance, identical-attention override, a uniform-attention null intervention, conditional contribution reconstruction, ranking behavior, and empty inputs. The executed notebook is regenerated with captured outputs; see outputs/notebook_execution.json.