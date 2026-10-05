# Case 6 instructor guide

## Purpose and place in the course

Case 6 is the image-explanation audit for the Deep Learning Explainability lecture. The case should not become a catalog of every explanation technique. Its purpose is to make students distinguish a convincing visualization from evidence that bears on a specific explanation claim.

The continuous story is deliberately stakeholder-centered: a strong pet classifier has attractive heatmaps, but the audit discovers that preprocessing, explanation reference choices, perturbation design, and learned parameters all matter to the interpretation.

## Preparation

For discussion-only teaching, open the executed HTML. For a hands-on class, run `prepare_case6_assets.py --data-dir pet_data` before class and keep the generated asset folders beside the notebook. Read `FINDINGS.md` and `VALIDATION.md` before facilitating discussion.

The standard notebook refits only the small logistic head from cached ResNet features and recomputes selected diagnostics. Do not regenerate the full dataset pipeline during a 90-minute class unless that is itself the exercise.

## Suggested 90-minute lesson plan

| Minutes | Activity | Evidence of learning |
|---|---|---|
| 0-10 | Stakeholder question, prediction contract, class imbalance | Correctness/confidence separated from explanation quality |
| 10-20 | Predeclared example selection and confident errors | Selection bias recognized |
| 20-30 | Inspect preprocessing and effective model input | Students explain why input transformation belongs to the audited pipeline |
| 30-42 | Gradients, Grad-CAM, and visual plausibility | Observation separated from stronger faithfulness claim |
| 42-55 | IG baseline choice and completeness | Baseline-relative meaning and numerical completeness interpreted correctly |
| 55-65 | Localization and area enrichment | Spatial concentration distinguished from ground-truth reasoning |
| 65-75 | Perturbation/removal tests | Replacement assumptions and score sensitivity articulated |
| 75-83 | Parameter randomization | Model dependence interpreted as a sanity check, not certification |
| 83-88 | Artificial marker shortcut challenge | Clever Hans hypothesis tested without forcing a positive result |
| 88-90 | Bounded audit verdict / exit ticket | Claim proportional to evidence |

## Mapping to the supplied Deep Learning Explainability lecture

| Lecture material | Case 6 activity |
|---|---|
| Slide 71: Grad-CAM mechanism | Final-convolution Grad-CAM and explicit 7x7 spatial-resolution discussion |
| Slide 93: Integrated Gradients | Path from baseline to input; two baselines; numerical completeness check |
| Slide 101: danger of visual assessment | Students first inspect plausible maps, then design/tests that could disconfirm the visual story |
| Slides 102-103: model randomization / input visualizers | Head and full-network parameter randomization across multiple seeds |
| Slide 105: Clever Hans problem | Label-correlated corner-marker stress test |
| Slide 114: practitioner checklist | Multiple IG baselines, model randomization, and "never rely solely on visual inspection" are exercised directly |

The lecture also mentions SmoothGrad, label randomization, class discriminativity, and adversarial robustness. They are not required for this case. Keep them as optional discussion or follow-up ideas rather than presenting their absence as a flaw in the core lesson.

## Facilitation guidance

**Correctness versus explanation quality.** High held-out accuracy establishes that the model predicts well on the declared subset. It does not identify the evidence used for an individual decision.

**Preprocessing is part of the model.** The Persian example is pedagogically important because the center crop removes visible information. The fit-and-pad comparison changes framing, scale, and padding together, so it is exploratory rather than an isolated causal test.

**Grad-CAM resolution.** The map originates from the final 7x7 convolutional representation and is upsampled. The smooth 224x224 overlay should not be described as fine-grained pixel evidence.

**Integrated Gradients baseline.** IG is relative to a reference. A mean-color baseline and a blurred-image baseline encode different counterfactual/reference questions. Baseline sensitivity should be reported, not hidden.

**Completeness.** The case numerically checks that the sum of IG attributions approximates the model-score difference between input and baseline. Passing this check supports the correctness of the numerical integration/accounting relation. It does not establish that the baseline is meaningful, the explanation is faithful, or every pixel attribution has converged. A failed tolerance should trigger investigation, not deletion of the inconvenient row.

**Localization.** The trimap is ground truth for pet location, not ground truth for model reasoning. Compare foreground attribution share with foreground area share; otherwise a large animal can make a diffuse map look impressive.

**Perturbation.** Top-attribution versus bottom/random removal is behavioral evidence only under the chosen replacement operation. Mean-fill and blur-fill can alter the conclusion because each creates a different intervention and distribution shift.

**Randomization.** If maps survive severe parameter randomization, they may be tracking input/architecture structure rather than learned parameters. Conversely, changing after randomization is only evidence of model dependence; it does not by itself prove faithfulness.

**Shortcut test.** The corner marker is deliberately designed. The marker-trained head does not show dramatic dominance in this release. Treat that modest result as scientifically useful: the correct lesson is to report the observed sensitivity, not the story the experiment was hoped to produce.

**Target convention.** During interventions, the explained class is fixed to the original predicted class. Changing the target after a prediction flip would silently change the quantity being measured.

## Common misconceptions to correct

- "The heatmap overlaps the animal, therefore it is correct." Localization and faithfulness are different claims.
- "Completeness proves IG is faithful." Completeness is a score-accounting property relative to the baseline.
- "A higher-confidence prediction has a better explanation." Confidence and explanation quality are separate.
- "Upsampled Grad-CAM gives pixel-level evidence." The underlying map is coarse.
- "Changing pixels is a clean causal experiment." Replacement can introduce out-of-distribution artifacts.
- "A randomization pass certifies the method." It checks one failure mode only.
- "The shortcut test failed because the marker did not dominate." A weak effect is a valid result.
- "Test accuracy validates the explanation." The test set validates predictive performance under the stated evaluation protocol.

## Grading rubric (10 points)

| Criterion | Points |
|---|---:|
| Correctly states model/data/evaluation scope | 2 |
| Uses at least two numerical findings from different audit stages | 2 |
| Distinguishes plausibility/localization from faithfulness | 2 |
| Correctly explains either IG baseline/completeness or perturbation/randomization assumptions | 2 |
| Gives a bounded stakeholder conclusion plus a relevant next test | 2 |

Accept different final verdicts when evidence is used correctly. Deduct for treating visual plausibility, completeness, or a selected diagnostic example as proof of the model's true reasoning.

## Example evidence-based verdict

The frozen ResNet-18 plus logistic head predicts strongly on the reserved same-dataset subset, but that does not establish that its heatmaps are faithful explanations. The preprocessing audit is already cautionary: the center-cropped Persian example is confidently misclassified as dog, while the exploratory whole-image fit-and-pad view changes that prediction. Grad-CAM and IG often concentrate on the animal, but the trimap only measures location and IG changes with the chosen baseline. The IG completeness checks pass after sufficient numerical integration, which supports the attribution accounting relation but not semantic correctness. Perturbation conclusions also depend on whether removed pixels are mean-filled or blurred, and parameter randomization shows that the explanation maps depend on learned parameters without certifying them. The artificial marker produces limited rather than dominant shortcut sensitivity. I would describe the visualizations as diagnostic attribution evidence, not as proof that the classifier is "right for the right reasons," and would next predeclare a larger diagnostic sample and matched intervention protocol.

## Optional follow-up experiments

1. Predeclare a larger development cohort for explanation diagnostics.
2. Compare matched-area contiguous perturbations to reduce geometry differences.
3. Run label-randomized retraining if you want to extend directly to the lecture's second sanity check.
4. Compare a fine-tuned backbone with the frozen-feature setup for shortcut susceptibility.
5. Evaluate alternative preprocessing policies on a fresh, predeclared subset.
6. Add SmoothGrad or adversarial robustness only if those lecture topics need a separate extension exercise.

## Release limits

The case uses a controlled subset rather than a full benchmark study. Four diagnostic examples are not prevalence estimates. The preprocessing follow-up changes multiple factors together. Exact-byte duplicate checking does not address near duplicates or ImageNet pretraining overlap. No external-domain, calibration, adversarial, human-subject, or deployment-safety study is claimed.
