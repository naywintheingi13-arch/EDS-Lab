# Case 6 — The Pet Classifier Audit: findings

**The case works best as an audit of the whole prediction pipeline.** Strong predictive performance
and an attractive heatmap answer different questions. The most revealing finding concerns what
preprocessing lets the model see.

## 1. Strong performance, bounded claim

Fixed frozen ImageNet ResNet-18 plus class-balanced logistic head, trained on 1,110 pet images.

| Held-out subset | Correct | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| Development | 367 / 370 | 99.19% | 98.97% |
| Reserved test | 364 / 370 | 98.38% | 97.93% |

The dog-only baseline is 67.57% ordinary accuracy and 50% balanced accuracy. These are breed-stratified
subsets, not full benchmark results. No external-domain or pretraining-overlap audit was performed.
The reconstructed run reproduced the first session's metrics and selected examples exactly.

## 2. A confident error exposed a preprocessing problem

The center-cropped `Persian_190` is classified as dog with p(dog)=0.994239. Inspection of the original
photograph shows the center crop removed the cat's face. A development-only follow-up preserved the
whole image by resizing it to fit and padding with ImageNet mean color.

| Diagnostic image | Center-crop p(dog) | Fit-and-pad p(dog) | Outcome |
|---|---:|---:|---|
| Persian_190 (cat) | 0.994239 | 0.000329 | Wrong → correct |
| american_bulldog_151 (dog) | 0.361332 | 0.753255 | Wrong → correct |

The two selected correct examples also stayed correct. This does not establish that fit-and-pad is
globally better: only four selected development examples were examined, and framing, scale and
padding changed together. The test pipeline and test score were not revised after this observation.
This follow-up was exploratory and added after the initial protocol.

**Teaching implication:** inspect original and transformed inputs before explaining a confident error.

## 3. An animal-focused map is not proof of a faithful explanation

For `Abyssinian_108`, Grad-CAM puts 83.38% of its mass on known foreground, but the animal already
occupies 65.51% of known pixels. That is about 1.27× the uniform-map reference, not “83% explanation
accuracy.” The trimap describes location; it does not identify the model's true causal mechanism.

Across the selected examples, magnitude IG maps from mean and blur baselines have only moderate or
low spatial-rank agreement. Reference choice changes the question. Signed maps also reveal evidence
that magnitude plots hide. Independently normalized color scales cannot compare attribution strength.

## 4. Removal-test conclusions depend on how pixels are replaced

For the same Abyssinian image, replacing top-10% Grad-CAM pixels with mean color drops the original
target margin by 3.123, versus 0.691 for bottom-10%. With blur replacement, the drops are 0.345 versus
0.610: the ordering reverses. This is a concrete reason not to rely on one deletion setting.

Random-mask comparisons use ten masks with equal pixel counts; bars report mean and SD, not a
confidence interval. Scattered random masks and spatially concentrated CAM masks have different
geometry. Full-foreground/background comparisons additionally differ in area. None is an isolated
real-world causal experiment.

## 5. Randomization changes maps, but does not certify them

Mean gradient-magnitude rank correlation with the trained model is about 0.310 after head
randomization and 0.117 after whole-network randomization across four images × three seeds.
Grad-CAM correlations vary widely. Four of 24 randomization comparisons produce a zero Grad-CAM map;
their correlation is undefined and excluded, not counted as zero. See randomization.csv for individual
values; averages over four selected examples are not general fidelity estimates.

## 6. The artificial marker did not dominate this classifier

Balanced accuracy on the same 370 development images:

| Training | No marker | Label-aligned marker | Swapped marker |
|---|---:|---:|---:|
| Clean images | 98.97% | 97.52% | 98.35% |
| Marked images | 98.15% | 98.77% | 97.10% |

For the marker-trained head, aligned versus swapped differs by 1.67 percentage points, with four
additional ordinary classification errors. This is limited sensitivity in this run, not dominance
or a statistically established general effect. The pretrained backbone was frozen; it may already
expose strong animal features. Fine-tuning could behave differently and was not tested.

The marker is a designed 32×32 label-correlated square, not a natural dataset artifact. It can cover
part of the animal. We did not keep changing the marker until obtaining a dramatic failure.

## Numerical and teaching status

IG diagnostics record integration method, resolution, completeness residual and pass/fail for every
image/reference, including all refinement attempts. One mean-baseline example needed investigation
beyond the initial 256-step cap. All eight final calculations meet the declared tolerance. For
`american_bulldog_151` with the mean baseline, trapezoidal integration up to 1,024 steps and
Gauss–Legendre integration at 256/512 nodes failed; 1,024 Gauss–Legendre nodes reduced the residual
to -0.00630 score units (tolerance about 0.03675). All attempts are retained. Numerical completeness
remains distinct from faithfulness and does not guarantee every individual pixel attribution has
converged.

Recommended central question: **When a prediction and its heatmap look convincing, what must we test
before accepting the explanation?** Use the preprocessing discovery as the narrative turning point;
keep the marker experiment as a lesson about honest, potentially negative stress-test results.
