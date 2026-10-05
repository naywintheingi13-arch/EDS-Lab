# Case 7 instructor guide

## Purpose and place in the course

Case 6 investigated image explanations. Case 7 changes modality to text and asks whether
an internal weighting mechanism is enough to justify a word-level explanation. Keep this
case focused on attention, contextual representations, and evidence for faithfulness.
Do not expand it into a transformer survey or a whole-course revision.

The opening review-platform scenario is hypothetical; SST-2 is real benchmark data.
The uploaded SST-2 case is retained as the current teaching context. A news adaptation
is a future option, not part of this release.

## Preparation

Read FINDINGS.md, run the regression checks, open the executed HTML, and choose whether
students use the notebook or work from saved output. Confirm the supplied kernel works
on the teaching machine. The package was executed in the build environment; the user's
particular WSL environment has not been tested. Retrain before class, not during the lesson.
Consult outputs/run_manifest.json for the actual run duration and package versions.

The standard route loads the model, so every student starts from the same evidence.
The model uses a fixed 20,000-example subset and a small 64-dimensional embedding /
48-dimensional bidirectional LSTM. It is an inspectable educational classifier.
Do not present this as a competitive SST-2 benchmark or a reproduction of a published paper.

## Suggested lesson plan

| Minutes | Activity | Evidence of learning |
|---|---|---|
| 0–10 | Platform story; predict what a highlight proves | Observation versus explanation claim |
| 10–20 | Data, vocabulary, baseline, prediction errors | Competence separated from faithfulness |
| 20–30 | Contextual state and pooling weight | Students explain why position is not isolated word |
| 30–45 | Attention, gradient magnitude, ID replacement | Distinct questions and signed replacement effects |
| 45–60 | Fixed-hidden intervention and TV distance | Attention change distinguished from output change |
| 60–72 | Cohort, extreme disagreement, confident error | Distribution separated from selected examples |
| 72–82 | Language probes and alternative explanations | One passing probe is not language understanding |
| 82–90 | Stakeholder verdict and exit ticket | Claim proportional to evidence |
| Optional +30 | Margin decomposition and three seeds | Conditional accounting versus causal attribution |

## Mapping to the supplied deep-learning lecture

| Lecture slides | Case activity |
|---|---|
| 6–10: attention weights and weighted context | Notebook mechanism and token table |
| 11–12: attention as explanation questions | Stakeholder hypothesis and evidence contract |
| 19: gradients and leave-one-out comparison | Three diagnostics; UNK replacement explicitly differs from deletion |
| 23–25: alternative attention and permutation | Frozen hidden states, uniform weights, repeated shuffles |
| 27–28: context and architecture limitations | Conditional contributions and scope of conclusions |

## Facilitation and answer guidance

**Correct versus faithful.** A correct prediction may rely on an unreliable cue. A faithful
account can explain a mistaken prediction. Ask students to identify which question a metric answers.

**Attention versus gradient.** Attention is nonnegative and sums to one. Our gradient diagnostic
is an unsigned L1 magnitude of gradient times embedding, normalized across positions, for the
original predicted class's logit margin. It is not a probability derivative, a signed dot product,
or a ground-truth importance label. High correlation is agreement between diagnostics, not proof.

**UNK replacement.** It replaces exactly one ID, preserving length. Re-encoding changes contextual
states and attention. The signed drop is original minus replacement margin; a negative value
means replacing the token increased the original target score. Already-UNK positions are no-ops.
Students should inspect the unknown-token indicator, not mistake zero change for irrelevance.

**Internal intervention.** It holds contextual states fixed and changes pooling weights.
Uniform attention and permutations can be off the naturally produced attention distribution.
Sensitivity identifies a role for this mechanism under the specified intervention. It does
not isolate the semantic effect of a word. Failure to find an alternative in 30 shuffles does
not prove that no alternative exists.

**Probability saturation.** Use both probability and margin changes. For two classes,
P(target)=sigmoid(target margin). A large margin change at a very confident prediction can
be almost invisible in probability space. Raw target logit alone would ignore the other class.

**TV distance.** A small output change is informative about alternative attention only when
attention itself changed substantially. An exactly uniform distribution is unchanged by
permutation. The targeted checks include this null case.

**Selection.** The cohort is fixed independently of diagnostic results. The disagreement
example minimizes a diagnostic; it is transparently selected but still extreme. The error
example deliberately prioritizes confidence. Neither estimates prevalence.

**Observed results.** Use FINDINGS.md and the notebook's current tables, not a predetermined
“attention fails” story. Attention changes do affect this model on a meaningful portion of
the cohort. Ranking agreement is imperfect. Both statements can hold simultaneously.
The baseline and neural results should be reported even when the neural model loses.

**Conditional decomposition.** Summing alpha_i times the projected contextual state plus
the bias reproduces the margin. It is algebraically exact at fixed states and not a causal
assignment of credit to independently manipulable words. This is an optional extension.

## Grading rubric (10 points)

| Criterion | Points |
|---|---:|
| Defines claim, model, data and intervention scope | 2 |
| Uses two correct numerical results, including aggregate/seed evidence | 2 |
| Separates correctness, confidence, plausibility and faithfulness | 2 |
| Names a relevant limitation and a test that addresses it | 2 |
| Gives calibrated stakeholder wording without forcing a conclusion | 2 |

Accept different verdicts if supported. Do not require both positive and negative evidence
when students cannot substantiate both. Deduct for treating diagnostic agreement as truth,
unchanged label as unchanged score, or selected extreme examples as representative.

## Example of an evidence-based verdict for this release

We audited attention pooling in a small BiLSTM trained on 20,000 SST-2 fragments.
The seed-2026 model achieved 77.6% validation accuracy, compared with 79.0% for TF-IDF
logistic regression. On the fixed 120-example cohort, mean attention–gradient rank
correlation was 0.493. Replacing attention with uniform weights changed 14 of 120 labels
(11.7%), with mean attention TV distance 0.741 and mean absolute target-margin change
1.567. Thus attention has measurable predictive influence, while the diagnostics do not
provide identical rankings of token importance. Across three seeds, uniform-attention
flip rates ranged from 5.8% to 11.7%. These results neither establish that attention is
irrelevant nor identify highlighted words as unique causes. Contextual states combine
information from multiple positions, and UNK replacement can create unnatural inputs.
We recommend describing the display as attention pooling weights, with no claim that
it is a complete explanation. A next study should assess a fresh evaluation set and
human understanding of the proposed display before using it on the platform.

## Optional follow-up experiments

1. Compare ID replacement with actual deletion, naming both interventions precisely.
2. Compare gradients of the margin with probability gradients to explore saturation.
3. Evaluate on a new held-out set before making confirmatory claims.
4. Compare an encoder without contextual recurrence; do not assume its result in advance.
5. Conduct a separate human study if claiming highlights are plausible or useful to users.

## Release limitations

Three seeds assess initialization/training-order variation only. Exact duplicate removal
does not resolve shared source reviews, near duplicates, or semantic overlap. SST-2 training
fragments differ from validation sentences. Vocabulary coverage is imperfect. No hyperparameter
search, calibrated uncertainty estimate, production safety review, or transformer evaluation
is provided. These are stated limits of this teaching case, not additional student gates.
