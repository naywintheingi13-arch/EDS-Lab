# Case 6 - The Pet Classifier Audit

**Central question:** When a prediction and its heatmap look convincing, what must we test before accepting the explanation?

This is the teaching-ready release for the Deep Learning Explainability portion of the Explainable Data Science course. It uses a frozen ImageNet ResNet-18 with a class-balanced logistic head on a controlled Oxford-IIIT Pet cat-vs-dog task.

Start with **`06_pet_classifier_audit.html`** for a no-install preview, or **`06_pet_classifier_audit.ipynb`** for the runnable lesson.

## What this case teaches

The case is intentionally narrower than a general XAI survey. It reinforces the lecture concepts that are directly exercised here:

- Grad-CAM and what the final convolutional feature maps represent;
- Integrated Gradients and the role of the baseline;
- the difference between visual plausibility and faithfulness;
- numerical completeness as an implementation/consistency check for IG, not proof of faithfulness;
- localization as evidence about spatial concentration, not ground-truth reasoning;
- perturbation tests and their replacement-distribution assumptions;
- model-parameter randomization as a sanity check;
- the Clever Hans / shortcut-learning problem;
- bounded communication of mixed and negative audit evidence.

SmoothGrad, label-randomized retraining, adversarial robustness, TCAV, attention methods, and a general class-discriminativity study are not part of the core case. They remain possible extensions rather than hidden requirements.

## Classroom route

For a lecture or discussion section, open the executed HTML. It contains the observed figures and results and does not require Python.

For an interactive lab, prepare the local teaching assets once, then run the notebook:

```bash
python3.12 -m venv .venv-case6
source .venv-case6/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python prepare_case6_assets.py --data-dir pet_data
jupyter lab
```

On native Windows, activate with `.venv-case6\Scripts\activate` instead.

The preparation command downloads the public Oxford-IIIT Pet archives and pretrained ResNet-18 weights, verifies published dataset archive checksums, generates the cached ResNet features and audit outputs, prepares the four selected diagnostic samples, and copies the model weights into the location expected by the notebook.

The first preparation is substantially heavier than opening the HTML because the original image dataset must be downloaded and processed. Keep the generated `cache/`, `outputs/`, `samples/`, and `weights/` folders beside the notebook for later quick runs.

## Full regeneration contract

The fixed experiment uses:

- seed `2026`;
- 30 train / 10 development / 10 reserved-test images per breed;
- 1,110 training, 370 development, and 370 reserved-test images;
- frozen ImageNet ResNet-18 features;
- a class-balanced binary logistic head;
- no notebook hyperparameter search;
- four development examples selected by a predeclared rule before explanation inspection;
- original predicted-class margin as the fixed explanation target.

Running `prepare_case6_assets.py --fresh --data-dir pet_data` removes generated Case 6 assets before rebuilding them. Use a copy of the release if you want to preserve an existing teaching run.

## Contents

- `06_pet_classifier_audit.ipynb` - student-facing lesson source.
- `06_pet_classifier_audit.html` - executed no-install preview.
- `INSTRUCTOR_GUIDE.md` - lesson timing, lecture mapping, interpretation guidance, rubric.
- `STUDENT_WORKSHEET.md` - evidence sheet and final stakeholder/audit task.
- `FINDINGS.md` - measured release findings and bounded claims.
- `VALIDATION.md` - numerical and implementation checks already performed.
- `SOURCES.md` - data, model, method, and course-source attribution.
- `CHANGELOG.md` - teaching-release changes and scope decisions.
- `audit.py` - fixed experiment and explanation diagnostics.
- `check_preprocessing.py` - development-only original-vs-preprocessing diagnostic.
- `plots.py` - figures used by the case.
- `download_data.py` - public-data downloader and archive verification.
- `prepare_case6_assets.py` - one-command asset preparation for the notebook.
- `requirements.txt` - release environment requirements.
- `Oxford_dataset_README.txt` - original dataset archive README retained for provenance.
- `cache/`, `outputs/`, `samples/`, `weights/` - generated teaching assets after preparation.

## Evidence contract

High predictive performance is evidence of predictive competence on the declared held-out subset. It is not evidence that a heatmap is faithful.

A foreground-focused attribution map shows where attribution mass is concentrated, not why the network made the prediction. IG completeness checks whether the attributions numerically account for the score difference between input and baseline under the implementation; it does not certify semantic correctness, baseline choice, causal relevance, or pixelwise convergence. Perturbation and randomization tests probe additional aspects of the explanation but each carries its own assumptions.

The marker experiment is a designed shortcut stress test, not a discovered artifact in Oxford-IIIT Pet. Its modest effect is retained as a teaching example of why an audit should report what happened rather than force a dramatic story.

## Data and redistribution

The Oxford-IIIT Pet Dataset is attributed to Parkhi, Vedaldi, Zisserman, and Jawahar (2012). The original archive README is preserved because its wording and source-image terms matter. See `SOURCES.md` before redistributing dataset-derived files outside the course.

The supplied course lecture PDF is not redistributed in this package.
