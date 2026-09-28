# Explainable Data Science — Teaching Cases

Teaching materials developed for the Explainable Data Science course.

The repository currently contains four completed case-based notebooks covering foundations, interpretable models, decision-support models, and model-agnostic local explanation methods.

## Teaching Cases

### Case 1 — Foundations
**Can We Trust This Credit-Risk Model?**

Topics include:
- evaluation beyond accuracy
- global and local interpretability
- model behavior and decision context
- stakeholder-oriented explanation
- limits of explanation and causal interpretation

Notebook: `notebooks/01_credit_risk_foundations.ipynb`

### Case 2 — Interpretable Models
**Designing an Interpretable Home Energy Model**

Topics include:
- linear models and regularization
- decision trees
- generalized additive models
- comparison of interpretable and more complex models
- purposeful model evaluation and interpretation

Notebook: `notebooks/02_home_energy_interpretable_models.ipynb`

### Case 3 — From Prediction to Decision Support
**From Risk Prediction to Decision Support**

Topics include:
- sparse predictive models
- interpretable scorecards
- ranking versus probability calibration
- decision thresholds
- recourse and interpretation boundaries

Notebook: `notebooks/03_bankruptcy_decision_support.ipynb`

### Case 4 — Model-Agnostic Local Explanations
**Why Does the Model Think This Shopper Will Buy?**

Topics include:
- LIME
- local explanation stability
- local surrogate fidelity
- perturbation-neighborhood validity
- SHAP
- SHAP reference/background sensitivity
- correlated-feature attribution
- explanation auditing

Notebook: `notebooks/04_online_shopping_local_explanations.ipynb`

## Repository Structure

```text
EDS-Lab/
├── notebooks/      Teaching notebooks
├── Data/           Raw, processed, and metadata files
├── outputs/        Supporting experiment artifacts
├── src/            Supporting implementation code
├── requirements.txt
└── README.md

```

Data-preparation and research scripts are kept at the repository root to preserve the current reproducible workflow.

## Current Status

Cases 1–4 are completed teaching cases.

Additional course cases are under development.
