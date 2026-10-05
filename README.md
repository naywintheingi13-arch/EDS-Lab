# Explainable Data Science — Teaching Cases

Teaching materials developed for the Explainable Data Science course.

The repository contains seven completed case-based teaching modules covering foundations of explainability, interpretable models, decision support, model-agnostic explanation methods, global explanations, deep-learning visual explanations, and attention explanation auditing.

## Teaching Cases

### Case 1 — Foundations
**Can We Trust This Credit-Risk Model?**

Topics:
- evaluation beyond accuracy
- global and local interpretability
- model behavior and decision context
- stakeholder-oriented explanation
- limits of explanation and causal interpretation

Notebook: `notebooks/01_credit_risk_foundations.ipynb`

---

### Case 2 — Interpretable Models
**Designing an Interpretable Home Energy Model**

Topics:
- linear models and regularization
- decision trees
- generalized additive models
- comparison of interpretable and more complex models
- purposeful model evaluation and interpretation

Notebook: `notebooks/02_home_energy_interpretable_models.ipynb`

---

### Case 3 — From Prediction to Decision Support
**From Risk Prediction to Decision Support**

Topics:
- sparse predictive models
- interpretable scorecards
- ranking versus probability calibration
- decision thresholds
- recourse
- interpretation boundaries

Notebook: `notebooks/03_bankruptcy_decision_support.ipynb`

---

### Case 4 — Model-Agnostic Local Explanations
**Why Does the Model Think This Shopper Will Buy?**

Topics:
- LIME
- local explanation stability
- local surrogate fidelity
- perturbation-neighborhood validity
- SHAP
- SHAP reference/background sensitivity
- correlated-feature attribution
- explanation auditing

Notebook: `notebooks/04_online_shopping_local_explanations.ipynb`

---

### Case 5 — Global Model-Agnostic Explanations
**Understanding Bike-Rental Demand Beyond Feature Importance**

Topics:
- permutation importance
- partial dependence plots
- individual conditional expectation
- grouped feature effects
- ALE
- support and extrapolation
- sensitivity of global explanations

Package: `case5_bike_rentals_global_explanations/`

Start with:
- `05_bike_rentals_global_explanations.html` for an executed preview
- `05_bike_rentals_global_explanations.ipynb` for the runnable notebook

---

### Case 6 — Deep-Learning Explainability
**The Pet Classifier Audit**

Central question:

> When a prediction and its heatmap look convincing, what must we test before accepting the explanation?

Topics:
- preprocessing as part of the explanation pipeline
- vanilla gradients
- Grad-CAM
- Integrated Gradients
- attribution baselines
- signed versus magnitude attribution
- Integrated Gradients completeness
- localization
- perturbation tests
- parameter randomization
- shortcut / Clever Hans reasoning
- bounded interpretation of explanation evidence

Package: `case6_pet_classifier/`

Start with:
- `06_pet_classifier_audit.html` for an executed preview
- `06_pet_classifier_audit.ipynb` for the runnable lesson

Teaching support:
- `INSTRUCTOR_GUIDE.md`
- `STUDENT_WORKSHEET.md`
- `FINDINGS.md`
- `VALIDATION.md`
- `SOURCES.md`

---

### Case 7 — Attention Explanation Audit
**Can We Trust the Highlight?**

Topics:
- attention-based explanations
- attention versus attribution
- explanation plausibility versus evidence
- input interventions
- fixed-hidden attention interventions
- aggregate explanation comparison
- disagreement cases
- explanation stress testing
- stakeholder-facing audit conclusions

Package: `case7_attention_explanations/`

Start with:
- `07_attention_explanation_audit.html` for an executed preview
- `07_attention_explanation_audit.ipynb` for the runnable lesson

Teaching support:
- `INSTRUCTOR_GUIDE.md`
- `STUDENT_WORKSHEET.md`
- `FINDINGS.md`
- `CHANGELOG.md`

## Repository Structure

```text
EDS-Lab/
├── notebooks/
│   ├── 01_credit_risk_foundations.ipynb
│   ├── 02_home_energy_interpretable_models.ipynb
│   ├── 03_bankruptcy_decision_support.ipynb
│   └── 04_online_shopping_local_explanations.ipynb
│
├── Data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── outputs/
│   └── case4_online_shopping/
│
├── src/
│
├── case5_bike_rentals_global_explanations/
├── case6_pet_classifier/
├── case7_attention_explanations/
│
├── prepare_appliances_course_data.py
├── prepare_bankruptcy_course_data.py
├── prepare_credit_course_data.py
├── prepare_credit_data.py
├── run_case4_online_shopping_experiments.py
│
├── requirements.txt
└── README.md



history | tail -20
