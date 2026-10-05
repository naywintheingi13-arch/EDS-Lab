# Case 6 teaching-release changelog

## Teaching-ready release

- Preserved the frozen ResNet-18 + logistic-head experiment and the fixed development/test protocol.
- Reframed the case around one continuous stakeholder audit question: whether a convincing heatmap is enough to trust the model's explanation.
- Made correctness, confidence, localization, behavioral relevance, model dependence, and shortcut robustness explicit as different claims.
- Moved preprocessing inspection before explanation interpretation so students audit the effective model input first.
- Clarified the explained quantity as the original predicted-class margin held fixed during interventions.
- Added explicit teaching around IG baseline dependence and numerical completeness, including the distinction between completeness and faithfulness.
- Added area-normalized localization interpretation rather than treating foreground overlap as explanation accuracy.
- Added perturbation caveats about replacement choice, distribution shift, region size, and geometry.
- Added parameter-randomization interpretation consistent with the lecture's sanity-check framing.
- Retained the artificial marker experiment even though it produces limited rather than dramatic shortcut sensitivity, emphasizing honest reporting of mixed/negative findings.
- Reserved the test subset until the audit protocol is established in the notebook narrative.
- Added a student evidence sheet and instructor guide aligned only to the deep-learning explanation concepts exercised by this case.
- Added `prepare_case6_assets.py` so the released notebook can be rebuilt from public data without relying on undocumented local folders.

## Deliberately not added to the core

SmoothGrad, label-randomized retraining, adversarial explanation robustness, TCAV, attention methods, and a general class-discriminativity experiment remain extensions. They are not required merely to mirror another course case.
