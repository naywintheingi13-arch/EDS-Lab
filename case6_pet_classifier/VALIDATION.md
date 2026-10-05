# Validation scope

The source was reconstructed after the unfinished first-session files were lost. Data were downloaded
again, published archive MD5 values checked, and the fixed experiment rerun from original images.
The development accuracy, confusion matrix, and four selected images reproduced the earlier run.
The first session's transcript is not used as a substitute for the regenerated output files.

## Checks performed by source code

- Source archive checksums; official pretrained weight SHA-256 retained in summary.json.
- Train/dev/test image names disjoint; exact JPEG-byte duplicates across splits checked separately.
- Trimap label range and shared resize/crop geometry; nearest-neighbor masks.
- Torch head scores agree with fitted scikit-learn logistic-regression scores.
- IG checked against a known linear model with signed coefficients and completeness.
- Each Grad-CAM calculation checked against exact CAM algebra for this pooling + linear head.
- IG integration tolerance recorded for every image/baseline, with step escalation and residuals.
  All eight final calculations pass. A difficult mean-baseline case required 1,024 Gauss–Legendre
  nodes; failed earlier attempts remain in the record. Both integration rules were checked on the
  known linear function. Passing completeness does not prove pixelwise convergence.
- Notebook checks model weight identity, sample selection, a cached-vs-image prediction, and reproduced
  development/test metrics. All numerical results come from code execution, not invented examples.

The final notebook was executed sequentially in a fresh IPython process during validation. No Jupyter GUI
or standard kernel launch is implied by that validation method. A clean install on every student operating
system remains untested.

## Practical limits

CPU host used two PyTorch threads. Timing varies by machine. Package dependencies are pinned for
the analysis, but the JupyterLab UI version is a compatible range and was not exercised here.
No representative large-scale explanation benchmark, calibration study, near-duplicate screening,
ImageNet overlap analysis, external-domain testing, or deployment readiness assessment was done.
The randomization check covers raw gradient magnitude and Grad-CAM, not every possible explainer.

The quick notebook refits from cached embeddings and recomputes one image's explanations. The full
source experiment generated all four. Notebook cache reuse is declared explicitly; it is not a claim
that every image was decoded during the notebook run.

## Final artifact checks

The final notebook has 26 cells, including 12 executed code cells, with no error outputs and eight
embedded figures. Sequential fresh-IPython execution took approximately 43 seconds on this host.
The generated analysis figures were visually inspected during validation. The notebook passed nbformat
schema validation. No browser GUI rendering or local Jupyter launch is claimed.

The exploratory original-versus-preprocessed-image comparison is explicitly labelled and does not
change the fitted model or the reserved test metric. All original images remain attributed to their
source; the experiment includes no generated pet imagery.
