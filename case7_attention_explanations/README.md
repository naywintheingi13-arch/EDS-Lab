# Case 7 — Can We Trust the Highlight?

An executed teaching case on attention faithfulness using SST-2 movie-review sentiment.
Start with **07_attention_explanation_audit.html** for a no-install preview,
or **07_attention_explanation_audit.ipynb** to run the lesson.

## Classroom route

Unzip the complete package. Keep the notebook, Python modules, data, and outputs together.
The notebook loads the supplied seed-2026 checkpoint; it does not retrain by default.
Predictions and cohort results for three seeds are included. Selected diagnostics are
recomputed live. A 90-minute core lesson plus optional 30-minute extension is suggested.

In a terminal inside the extracted directory (Linux, macOS, or WSL):

Use Python 3.12 for the closest match to the executed release (3.12.14).

```bash
python3 -m venv .venv-case7
source .venv-case7/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m ipykernel install --user --name eds-case7 --display-name "EDS Case 7"
python smoke_test_case7.py
jupyter lab
```

Select **EDS Case 7** as the notebook kernel. In VS Code on WSL, open the extracted
folder from the WSL terminal with `code .`, then select that same kernel. This creates
a separate environment and does not require changing an existing course environment.
For native Windows, activate with `.venv-case7\Scripts\activate` instead.

## Instructor/reproduction route

```bash
python run_case7_experiments.py
python build_teaching_materials.py
python execute_notebook.py
```

The first command reruns three CPU models and all cohort diagnostics. It overwrites
the generated outputs in this extracted package. Use a separate copy to preserve a release.
The run manifest records the actual build environment and training duration. Results can
differ slightly across operating systems and library versions. No GPU is required.

## Contents

- Notebook + HTML: full lesson with observed results and student prompts.
- `INSTRUCTOR_GUIDE.md`: timing, lecture mapping, expected reasoning, grading rubric.
- `STUDENT_WORKSHEET.md`: questions and an evidence-based stakeholder verdict.
- `FINDINGS.md`: measured release results and limits.
- `case7_attention_audit.py`: tokenization, BiLSTM, attention, training, evaluation.
- `audit_pipeline.py`: shared token and attention interventions.
- `run_case7_experiments.py`: identical fixed recipe across three seeds.
- `smoke_test_case7.py`: targeted implementation checks.
- `outputs/`: model weights, vocabulary, exact training/audit subsets, per-example
  diagnostics, all attention trials, metrics, provenance, and histories for every seed.
- `data/`: official GLUE SST-2 CSV copies prepared from its public archive.

## Scope and evidence contract

This is a small educational BiLSTM, not a transformer or a production review platform.
The hypothetical platform provides the stakeholder story. The benchmark data are real.
The model trains on a fixed 20,000-row sample of cleaned official training data, not all
67,349 rows. Training contains review fragments; validation contains sentences.
No validation-driven early stopping or hyperparameter tuning is performed by the script.
Official labeled validation is used for evaluation and exploratory audits; it is not an
independent confirmation set after lesson development. Public test labels are not used.
Exact duplicate/conflict checks do not eliminate related fragments or near duplicates.

Attention is a pooling weight over contextual states. Gradients and replacements answer
different questions. Neither is a ground-truth explanation. Results apply to this setup;
three seeds do not establish generality across architectures, tasks, or populations.

## Sources and data use

- SST: Socher et al. (2013), https://aclanthology.org/D13-1170/
- Official GLUE archive: https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
- Dataset project: https://nlp.stanford.edu/sentiment/
- Jain & Wallace (2019): https://aclanthology.org/N19-1357/
- Wiegreffe & Pinter (2019): https://aclanthology.org/D19-1002/

This package does not claim ownership of dataset text or grant additional rights to it.
Retain source attribution and consult the dataset's terms before redistribution beyond
the course. The experiments are a teaching adaptation, not an exact paper replication.
The supplied course PDF is not redistributed in this package.
