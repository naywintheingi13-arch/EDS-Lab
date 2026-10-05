from pathlib import Path
import json,textwrap
import pandas as pd
import nbformat as nb
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs'
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md(r'''
# Case 7 — Can We Trust the Highlight?
### Auditing attention explanations with real SST-2 data

**Stakeholder story (hypothetical):** a review platform predicts positive or negative
sentiment and wants to show highlighted words as the reason for its decision.
**Audit question:** what claim about those highlights can the evidence justify?

In Case 6 we examined highlighted image regions. Here we examine highlighted text
positions, including weights from inside the prediction mechanism itself.
Being internal to a model does not automatically make a signal a complete explanation.

**Learning outcomes:** explain attention pooling; distinguish correctness, confidence,
plausibility and faithfulness; compare attribution diagnostics; separate input from
fixed-hidden-state interventions; summarize evidence and its limits.

**Prerequisites:** train/evaluation separation, classification scores, basic gradients.
Core lesson: 90 minutes. Optional mechanism/robustness extension: 30 minutes.
Run top to bottom. The supplied checkpoint avoids training during class.

**Opening prediction:** would a correct label and a sensible-looking highlight be enough
to tell a user, “The model decided this because of these words”? Write your answer first.
''')
code('''
from pathlib import Path
import json, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p/'audit_pipeline.py').exists()), None)
if ROOT is None:
    raise FileNotFoundError('Extract the whole package and open the notebook inside it.')
sys.path.insert(0,str(ROOT))
import case7_attention_audit as audit
from audit_pipeline import load_model, token_audit, attention_audit, CONFIG, tensors
torch.set_num_threads(2)
OUT=ROOT/'outputs'
model,vocab=load_model(ROOT,2026)
train=pd.read_csv(OUT/'training_subset.csv')
validation=pd.read_csv(ROOT/'data/sst2_validation.csv')
pred=pd.read_json(OUT/'validation_predictions_seed2026.jsonl',lines=True)
aggregate=pd.read_csv(OUT/'audit_seed2026.csv')
trials=pd.read_csv(OUT/'interventions_seed2026.csv')
metrics=pd.DataFrame(json.loads((OUT/'predictive_metrics.json').read_text()))
quality=json.loads((OUT/'data_quality.json').read_text())
print('Loaded model seed 2026; label 0 = negative, label 1 = positive.')
print('Fixed classroom configuration:', CONFIG)
''')
md('''
## 1. Establish the data and prediction contract

SST-2 contains movie-review text. Official training includes short labeled fragments;
official validation has sentence-level examples. Our fixed sample contains 20,000 cleaned
training rows. The vocabulary and TF-IDF baseline are fitted only on those rows.

We remove normalized training duplicates and conflicting-label keys, and exclude any
exact training/validation overlap. This is **not** a near-duplicate or source-review audit.
Related fragments may remain. We do not randomly split fragments for early stopping.
All models train for four fixed epochs; the three seeds use the same data and recipe.

The 872 labeled validation examples support evaluation and exploratory lesson development.
They are not a fresh confirmation set after this development. Hidden public test labels
are not used. No claim of production readiness follows from this benchmark.
''')
code('''
display(pd.Series(quality,name='value').to_frame())
display(pd.DataFrame({'train_count':train.label.value_counts(),
                      'validation_count':validation.label.value_counts()}).sort_index())
lengths=train.sentence.map(lambda s:len(audit.simple_tokenize(s)))
display(lengths.quantile([.5,.9,.95,.99,1]).rename('training token length').to_frame())
fig,ax=plt.subplots(figsize=(8,3))
ax.hist(lengths,bins=35,color='#356c9b');ax.set(xlabel='Tokens',ylabel='Training examples',title='Training includes short review fragments')
plt.tight_layout();plt.show()
''')
md('''
**Checkpoint:** distinguish a duplicate check from evidence that every example is independent.
What might the fragment/sentence difference do to model performance?

## 2. Establish predictive competence

Compare the attention model with a class-prior baseline and TF-IDF logistic regression
trained on the same subset. A deeper model need not win. Explanation auditing remains
useful for a competent but imperfect model; do not interpret it as a deployment endorsement.
Accuracy measures agreement with reference labels. Log loss also penalizes confident errors;
it is not, by itself, a calibration assessment.
''')
code('''
display(metrics.drop(columns='confusion_matrix').round(4))
cm=np.asarray(metrics.loc[metrics.model=='BiLSTM attention seed 2026','confusion_matrix'].iloc[0])
display(pd.DataFrame(cm,index=['reference negative','reference positive'],columns=['predicted negative','predicted positive']))
majority=float(metrics.iloc[0].accuracy)
if float(metrics.loc[metrics.model=='BiLSTM attention seed 2026','accuracy'].iloc[0]) <= majority+.10:
    raise RuntimeError('Predictive competence gate failed: investigate before explaining.')
print('Competence gate passed: >10 percentage points above majority (a teaching threshold, not a deployment standard).')
''')
md(r'''
## 3. Understand the mechanism before interpreting it

Tokens become IDs and embeddings. A bidirectional LSTM produces contextual states
$h_i$: the state at one position can encode information from other words.
Additive attention gives $\alpha_i=\mathrm{softmax}(v^T\tanh(Wh_i+b))$.
The classifier uses $c=\sum_i\alpha_i h_i$ and logits $z=W_c c+b_c$.

Padding is masked. At most 80 tokens are used. Unknown words share one learned embedding.
This is one attention-pooling distribution, not transformer self-attention.

**High attention means:** a larger pooling weight for that contextual state.
It does not directly specify that word's signed contribution to a class.

First select a correct example with 6–35 tokens and confidence 0.80–0.97, nearest 0.90.
This is an inspectable example, not evidence that its explanation is good. If the preferred
pool is empty, the code transparently falls back to another correct example.
''')
code('''
pred['n_tokens']=pred.tokens.map(len)
pool=pred[pred.correct & pred.n_tokens.between(6,35) & pred.confidence.between(.8,.97)].copy()
if pool.empty:
    print('Preferred example pool empty; falling back to all correct examples.')
    pool=pred[pred.correct].copy()
if pool.empty: raise RuntimeError('No correct examples available; inspect the model.')
pool['distance']=(pool.confidence-.90).abs()
first=pool.sort_values(['distance','idx']).iloc[0]
table,info=token_audit(model,first.sentence,vocab)
print('Review:',first.sentence)
print('Reference:',first.label,'prediction:',info['target'],'probability:',round(info['target_probability'],4))
display(table[['position','token','token_id','is_unknown','attention']])
def plot_tokens(t,columns,title):
    fig,axes=plt.subplots(len(columns),1,figsize=(max(9,min(16,len(t)*.35)),2.5*len(columns)),squeeze=False)
    for ax,col in zip(axes[:,0],columns):
        ax.bar(np.arange(len(t)),t[col],color='#356c9b')
        ax.set_xticks(np.arange(len(t)),[f'{i}: {w}' for i,w in zip(t.position,t.token)],rotation=60,ha='right')
        ax.set_ylabel(col.replace('_',' '));ax.axhline(0,color='black',lw=.6)
    fig.suptitle(title);fig.tight_layout();plt.show()
plot_tokens(table,['attention'],'Observe the pooling weights before making an explanation claim')
''')
md('''
**Stop and write:** one observation, one hypothesis, and one claim requiring stronger evidence.
Example observation: “Position 5 receives more pooling weight than position 2.”
Do not yet substitute “causes the prediction.”

## 4. Compare three different questions

We use the originally predicted class throughout. Its **margin** is its logit minus the
other class's logit. Positive margin favors that class; its probability is sigmoid(margin).

| Diagnostic | Question | Limitation |
|---|---|---|
| Attention | Which contextual states receive pooling weight? | Not signed word importance |
| Gradient × embedding magnitude | Where is the margin locally sensitive, scaled by the embedding? | Local, representation-dependent, unsigned |
| Single-token UNK replacement | How does the margin change when this token ID is replaced? | Changes context; may create unnatural input |

Gradient magnitude is the sum of absolute componentwise gradient × embedding products,
normalized across positions. It is **not** the absolute value of a signed sum. We differentiate
the class margin to match the intervention target. Rankings use absolute replacement effects;
the signed drops remain visible: a negative drop means replacement increased the target margin.
No diagnostic is ground truth.
''')
code('''
display(table[['position','token','attention','gradient_magnitude','replacement_margin_drop','replacement_probability_drop']].round(4))
print('Attention–gradient Spearman:',round(audit.spearman(table.attention,table.gradient_magnitude),3))
print('Attention–|replacement| Spearman:',round(audit.spearman(table.attention,table.replacement_margin_drop.abs()),3))
print('Gradient–|replacement| Spearman:',round(audit.spearman(table.gradient_magnitude,table.replacement_margin_drop.abs()),3))
print('Top-3 overlap:',round(audit.topk_overlap(table.attention,table.gradient_magnitude),3))
plot_tokens(table,['gradient_magnitude','replacement_margin_drop'],'Sensitivity is a different question from pooling weight')
''')
md('''
## 5. Check the intervention itself

Replacing a word with the literal string `<UNK>` and re-tokenizing would split that string
into punctuation and letters. Instead, we replace exactly one ID with the unknown ID,
keeping sequence length and all other IDs fixed. We then **re-encode the whole sequence**.
Existing unknown tokens cannot be further changed by this replacement; their zero effect
does not establish that the original word was unimportant.

The intervention is replacement, not deletion. A deletion exercise is an optional alternative,
not a second name for the same operation. Probability changes and margin changes are both
reported because probabilities can saturate near 0 and 1.
''')
code('''
high=int(table.attention.argmax());low=int(table.attention.argmin())
display(table.iloc[[high,low]][['position','token','is_unknown','attention','replacement_margin_drop','replacement_probability_drop']])
x,lens,mask,tokens=tensors(first.sentence,vocab)
changed=x.clone();changed[0,high]=vocab[audit.UNK]
print('Original IDs:',x.tolist())
print('One-token replacement IDs:',changed.tolist())
assert changed.shape==x.shape
assert int((changed!=x).sum()) <= 1
''')
md(r'''
**Discuss:** does the higher-attention token always have the larger effect? If not, what
could explain the difference besides a broken implementation?

## 6. Hold the hidden states fixed and change attention

Now freeze the encoder output and change only the pooling weights: one uniform distribution
and 30 deterministic random permutations. These are internal interventions, not fluent edits
or proof that another naturally trained model would generate these weights.

Measure attention **total variation distance**, $TV=\frac12\sum_i|\alpha_i-\tilde\alpha_i|$,
alongside output changes. TV ranges from 0 to 1. A shuffled distribution is not necessarily
different; near-uniform attention can remain near-identical after shuffling.

A changed class is strong sensitivity evidence, but an unchanged class can conceal a large
score change. Conversely, a tiny probability change may conceal a substantial margin change.
''')
code('''
local_trials=attention_audit(model,first.sentence,vocab,seed=2026,n_shuffles=30)
display(local_trials.groupby('kind')[['attention_tv','abs_probability_change','abs_margin_change','label_changed']].agg(['mean','max']).round(4))
fig,ax=plt.subplots(figsize=(7,4))
for kind,g in local_trials.groupby('kind'):
    ax.scatter(g.attention_tv,g.abs_margin_change,label=kind,alpha=.7)
ax.set(xlabel='Attention TV distance',ylabel='Absolute target-margin change',title='Did a different heatmap actually change the score?');ax.legend()
plt.tight_layout();plt.show()
''')
md('''
## 7. Move from one example to a fixed cohort

The 120 validation IDs were selected with random seed 2026 before seeing explanation results.
All model seeds use those same IDs and permutation seeds. Below are saved outputs of the shared
audit functions, not invented teaching numbers. Rerun `run_case7_experiments.py` to regenerate
the full cohort; the notebook recomputes the detailed examples live.

For correlations, compare magnitudes rather than interpreting attention as signed support.
Spearman is undefined for constant vectors; report valid counts. Top-3 overlap is easier by
chance on short sentences: the independent random-set expectation is min(3,n)/n.
It is a reference, not a significance test.
''')
code('''
cols=['attention_gradient_rho','attention_replacement_rho','gradient_replacement_rho',
      'top3_overlap','top3_chance_overlap','uniform_tv','uniform_abs_dp','uniform_abs_dm','uniform_flip',
      'shuffle_mean_tv','shuffle_mean_abs_dp','shuffle_mean_abs_dm','shuffle_flip_rate']
display(aggregate[cols].agg(['count','mean','median','min','max']).T.round(4))
display(aggregate.groupby('correct')[['confidence','attention_gradient_rho','uniform_abs_dm','uniform_flip']].mean().round(4))
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].hist(aggregate.attention_gradient_rho.dropna(),bins=16,color='#356c9b')
axes[0].set(xlabel='Attention–gradient Spearman',ylabel='Examples',title='Agreement varies across examples')
points=axes[1].scatter(aggregate.uniform_tv,aggregate.uniform_abs_dm,c=aggregate.confidence,cmap='viridis',s=25)
fig.colorbar(points,ax=axes[1],label='Original predicted-class probability')
axes[1].set(xlabel='Uniform-attention TV distance',ylabel='Absolute margin change',title='One point per example; color = confidence')
plt.tight_layout();plt.show()
''')
md('''
**Interpretation checkpoint:** which observations support a role for attention? Which limit
the claim that its highlighted words are the full explanation? Use numerical results.
Thirty shuffles of the same sentence are not thirty independent sentences.

## 8. Inspect disagreement and a model error

We select the correctly classified audited example (6–35 tokens) with the lowest
attention–gradient correlation. This is a reproducible **extreme example**, not a representative
sample and not evidence that disagreement is typical. Then we inspect the highest-confidence
error of that length. Selection rules are explicit so they can be challenged.
''')
code('''
pool=aggregate[aggregate.correct & aggregate.n_tokens.between(6,35)].dropna(subset=['attention_gradient_rho'])
if len(pool):
    row=pool.sort_values(['attention_gradient_rho','idx']).iloc[0]
    ex=pred[pred.idx==row.idx].iloc[0]
    t,inf=token_audit(model,ex.sentence,vocab)
    print('Selected disagreement:',ex.sentence,'rho:',round(row.attention_gradient_rho,3))
    plot_tokens(t,['attention','gradient_magnitude'],'An extreme disagreement example, selected by rule')
else: print('No eligible disagreement example; do not fabricate one.')
errors=pred[(~pred.correct) & pred.n_tokens.between(6,35)].sort_values(['confidence','idx'],ascending=[False,True])
if len(errors):
    error=errors.iloc[0];et,ei=token_audit(model,error.sentence,vocab)
    print('Selected error:',error.sentence)
    print('Reference:',error.label,'prediction:',ei['target'],'confidence:',round(ei['target_probability'],4))
    display(et.sort_values('attention',ascending=False)[['position','token','attention','replacement_margin_drop']].head(8))
else: print('No error in the selected length range.')
''')
md('''
**Question:** can an explanation faithfully describe an incorrect prediction? Separate the
truth of the review label from the behavior of the model.

## 9. Language composition: a diagnostic slice and controlled probes

The cue-word slice (`not`, `never`, `but`, `however`, `although`, `though`, `yet`, `n't`)
is only a rough proxy for negation/contrast. Report its size and accuracy before choosing
an example. A correct sentence containing “not” does not establish negation understanding.

The following short pairs are **handwritten probes**, not SST-2 benchmark items. Their
purpose is to ask whether one intuitive wording change has the expected direction.
They are not a general language-understanding test. Discuss alternatives when human meaning
is ambiguous, and inspect unknown tokens before judging a failure.
''')
code('''
cue=pred.sentence.str.contains(r"\\b(?:not|never|but|however|although|though|yet)\\b|n't",case=False,regex=True)
display(pred.assign(cue_slice=cue).groupby('cue_slice').agg(n=('correct','size'),accuracy=('correct','mean')))
probes=['the film is good','the film is not good','the film is bad','the film is not bad',
        'the acting is good but the story is terrible','the acting is terrible but the story is good']
probe_rows=[]
for text in probes:
    t,inf=token_audit(model,text,vocab)
    pp=inf['target_probability'] if inf['target']==1 else 1-inf['target_probability']
    probe_rows.append({'text':text,'P(positive)':pp,'unknown_tokens':int(t.is_unknown.sum())})
display(pd.DataFrame(probe_rows))
''')
md(r'''
## 10. Optional: separate pooling weight from class contribution

Because this model has a linear classifier after pooling (dropout disabled for evaluation),
the target-class margin decomposes exactly as:

$$m=b_t-b_o+\sum_i\alpha_i (w_t-w_o)^T h_i.$$

The sign and size of the projected hidden state matter as well as attention. This is an
exact accounting identity **conditional on the hidden states**. It does not establish the
causal contribution of the isolated word, since each state is contextual.
''')
code('''
display(table[['position','token','attention','conditional_margin_contribution']].round(4))
reconstructed=table.conditional_margin_contribution.sum()+info['bias_margin']
print('Actual margin:',info['target_margin'],'reconstructed:',reconstructed)
assert np.isclose(reconstructed,info['target_margin'],atol=1e-5)
''')
md('''
## 11. Optional: stability across model seeds

All three models use the same vocabulary, data, four-epoch recipe, audit IDs, and intervention
seeds. The initialization and training shuffle vary. No “best explanation” seed is selected:
2026 is the classroom model fixed in advance. Three seeds give limited evidence of variation;
they do not establish robustness across architectures or datasets.
''')
code('''
seeds=pd.read_csv(OUT/'seed_summary.csv')
display(seeds[['seed','accuracy','attention_gradient_rho','attention_replacement_rho',
              'uniform_abs_dp','uniform_abs_dm','uniform_flip','shuffle_flip_rate']].round(4))
''')
md('''
## 12. Write the stakeholder verdict

| Claim | Evidence needed |
|---|---|
| Correctness | Agreement with reference labels, including errors |
| Confidence | Model score; no automatic claim of calibration |
| Plausibility | A human judgment, not established by model metrics alone |
| Faithfulness | Precisely specified diagnostics and interventions, with limitations |

Write 150–200 words for the hypothetical review platform:

1. State the claim you audited and name the model/data scope.
2. Cite two numerical results, including a distribution or across-seed observation.
3. Explain what supports the highlight **if such evidence was observed**.
4. Explain what limits it **if such evidence was observed**.
5. State one unresolved question and propose a test addressing it.
6. Recommend wording for the UI: for example “attention pooling weights” is a narrower
   claim than “the words that caused the decision.”

Do not force the verdict toward “attention is the explanation” or “attention is useless.”
An unchanged class alone is insufficient evidence of an unchanged prediction score.

**Exit ticket:** explain the difference between replacing an input token and changing attention
while holding hidden states fixed, in two sentences.

## Research lineage and curriculum alignment

This independent teaching adaptation follows the supplied deep-learning lecture's attention
questions (slides 6–12), gradient/leave-one-out comparison (19), alternative attention and
permutation tests (23–25), and contextual/architecture limitations (27–28).
Our UNK replacement is explicitly distinct from literal leave-one-out deletion.
We do not claim to reproduce the papers' architectures, datasets, or exact experimental protocols.

- Socher et al. (2013), [SST](https://aclanthology.org/D13-1170/).
- Jain & Wallace (2019), [Attention is not Explanation](https://aclanthology.org/N19-1357/).
- Wiegreffe & Pinter (2019), [Attention is not not Explanation](https://aclanthology.org/D19-1002/).

Read the latter two as a methodological debate. Our empirical verdict belongs to this run.
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
nb.write(book,ROOT/'07_attention_explanation_audit.ipynb')

metrics=pd.DataFrame(json.loads((OUT/'predictive_metrics.json').read_text()))
seeds=pd.read_csv(OUT/'seed_summary.csv');q=json.loads((OUT/'data_quality.json').read_text())
rows=['# Case 7 — Release findings','',
'These are measured outputs of the included fixed experiment, not expected answers invented before training.','',
'## Prediction results','',metrics.drop(columns='confusion_matrix').to_markdown(index=False,floatfmt='.4f'),'',
'## Explanation audit: means over the same 120 examples','',
seeds[['seed','attention_gradient_rho','attention_replacement_rho','gradient_replacement_rho','uniform_tv','uniform_abs_dp','uniform_abs_dm','uniform_flip','shuffle_flip_rate']].to_markdown(index=False,floatfmt='.4f'),'',
'TV measures attention change; dp is absolute probability change; dm is absolute target-logit-margin change. '
'Uniform flip is a sentence-level rate. Shuffle flip is the mean within-sentence flip rate across 30 permutations. '
'These are descriptive results, not independent trials for a significance test.','',
'## Interpretation','',
'Attention has measurable predictive influence in this setup, while attention, gradients, and token replacement '
'do not give identical rankings. A changed internal distribution can change output scores, so the results do not '
'justify saying that attention is irrelevant. Nor does sensitivity to attention prove that a highlighted visible word '
'is the unique cause: the hidden states contain context. Inspect the distributions and model errors in the notebook.','',
'The simple baseline is a substantive comparison. A neural model need not outperform it for the audit to teach '
'something; its performance also does not justify deployment. Probability confidence is not calibrated confidence.','',
'## Data and limits','',json.dumps(q,indent=2),'',
'One fixed 20,000-row training sample, one labeled validation set, three seeds, one small BiLSTM architecture. '
'No confirmatory holdout after case development. Training fragments can share source reviews; exact checks do not '
'resolve near duplication. UNK replacement can create unnatural inputs and cannot further change an already '
'unknown ID. The cohort is modest, and the detailed disagreement/error cases are intentionally extreme. '
'No human plausibility study, calibration study, production audit, transformer claim, or cross-dataset generalization.','',
'## Verification','',
'Targeted checks cover one-ID replacement, padding invariance, identical-attention override, a uniform-attention '
'null intervention, conditional contribution reconstruction, ranking behavior, and empty inputs. '
'The executed notebook is regenerated with captured outputs; see outputs/notebook_execution.json.']
(ROOT/'FINDINGS.md').write_text('\n'.join(rows))
print('Built notebook and release findings.')
