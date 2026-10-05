"""Reproduce the release: fixed recipe, same data/cohort across three seeds."""
from pathlib import Path
import json,time,hashlib,platform,subprocess,sys
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss,confusion_matrix
from torch.utils.data import DataLoader
import case7_attention_audit as a
from audit_pipeline import CONFIG,audit_cohort

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
if not (ROOT/'data/sst2_train.csv').exists():
    subprocess.run([sys.executable,str(ROOT/'prepare_sst2_course_data.py')],check=True)
train_raw=pd.read_csv(ROOT/'data/sst2_train.csv');val=pd.read_csv(ROOT/'data/sst2_validation.csv')
train_raw['source_idx']=np.arange(len(train_raw));val['idx']=np.arange(len(val))
# Normalize exactly as the tokenizer does for duplicate checks; preserve original text.
key=lambda s:' '.join(a.simple_tokenize(s))
train_raw['key']=train_raw.sentence.map(key);val['key']=val.sentence.map(key)
conflicts=train_raw.groupby('key').label.nunique();conflict_keys=set(conflicts[conflicts>1].index)
clean=train_raw[~train_raw.key.isin(conflict_keys)].drop_duplicates('key').copy()
overlap=set(clean.key)&set(val.key)
clean=clean[~clean.key.isin(overlap)]
selected,_=train_test_split(clean,train_size=CONFIG['train_n'],stratify=clean.label,random_state=2026)
selected=selected.sort_values('source_idx').reset_index(drop=True)
selected[['source_idx','sentence','label']].to_csv(OUT/'training_subset.csv',index=False)
vocab=a.build_vocab(selected.sentence,min_freq=2,max_size=20000)
(OUT/'vocab.json').write_text(json.dumps(vocab))
# Fixed audit cohort chosen before explanation results and shared across seeds.
cohort=val.sample(CONFIG['audit_n'],random_state=2026).sort_values('idx')
cohort[['idx','sentence','label']].to_csv(OUT/'audit_cohort.csv',index=False)
quality=dict(original_train=len(train_raw),original_validation=len(val),
    train_duplicate_rows=int(train_raw.key.duplicated().sum()),conflicting_train_keys=len(conflict_keys),
    cross_split_exact_keys_removed=len(overlap),validation_duplicate_rows=int(val.key.duplicated().sum()),
    clean_training_pool=len(clean),selected_training_rows=len(selected),vocab_size=len(vocab),
    validation_unknown_rate=sum(t not in vocab for s in val.sentence for t in a.simple_tokenize(s)[:80])/sum(len(a.simple_tokenize(s)[:80]) for s in val.sentence),
    train_truncation_rate=float(selected.sentence.map(lambda s:len(a.simple_tokenize(s))>80).mean()),
    validation_truncation_rate=float(val.sentence.map(lambda s:len(a.simple_tokenize(s))>80).mean()))
(OUT/'data_quality.json').write_text(json.dumps(quality,indent=2))
metrics=[]
def report(name,y,p):
    pred=np.asarray(p)>=.5
    return dict(model=name,accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),
        log_loss=log_loss(y,np.column_stack([1-p,p])),confusion_matrix=confusion_matrix(y,pred).tolist())
majority=float(selected.label.mean());metrics.append(report('majority / class prior',val.label,np.full(len(val),majority)))
baseline=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=40000),LogisticRegression(C=1,max_iter=1000,random_state=2026))
baseline.fit(selected.sentence,selected.label);p=baseline.predict_proba(val.sentence)[:,1]
metrics.append(report('TF-IDF logistic regression',val.label,p))
pd.DataFrame({'idx':val.idx,'label':val.label,'p_positive':p}).to_csv(OUT/'baseline_predictions.csv',index=False)
histories={};summaries=[];start=time.time()
for seed in [2026,2027,2028]:
    a.set_seed(seed)
    model=a.BiLSTMAttention(len(vocab),**{k:CONFIG[k] for k in ['emb_dim','hidden_dim','dropout']})
    loader=DataLoader(a.SSTDataset(selected,vocab,80),batch_size=CONFIG['batch'],shuffle=True,collate_fn=a.make_collate())
    vloader=DataLoader(a.SSTDataset(val,vocab,80),batch_size=256,shuffle=False,collate_fn=a.make_collate())
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    histories[str(seed)]=[]
    for epoch in range(CONFIG['epochs']):
        result=a.train_one_epoch(model,loader,opt);histories[str(seed)].append(dict(epoch=epoch+1,**result))
        print(seed,epoch+1,result,'elapsed',round(time.time()-start),flush=True)
    pred=a.evaluate(model,vloader)
    metrics.append(report(f'BiLSTM attention seed {seed}',val.label,pred.p_positive.values))
    torch.save(model.state_dict(),OUT/f'model_seed{seed}.pt')
    pred.to_json(OUT/f'validation_predictions_seed{seed}.jsonl',orient='records',lines=True)
    agg,inter=audit_cohort(model,cohort,vocab,seed)
    agg.to_csv(OUT/f'audit_seed{seed}.csv',index=False);inter.to_csv(OUT/f'interventions_seed{seed}.csv',index=False)
    summaries.append(dict(seed=seed,accuracy=float(pred.correct.mean()),**agg.select_dtypes(include=['number','bool']).drop(columns=['idx','label','prediction']).mean().to_dict()))
    (OUT/'training_history.json').write_text(json.dumps(histories,indent=2))
    (OUT/'predictive_metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame(summaries).to_csv(OUT/'seed_summary.csv',index=False)
    print('AUDIT COMPLETE',seed,flush=True)
manifest=dict(config=CONFIG,seeds=[2026,2027,2028],seconds=time.time()-start,
    python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
    source_url='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    source_sha256=hashlib.sha256((ROOT/'data/SST-2.zip').read_bytes()).hexdigest())
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
print(pd.DataFrame(summaries).to_string(index=False),flush=True)
