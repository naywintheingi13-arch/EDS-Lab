"""Shared classroom/instructor diagnostics. All scores target the original prediction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import case7_attention_audit as a

CONFIG = dict(train_n=20000, max_len=80, epochs=4, batch=128,
              emb_dim=64, hidden_dim=48, dropout=0.25, audit_n=120, shuffles=30)

def tensors(text, vocab):
    ids, tokens = a.encode_text(text, vocab, CONFIG['max_len'])
    x = torch.tensor([ids]); lens = torch.tensor([len(ids)])
    return x, lens, torch.ones_like(x, dtype=torch.bool), tokens

def load_model(root, seed=2026):
    root = Path(root)
    vocab = json.loads((root/'outputs/vocab.json').read_text())
    model = a.BiLSTMAttention(len(vocab), **{k:CONFIG[k] for k in ['emb_dim','hidden_dim','dropout']})
    model.load_state_dict(torch.load(root/f'outputs/model_seed{seed}.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model, vocab

def token_audit(model, text, vocab):
    x,lens,mask,tokens = tensors(text,vocab)
    model.eval()
    with torch.no_grad():
        z,alpha,_,h = model(x,lens,mask,return_hidden=True)
        target=int(z.argmax(-1)); other=1-target
        margin=float(z[0,target]-z[0,other]); p=float(z.softmax(-1)[0,target])
        # Replace precisely one token ID; no text round-trip and no length change.
        replacements=x.repeat(len(tokens),1)
        replacements[torch.arange(len(tokens)),torch.arange(len(tokens))]=vocab[a.UNK]
        zr,_=model(replacements,lens.repeat(len(tokens)),mask.repeat(len(tokens),1))
        drops=margin-(zr[:,target]-zr[:,other])
        pdrops=p-zr.softmax(-1)[:,target]
        # Exact conditional decomposition of the class margin, at fixed hidden states.
        w=model.classifier.weight[target]-model.classifier.weight[other]
        contribution=alpha[0]*(h[0]@w)
        bias=model.classifier.bias[target]-model.classifier.bias[other]
        assert torch.allclose(contribution.sum()+bias,z[0,target]-z[0,other],atol=1e-5)
    grad=a.gradient_importance(model,x,lens,mask,target)
    table=pd.DataFrame({'position':range(len(tokens)), 'token':tokens,
        'token_id':x[0].numpy(), 'is_unknown':x[0].numpy()==vocab[a.UNK],
        'attention':alpha[0].numpy(), 'gradient_magnitude':grad,
        'replacement_margin_drop':drops.numpy(), 'replacement_probability_drop':pdrops.numpy(),
        'conditional_margin_contribution':contribution.numpy()})
    return table, {'target':target,'target_probability':p,'target_margin':margin,'bias_margin':float(bias)}

@torch.no_grad()
def attention_audit(model,text,vocab,seed=2026,n_shuffles=30):
    x,lens,mask,tokens=tensors(text,vocab);model.eval()
    _,h=model.encode(x,lens);z,alpha=model.classify_from_hidden(h,mask)
    target=int(z.argmax(-1));other=1-target
    p0=float(z.softmax(-1)[0,target]);m0=float(z[0,target]-z[0,other])
    variants=[('uniform',0,a.uniform_attention(mask))]
    variants += [('shuffle',i,a.shuffled_attention(alpha,mask,seed+i)) for i in range(n_shuffles)]
    rows=[]
    for kind,trial,alt in variants:
        za,_=model.classify_from_hidden(h,mask,alt)
        p=float(za.softmax(-1)[0,target]);margin=float(za[0,target]-za[0,other])
        rows.append(dict(kind=kind,trial=trial,attention_tv=float((alpha-alt).abs().sum()/2),
            probability_change=p-p0,abs_probability_change=abs(p-p0),
            margin_change=margin-m0,abs_margin_change=abs(margin-m0),
            label_changed=int(za.argmax(-1))!=target))
    return pd.DataFrame(rows)

def audit_cohort(model,frame,vocab,seed):
    rows=[];all_interventions=[]
    for row in frame.itertuples():
        t,info=token_audit(model,row.sentence,vocab)
        interventions=attention_audit(model,row.sentence,vocab,2026+int(row.idx)*100,CONFIG['shuffles'])
        interventions['idx']=row.idx;interventions['model_seed']=seed
        all_interventions.append(interventions)
        uniform=interventions.iloc[0]; sh=interventions.iloc[1:]
        rows.append(dict(idx=row.idx,label=row.label,prediction=info['target'],correct=info['target']==row.label,
            confidence=info['target_probability'],n_tokens=len(t),
            attention_gradient_rho=a.spearman(t.attention,t.gradient_magnitude),
            attention_replacement_rho=a.spearman(t.attention,t.replacement_margin_drop.abs()),
            gradient_replacement_rho=a.spearman(t.gradient_magnitude,t.replacement_margin_drop.abs()),
            top3_overlap=a.topk_overlap(t.attention,t.gradient_magnitude,3),
            top3_chance_overlap=min(3,len(t))/len(t),
            uniform_tv=uniform.attention_tv,uniform_abs_dp=uniform.abs_probability_change,
            uniform_abs_dm=uniform.abs_margin_change,uniform_flip=uniform.label_changed,
            shuffle_mean_tv=sh.attention_tv.mean(),shuffle_mean_abs_dp=sh.abs_probability_change.mean(),
            shuffle_mean_abs_dm=sh.abs_margin_change.mean(),shuffle_flip_rate=sh.label_changed.mean()))
    return pd.DataFrame(rows),pd.concat(all_interventions,ignore_index=True)
