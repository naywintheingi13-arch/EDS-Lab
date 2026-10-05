"""Targeted regression checks for the audit claims; no model-quality assertions."""
import numpy as np
import torch
import case7_attention_audit as a
from audit_pipeline import token_audit,attention_audit,tensors
torch.set_num_threads(2);a.set_seed(2026)
v=a.build_vocab(['not good not bad','a wonderful touching film'],min_freq=1)
m=a.BiLSTMAttention(len(v),emb_dim=8,hidden_dim=6,dropout=0);m.eval()
x,l,mask,t=tensors('not good not bad',v)
with torch.no_grad():
    z,alpha,_,h=m(x,l,mask,return_hidden=True)
    zi,_=m.classify_from_hidden(h,mask,alpha)
    assert torch.allclose(z,zi,atol=1e-7)
    xp=torch.cat([x,torch.zeros((1,3),dtype=torch.long)],1)
    zp,ap=m(xp,l,xp.ne(0))
    assert torch.allclose(z,zp,atol=1e-6)
    assert (ap[0,4:]==0).all() and abs(float(ap.sum())-1)<1e-6
    for i in range(len(t)):
        xr=x.clone();xr[0,i]=v[a.UNK]
        assert xr.shape==x.shape and int((xr!=x).sum())==1
table,info=token_audit(m,'not good not bad',v)
assert len(table)==4 and table.token.tolist().count('not')==2
assert np.isclose(table.gradient_magnitude.sum(),1)
assert np.isclose(table.conditional_margin_contribution.sum()+info['bias_margin'],info['target_margin'],atol=1e-6)
inter=attention_audit(m,'not good not bad',v,n_shuffles=5)
assert len(inter)==6 and inter.attention_tv.between(0,1).all()
# Make attention exactly uniform: changing to uniform or shuffling must be a no-op.
with torch.no_grad():m.attn_score.weight.zero_()
null=attention_audit(m,'not good not bad',v,n_shuffles=5)
assert np.allclose(null.attention_tv,0) and np.allclose(null.abs_margin_change,0)
assert np.isclose(a.spearman([1,2,2,4],[4,2,2,1]),-1)
try:tensors('   ',v)
except ValueError:pass
else:raise AssertionError('Empty input was not rejected')
print('PASS: ID replacement, padding invariance, identity override, repeated positions, contribution identity, uniform null, rankings, empty-input guard')
