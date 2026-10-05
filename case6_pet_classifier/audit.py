"""Case 6: frozen ResNet18 + binary logistic head and explicit explanation tests."""
from pathlib import Path
import copy,json,hashlib,time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import functional as TF,InterpolationMode
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,log_loss
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
SEED=2026
MEAN=[.485,.456,.406];STD=[.229,.224,.225]

def setup():
    torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def splits(data):
    def read(name):
        return pd.read_csv(Path(data)/f'annotations/{name}.txt',sep=r'\s+',comment='#',header=None,
                           names=['name','breed','species','within']).assign(y=lambda d:d.species-1)[['name','breed','y']]
    rng=np.random.default_rng(SEED);parts={s:[] for s in ['train','dev','test']}
    tv,test=read('trainval'),read('test')
    assert not set(tv.name)&set(test.name)
    for _,g in tv.groupby('breed',sort=True):
        g=g.iloc[rng.permutation(len(g))];parts['train'].append(g.iloc[:30]);parts['dev'].append(g.iloc[30:40])
    for _,g in test.groupby('breed',sort=True):parts['test'].append(g.iloc[rng.permutation(len(g))[:10]])
    return {k:pd.concat(v).reset_index(drop=True) for k,v in parts.items()}

def load_image(data,name):
    with Image.open(Path(data)/f'images/{name}.jpg') as im:
        x=TF.to_tensor(TF.center_crop(TF.resize(im.convert('RGB'),256,InterpolationMode.BILINEAR,antialias=True),224))
    with Image.open(Path(data)/f'annotations/trimaps/{name}.png') as im:
        mask=np.array(TF.center_crop(TF.resize(im,256,InterpolationMode.NEAREST),224)).copy()
    assert set(np.unique(mask))<={1,2,3}
    return x,mask

def marker(x,y):
    z=x.clone();colors=torch.tensor([[1.,0,0],[0.,0,1]],dtype=x.dtype)
    z[:,:,:32,:32]=colors[torch.as_tensor(y)][:,:,None,None]
    return z

class PetNet(nn.Module):
    def __init__(self,weights=None):
        super().__init__();self.backbone=models.resnet18(weights=None)
        if weights:self.backbone.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
        self.backbone.fc=nn.Identity();self.head=nn.Linear(512,1)
        self.register_buffer('mean',torch.tensor(MEAN)[None,:,None,None])
        self.register_buffer('std',torch.tensor(STD)[None,:,None,None]);self.eval()
    def features(self,x):return self.backbone((x-self.mean)/self.std)
    def forward(self,x):return self.head(self.features(x)).flatten()
    def score(self,x,target):return self(x)*(1 if int(target)==1 else -1)
    def fit(self,X,y):
        lr=LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=SEED).fit(X,y)
        assert lr.classes_.tolist()==[0,1]
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(lr.coef_,dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(lr.intercept_,dtype=torch.float32))
        np.testing.assert_allclose(self.head(torch.tensor(X)).detach().numpy().ravel(),lr.decision_function(X),rtol=1e-5,atol=1e-5)
        return lr

def extract(net,data,rows,condition):
    result=[]
    for i in range(0,len(rows),24):
        r=rows.iloc[i:i+24];x=torch.stack([load_image(data,n)[0] for n in r.name])
        if condition!='clean':x=marker(x,r.y.to_numpy() if condition=='aligned' else 1-r.y.to_numpy())
        with torch.no_grad():result.append(net.features(x).numpy())
        if i%240==0:print(condition,min(i+24,len(rows)),'/',len(rows),flush=True)
    return np.concatenate(result)

def evaluate(net,X,rows):
    with torch.no_grad():z=net.head(torch.tensor(X)).flatten().numpy()
    p=1/(1+np.exp(-z));pred=(z>=0).astype(int)
    m=dict(n=len(rows),accuracy=float(accuracy_score(rows.y,pred)),balanced_accuracy=float(balanced_accuracy_score(rows.y,pred)),
           log_loss=float(log_loss(rows.y,np.c_[1-p,p],labels=[0,1])),confusion_matrix=confusion_matrix(rows.y,pred,labels=[0,1]).tolist())
    r=rows.copy();r['p_dog']=p;r['prediction']=pred;r['confidence']=np.maximum(p,1-p);r['correct']=pred==r.y
    return m,r

def select(pred):
    chosen=[]
    for y in [0,1]:
        r=pred[pred.y==y];correct=r[r.correct].sort_values(['confidence','name'],ascending=[False,True])
        if len(correct):chosen.append(correct.iloc[0]['name'])
        wrong=r[~r.correct].sort_values(['confidence','name'],ascending=[False,True])
        if len(wrong):chosen.append(wrong.iloc[0]['name'])
        else:chosen.append(r[~r.name.isin(chosen)].sort_values(['confidence','name']).iloc[0]['name'])
    return pred.set_index('name').loc[chosen].reset_index()

def gradient(net,x,target):
    x=x.detach().clone().requires_grad_(True)
    return torch.autograd.grad(net.score(x,target).sum(),x)[0].detach()

def gradcam(net,x,target):
    acts=[];hook=net.backbone.layer4.register_forward_hook(lambda m,i,o:acts.append(o))
    try:
        s=net.score(x.detach().clone().requires_grad_(True),target);a=acts[0]
        g=torch.autograd.grad(s.sum(),a)[0]
        raw=(g.mean((2,3),keepdim=True)*a).sum(1,keepdim=True)
        w=net.head.weight.reshape(1,512,1,1)*(1 if target==1 else -1)
        assert torch.allclose(raw,(w*a).sum(1,keepdim=True)/49,atol=1e-5,rtol=1e-4)
        return F.interpolate(raw.relu(),size=x.shape[-2:],mode='bilinear',align_corners=False).detach()
    finally:hook.remove()

def ig(net,x,b,target,steps=64,method='trapezoid'):
    total=torch.zeros_like(x)
    if method=='gausslegendre':
        nodes,weights=np.polynomial.legendre.leggauss(steps)
        alphas=torch.tensor((nodes+1)/2,dtype=x.dtype)
        quadrature=torch.tensor(weights/2,dtype=x.dtype)
    else:
        alphas=torch.linspace(0,1,steps+1);quadrature=torch.ones(steps+1)/steps
        quadrature[0]*=.5;quadrature[-1]*=.5
    for start in range(0,len(alphas),16):
        ids=torch.arange(start,min(start+16,len(alphas)))
        path=(b+alphas[ids,None,None,None]*(x-b)).detach().requires_grad_(True)
        g=torch.autograd.grad(net.score(path,target).sum(),path)[0]
        total+=(g*quadrature[ids,None,None,None]).sum(0,keepdim=True)
    a=(x-b)*total
    with torch.no_grad():delta=float((net.score(x,target)-net.score(b,target)).item())
    residual=float(a.sum().item()-delta)
    return a.detach(),dict(steps=steps,method_integration=method,score_delta=delta,residual=residual,passes=abs(residual)<=.02+.02*abs(delta))

def checked_ig(net,x,b,target):
    history=[]
    for n in [64,128,256,512,1024]:
        a,d=ig(net,x,b,target,n);history.append(d.copy())
        if d['passes']:break
    if not d['passes']:
        for n in [256,512,1024]:
            a,d=ig(net,x,b,target,n,method='gausslegendre');history.append(d.copy())
            if d['passes']:break
    return a,dict(**d,attempts=history)

def references(x):return {'mean':torch.tensor(MEAN)[None,:,None,None].expand_as(x).clone(),'blur':TF.gaussian_blur(x,[51,51],[12.,12.])}
def spatial(a):return a.abs().sum(1)[0].numpy()
def corr(a,b):return None if np.ptp(a)<1e-12 or np.ptp(b)<1e-12 else float(spearmanr(a.ravel(),b.ravel()).statistic)
def overlap(a,mask):
    mass=float(a[mask!=3].sum());area=float((mask==1).sum()/(mask!=3).sum())
    share=float(a[mask==1].sum()/mass) if mass>1e-12 else None
    return dict(foreground_share=share,area_share=area,enrichment=share/area if share is not None and area else None)

def removal(net,x,target,mask,cam):
    rng=np.random.default_rng(SEED);k=int(.1*224*224);rank=np.argsort(cam.ravel(),kind='stable')
    regions={'foreground':mask==1,'background':mask==2}
    for label,ids in [('top10',rank[-k:]),('bottom10',rank[:k])]+[(f'random10_{i}',rng.choice(224*224,k,replace=False)) for i in range(10)]:
        m=np.zeros(224*224,dtype=bool);m[ids]=True;regions[label]=m.reshape(224,224)
    with torch.no_grad():original=float(net.score(x,target).item())
    result=[]
    for fill,b in references(x).items():
        for region,m in regions.items():
            changed=torch.where(torch.tensor(m)[None,None],b,x)
            with torch.no_grad():score=float(net.score(changed,target).item())
            result.append(dict(fill=fill,region=region,area_fraction=float(m.mean()),original=original,altered=score,drop=original-score))
    return result

def unit_test():
    class Linear:
        def score(self,x,target):return (x*torch.tensor([1.,-2.,3.])[None,:,None,None]).sum((1,2,3))+7
    x=torch.linspace(0,1,12).reshape(1,3,2,2);a,d=ig(Linear(),x,torch.zeros_like(x),1,8)
    assert torch.allclose(a,x*torch.tensor([1.,-2.,3.])[None,:,None,None],atol=1e-6)
    assert abs(d['residual'])<2e-6
    gauss,gd=ig(Linear(),x,torch.zeros_like(x),1,8,method='gausslegendre')
    assert torch.allclose(gauss,a,atol=1e-6) and abs(gd['residual'])<2e-6
    print('PASS: linear-model IG values and completeness',flush=True)

def run(data):
    setup();unit_test();data=Path(data)
    out=ROOT/'outputs';cache=ROOT/'cache';samples=ROOT/'samples'
    for p in [out,cache,samples]:p.mkdir(parents=True,exist_ok=True)
    rows=splits(data);net=PetNet(data/'resnet18-f37072fd.pth');features={}
    for s,r in rows.items():
        r.to_csv(cache/f'{s}_manifest.csv',index=False)
        for c in (['clean','aligned'] if s=='train' else ['clean','aligned','swapped'] if s=='dev' else ['clean']):
            p=cache/f'{s}_{c}.npy'
            features[f'{s}_{c}']=np.load(p) if p.exists() else extract(net,data,r,c)
            np.save(p,features[f'{s}_{c}'])
    net.fit(features['train_clean'],rows['train'].y)
    dev,pred=evaluate(net,features['dev_clean'],rows['dev']);pred.to_csv(out/'development_predictions.csv',index=False)
    selected=select(pred);selected.to_csv(out/'selected.csv',index=False)
    shortcut=copy.deepcopy(net);shortcut.fit(features['train_aligned'],rows['train'].y)
    performance=[]
    for label,model in [('clean',net),('marker_trained',shortcut)]:
        for c in ['clean','aligned','swapped']:
            m,t=evaluate(model,features[f'dev_{c}'],rows['dev']);performance.append(dict(model=label,condition=c,**m))
            t.to_csv(out/f'{label}_{c}_predictions.csv',index=False)
    (out/'marker_performance.json').write_text(json.dumps(performance,indent=2))
    print('DEV',dev,'\nSELECTED',selected.to_string(index=False),flush=True)
    diag=[];perturb=[];sanity=[]
    for row in selected.itertuples():
        name=row.name;target=int(row.prediction);xx,mask=load_image(data,name);x=xx[None]
        np.savez_compressed(samples/f'{name}.npz',rgb=x.numpy(),mask=mask)
        print('Explaining',name,flush=True)
        g=spatial(gradient(net,x,target));cam=gradcam(net,x,target)[0,0].numpy()
        maps=dict(rgb=xx.permute(1,2,0).numpy(),mask=mask,gradient=g,cam=cam)
        for b,reference in references(x).items():
            a,d=checked_ig(net,x,reference,target);maps[f'ig_{b}']=spatial(a);maps[f'signed_{b}']=a.sum(1)[0].numpy()
            diag.append(dict(name=name,method=f'IG_{b}',**d,**overlap(spatial(a),mask)))
        for method,a in [('gradient',g),('GradCAM',cam)]:diag.append(dict(name=name,method=method,**overlap(a,mask)))
        perturb.extend([dict(name=name,**r) for r in removal(net,x,target,mask,cam)])
        for seed in [11,22,33]:
            for stage in ['head','full']:
                torch.manual_seed(seed);rand=copy.deepcopy(net) if stage=='head' else PetNet()
                if stage=='head':rand.head.reset_parameters()
                rand.eval();rg=spatial(gradient(rand,x,target));rc=gradcam(rand,x,target)[0,0].numpy()
                sanity.append(dict(name=name,seed=seed,stage=stage,gradient_spearman=corr(g,rg),cam_spearman=corr(cam,rc),cam_zero=bool(rc.max()<1e-12)))
                if seed==11:maps[f'{stage}_gradient']=rg;maps[f'{stage}_cam']=rc
        for c,z in [('clean',x),('aligned',marker(x,[row.y])),('swapped',marker(x,[1-row.y]))]:
            maps[f'marker_{c}_rgb']=z[0].permute(1,2,0).numpy();maps[f'marker_{c}_cam']=gradcam(shortcut,z,target)[0,0].numpy()
            with torch.no_grad():maps[f'marker_{c}_p']=np.array(torch.sigmoid(shortcut(z)).item())
        np.savez_compressed(out/f'maps_{name}.npz',**maps)
    (out/'attributions.json').write_text(json.dumps(diag,indent=2,allow_nan=False))
    pd.DataFrame(perturb).to_csv(out/'perturbations.csv',index=False);pd.DataFrame(sanity).to_csv(out/'randomization.csv',index=False)
    final,t=evaluate(net,features['test_clean'],rows['test']);t.to_csv(out/'test_predictions.csv',index=False)
    # Exact byte duplicates; this does NOT screen near duplicates or pretraining overlap.
    hashes={};duplicates=[]
    for split,r in rows.items():
        for name in r.name:
            h=hashlib.sha256((data/f'images/{name}.jpg').read_bytes()).hexdigest()
            if h in hashes and hashes[h][0]!=split:duplicates.append([hashes[h],[split,name]])
            hashes[h]=[split,name]
    summary=dict(seed=SEED,development=dev,test=final,majority_test=float(rows['test'].y.mean()),
                 cross_split_exact_duplicates=duplicates,weights_sha256=hashlib.sha256((data/'resnet18-f37072fd.pth').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',summary,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);run(p.parse_args().data)
