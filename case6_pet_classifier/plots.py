from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs'

def data():
    rows=pd.read_csv(OUT/'selected.csv')
    return rows,{n:dict(np.load(OUT/f'maps_{n}.npz')) for n in rows.name}

def heat(ax,rgb,a,title):
    ax.imshow(rgb)
    if a.max()>1e-12:ax.imshow(a,cmap='magma',alpha=.6,vmin=0,vmax=a.max())
    else:ax.text(.5,.1,'ZERO MAP',ha='center',transform=ax.transAxes,color='white',backgroundcolor='black')
    ax.set_title(title,fontsize=9);ax.axis('off')

def maps_figure():
    rows,maps=data();fig,ax=plt.subplots(4,5,figsize=(14,12))
    for i,r in rows.iterrows():
        m=maps[r['name']];ax[i,0].imshow(m['rgb']);ax[i,0].axis('off')
        ax[i,0].set_title(f"{r['name']}\nTrue {['cat','dog'][r.y]} / predicted {['cat','dog'][r.prediction]}\np(dog)={r.p_dog:.3f}",fontsize=8)
        for j,(key,title) in enumerate([('gradient','Gradient magnitude'),('cam','Grad-CAM: 7×7 enlarged'),('ig_mean','IG: mean baseline'),('ig_blur','IG: blur baseline')],1):heat(ax[i,j],m['rgb'],m[key],title)
    fig.suptitle('Plausible does not mean faithful\nFixed original-prediction margin; colors independently scaled for spatial shape',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));return fig

def signed_figure():
    rows,maps=data();fig,ax=plt.subplots(4,3,figsize=(10,12),layout='constrained')
    for i,r in rows.iterrows():
        m=maps[r['name']];colors=np.array([[0,0,0],[.2,.65,.4],[.8,.8,.8],[1,.8,.2]])
        ax[i,0].imshow(colors[m['mask']]);ax[i,0].set_title(r['name']+'\nGreen animal / gray background / yellow boundary',fontsize=8)
        lim=max(np.quantile(np.abs(np.concatenate([m['signed_mean'].ravel(),m['signed_blur'].ravel()])),.99),1e-12)
        for j,b in enumerate(['mean','blur'],1):
            im=ax[i,j].imshow(m[f'signed_{b}'],cmap='RdBu_r',vmin=-lim,vmax=lim);ax[i,j].set_title('Signed IG: '+b)
        fig.colorbar(im,ax=list(ax[i,1:]),shrink=.6)
        for a in ax[i]:a.axis('off')
    fig.suptitle('Reference and sign matter\nRed supports target; blue opposes; shared row scale clipped at 99th percentile')
    return fig

def marker_performance():
    d=pd.DataFrame(json.loads((OUT/'marker_performance.json').read_text()));fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(3)
    for i,(key,label) in enumerate([('clean','Clean-trained'),('marker_trained','Marker-trained')]):
        vals=d[d.model==key].set_index('condition').loc[['clean','aligned','swapped']].balanced_accuracy*100
        bars=ax.bar(x+(i-.5)*.34,vals,.34,label=label);ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.set_xticks(x,['No marker','Aligned marker','Swapped marker']);ax.set_ylim(0,110)
    ax.axhline(50,color='gray',ls=':');ax.set_ylabel('Balanced accuracy (%)');ax.legend(loc='lower left')
    ax.set_title('Designed marker challenge: same 370 development images\n100% label correlation in marked training; 2.04% image area')
    fig.tight_layout();return fig

def sanity_figure():
    rows,maps=data();m=maps[rows.iloc[0]['name']];fig,ax=plt.subplots(2,3,figsize=(10,6))
    for j,(p,label) in enumerate([('','Trained'),('head_','Random head'),('full_','Random whole network')]):
        for i,key in enumerate(['gradient','cam']):heat(ax[i,j],m['rgb'],m[p+key],label+'\n'+key)
    fig.suptitle('Model dependence: same image and target\nSeed 11 shown; seeds 11, 22 and 33 measured; independently scaled maps')
    fig.tight_layout(rect=(0,0,1,.91));return fig

def removal_figure():
    d=pd.read_csv(OUT/'perturbations.csv');fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,name in zip(axes.ravel(),d.name.unique()):
        s=d[d.name==name]
        for i,fill in enumerate(['mean','blur']):
            r=s[s.fill==fill].set_index('region');random=r.loc[r.index.str.startswith('random'),'drop']
            vals=[r.loc['top10','drop'],r.loc['bottom10','drop'],random.mean()]
            ax.bar(np.arange(3)+(i-.5)*.34,vals,.34,yerr=[0,0,random.std()],capsize=3,label=fill)
        ax.set_xticks(range(3),['CAM top 10%','CAM bottom 10%','Random 10%']);ax.axhline(0,color='gray',lw=1)
        ax.set_ylabel('Original target-margin drop');ax.set_title(name,fontsize=9);ax.legend(fontsize=8)
    fig.suptitle('Does removal support the visual impression?\nEqual pixel counts; random error bars are SD across 10 masks, not confidence intervals')
    fig.tight_layout(rect=(0,0,1,.92));return fig

def localization():
    d=pd.DataFrame(json.loads((OUT/'attributions.json').read_text()));names=d.name.unique();x=np.arange(4);fig,ax=plt.subplots(figsize=(10,4.5))
    for i,m in enumerate(['gradient','GradCAM','IG_mean','IG_blur']):
        ax.bar(x+(i-1.5)*.17,d[d.method==m].set_index('name').loc[names].foreground_share,.17,label=m)
    ax.scatter(x,d.groupby('name').area_share.first().loc[names],marker='_',s=500,color='black',label='Animal area share',zorder=4)
    ax.set_xticks(x,names,rotation=10,ha='right',fontsize=8);ax.set_ylim(0,1.05);ax.set_ylabel('Foreground mass share among known pixels')
    ax.set_title('Localization is not explanation ground truth\nUncertain boundary excluded; four diagnostic examples');ax.legend(fontsize=8,ncol=3)
    fig.tight_layout();return fig

def preprocessing():
    from PIL import Image
    rows,maps=data();d=pd.read_csv(OUT/'preprocessing_check.csv').set_index('name')
    errors=rows[~rows.correct];fig,ax=plt.subplots(len(errors),3,figsize=(10,7),squeeze=False)
    for i,r in enumerate(errors.itertuples()):
        original=Image.open(ROOT/f'samples/{r.name}_original.jpg')
        ax[i,0].imshow(original);ax[i,0].set_title(r.name+'\nOriginal photograph',fontsize=9)
        ax[i,1].imshow(maps[r.name]['rgb']);ax[i,1].set_title(f'Center crop: p(dog)={r.p_dog:.3f}')
        ax[i,2].imshow(np.load(ROOT/f'samples/{r.name}_padded.npy'))
        ax[i,2].set_title(f'Fit + mean padding: p(dog)={d.loc[r.name,"padded_p_dog"]:.3g}')
        for cell in ax[i]:cell.axis('off')
    fig.suptitle('Audit the input before interpreting the explanation\nDevelopment-only follow-up; framing, scale and padding change together')
    fig.tight_layout(rect=(0,0,1,.91));return fig

def save():
    for name,fn in [('01_maps',maps_figure),('02_signed',signed_figure),('03_marker',marker_performance),('04_sanity',sanity_figure),('05_removal',removal_figure),('06_localization',localization),('07_preprocessing',preprocessing)]:
        fig=fn();fig.savefig(OUT/f'{name}.png',dpi=120,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':save()
