"""Development-only diagnostic: center crop versus aspect-preserving resize/padding.

This changes scale and framing together; it is not an isolated causal intervention.
It does not alter the main model, feature caches, or reserved test result.
"""
import json,shutil
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision.transforms import functional as TF
import audit as a

def run(data):
    a.setup();data=a.Path(data)
    net=a.PetNet(a.ROOT/'weights/resnet18-f37072fd.pth')
    net.fit(np.load(a.ROOT/'cache/train_clean.npy'),pd.read_csv(a.ROOT/'cache/train_manifest.csv').y)
    selected=pd.read_csv(a.ROOT/'outputs/selected.csv');records=[]
    for row in selected.itertuples():
        source=data/f'images/{row.name}.jpg'
        shutil.copy2(source,a.ROOT/f'samples/{row.name}_original.jpg')
        with Image.open(source) as im:
            im=im.convert('RGB');w,h=im.size;scale=224/max(w,h)
            small=im.resize((round(w*scale),round(h*scale)),Image.Resampling.BILINEAR)
            z=torch.tensor(a.MEAN)[:,None,None].expand(3,224,224).clone()
            left=(224-small.width)//2;top=(224-small.height)//2
            z[:,top:top+small.height,left:left+small.width]=TF.to_tensor(small)
        with torch.no_grad():p=float(torch.sigmoid(net(z[None])).item())
        np.save(a.ROOT/f'samples/{row.name}_padded.npy',z.permute(1,2,0).numpy())
        records.append(dict(name=row.name,true_label=row.y,center_crop_p_dog=row.p_dog,padded_p_dog=p,
                            center_crop_prediction=row.prediction,padded_prediction=int(p>=.5)))
    pd.DataFrame(records).to_csv(a.ROOT/'outputs/preprocessing_check.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);run(p.parse_args().data)
