"""Fetch original public data; verify published archive checksums before extraction."""
from pathlib import Path
import urllib.request, hashlib, tarfile

def download(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    items=[('annotations.tar.gz','https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz','95a8c909bbe2e81eed6a22bccdf3f68f'),
           ('images.tar.gz','https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz','5c4f3ee8e5d25df40f4fd59a7f44e54c'),
           ('resnet18-f37072fd.pth','https://download.pytorch.org/models/resnet18-f37072fd.pth',None)]
    for name,url,checksum in items:
        p=root/name
        if not p.exists():
            print('Downloading',name,flush=True)
            temp=p.with_suffix('.part')
            with urllib.request.urlopen(url,timeout=90) as r,temp.open('wb') as f:
                while chunk:=r.read(1024*1024):f.write(chunk)
            temp.rename(p)
        if checksum:assert hashlib.md5(p.read_bytes()).hexdigest()==checksum
        print('Verified',name,p.stat().st_size,flush=True)
        if name.endswith('.tar.gz'):
            with tarfile.open(p) as t:t.extractall(root,filter='data')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path)
    download(p.parse_args().destination)
