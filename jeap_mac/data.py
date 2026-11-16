"""Explicit synchronized RGB/3D-pose clips. Missing data is an error, never replaced."""
import json
from pathlib import Path
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES=False

class Clips(Dataset):
    def __init__(self,manifest,size=64):
        self.manifest=Path(manifest).resolve()
        self.rows=[json.loads(s) for s in self.manifest.read_text().splitlines() if s.strip()]
        if not self.rows:raise ValueError('Empty clip manifest')
        self.tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(.5,.5)])
        self.aug=transforms.Compose([transforms.RandomResizedCrop(size,scale=(.6,1.)),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),transforms.Normalize(.5,.5)])
        lengths=set()
        for r in self.rows:
            if r.get('coordinate_system')!='camera_meters':raise ValueError('Expected camera_meters coordinates')
            if r.get('fps',0)<=0:raise ValueError('FPS required')
            ids=r['frame_ids'];lengths.add(len(ids))
            if len(ids)!=len(r['frames']) or len(ids)<2 or any(b<=a for a,b in zip(ids,ids[1:])):raise ValueError('Invalid synchronized frame IDs')
            poses=torch.tensor(r['actions']);mask=torch.tensor(r['action_mask'])
            if poses.shape!=(len(ids)-1,17,3) or mask.shape!=(len(ids)-1,17):raise ValueError('Pose/mask alignment mismatch')
            if not torch.isfinite(poses).all() or not ((mask==0)|(mask==1)).all():raise ValueError('Invalid pose values or masks')
        if len(lengths)!=1:raise ValueError('All clips must have equal length')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];images=[]
        for name in r['frames']:
            with Image.open(self.manifest.parent/name) as im:
                im.load();images.append(im.convert('RGB'))
        return {'frames':torch.stack([self.tf(x) for x in images]),'view1':torch.stack([self.aug(x) for x in images[:-1]]),'view2':torch.stack([self.aug(x) for x in images[:-1]]),'actions':torch.tensor(r['actions'],dtype=torch.float32),'mask':torch.tensor(r['action_mask'],dtype=torch.float32)}
