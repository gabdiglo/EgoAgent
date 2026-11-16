"""Evaluate a real joint-trained checkpoint. Rejects the visual-only probe checkpoint."""
import os
from pathlib import Path
os.environ.setdefault('TRANSFORMERS_CACHE',str(Path(__file__).parent/'cache'))
import argparse,json
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import JEAP,Config
from data import Clips

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ckpt.get('training_mode')!='joint_real_RGB_3D_pose':raise ValueError('A genuinely joint-trained checkpoint is required')
    m=JEAP(Config(**ckpt['configuration']));m.load_state_dict(ckpt['state_dict']);m.eval();ds=Clips(a.manifest,m.c.image_size)
    sums={'state_cosine':0.,'persistence_state_cosine':0.,'action_error_m':0.,'persistence_action_error_m':0.};states=0;joints=0
    with torch.inference_mode():
        for b in DataLoader(ds,batch_size=2,shuffle=False):
            x=b['frames'];out=m(x[:,:-1],b['actions'],b['mask'])
            future,_=m.observer.encode(x[:,1:].flatten(0,1));current,_=m.observer.encode(x[:,:-1].flatten(0,1))
            sums['state_cosine']+=F.cosine_similarity(out['state_logits'].flatten(0,1),future).sum().item()
            sums['persistence_state_cosine']+=F.cosine_similarity(current,future).sum().item();states+=len(future)
            # Match model and previous-pose baseline on exactly the same visible joints.
            mask=b['mask'][:,1:]*b['mask'][:,:-1]
            sums['action_error_m']+=((out['actions'][:,1:]-b['actions'][:,1:]).norm(dim=-1)*mask).sum().item()
            sums['persistence_action_error_m']+=((b['actions'][:,:-1]-b['actions'][:,1:]).norm(dim=-1)*mask).sum().item();joints+=mask.sum().item()
    if states==0 or joints==0:raise ValueError('Insufficient evaluated states/joints')
    result={'evaluation':'local mini-JEAP; not the original paper retrieval benchmark','state_embedding':'DINO-head logits','states':states,'visible_joint_comparisons':joints,'state_cosine':sums['state_cosine']/states,'persistence_state_cosine':sums['persistence_state_cosine']/states,'MPJPE_cm':100*sums['action_error_m']/joints,'persistence_MPJPE_cm':100*sums['persistence_action_error_m']/joints}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
