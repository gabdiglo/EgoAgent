"""Train all three JEAP objectives from real synchronized RGB/pose manifests."""
import os
from pathlib import Path
os.environ.setdefault('TRANSFORMERS_CACHE',str(Path(__file__).parent/'cache'))
import argparse,json,random,time
import torch
from torch.utils.data import DataLoader
from model import JEAP,Config
from data import Clips

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--val',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'runs/joint')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed)
    train=Clips(args.train);val=Clips(args.val)
    if {x['take_uid'] for x in train.rows}&{x['take_uid'] for x in val.rows}:raise ValueError('Train/validation takes overlap')
    m=JEAP().to(args.device);opt=torch.optim.AdamW([x for x in m.parameters() if x.requires_grad],lr=args.lr)
    args.output.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');start=time.perf_counter()
    for epoch in range(args.epochs):
        summary={'epoch':epoch+1}
        for name,ds in [('train',train),('val',val)]:
            m.train(name=='train');sums={k:0. for k in ['total','representation','prediction','action']};n=0
            # Fix validation augmentation randomness each epoch for comparable loss.
            if name=='val':torch.manual_seed(12345);random.seed(12345)
            loader=DataLoader(ds,batch_size=args.batch_size,shuffle=name=='train',num_workers=0)
            for batch in loader:
                batch={k:v.to(args.device) for k,v in batch.items()}
                with torch.set_grad_enabled(name=='train'):
                    loss,parts,teacher=m.losses(batch['frames'],batch['view1'],batch['view2'],batch['actions'],batch['mask'])
                    if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                    if name=='train':
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.requires_grad],1.);opt.step();m.update_observer(teacher)
                b=len(batch['frames']);n+=b
                for k,v in dict(parts,total=loss).items():sums[k]+=float(v.detach())*b
            summary[name]={k:v/n for k,v in sums.items()}
        history.append(summary);print(json.dumps(summary),flush=True)
        if summary['val']['total']<best:
            best=summary['val']['total'];torch.save({'state_dict':m.state_dict(),'configuration':m.configuration(),'training_mode':'joint_real_RGB_3D_pose','epoch':epoch+1,'seed':args.seed},args.output/'best.pt')
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
    (args.output/'run.json').write_text(json.dumps({'configuration':m.configuration(),'device':args.device,'seconds':time.perf_counter()-start,'train_clips':len(train),'val_clips':len(val),'epochs':args.epochs,'seed':args.seed,'training_mode':'joint_real_RGB_3D_pose'},indent=2))
if __name__=='__main__':main()
