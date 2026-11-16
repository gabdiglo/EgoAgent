"""Real-image optimization probe only. Does not train the action component."""
import os
from pathlib import Path
os.environ.setdefault('TRANSFORMERS_CACHE',str(Path(__file__).parent/'cache'))
import argparse,csv,json,time
import torch
from PIL import Image,ImageFile
from torchvision import transforms
from model import JEAP
ImageFile.LOAD_TRUNCATED_IMAGES=False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).parent/'runs/visual_probe');a=p.parse_args()
    torch.manual_seed(42);torch.set_num_threads(2);rows=list(csv.DictReader((a.project/'data/processed/wtour_pairs_1fps_gap5_train_clean.csv').open()))
    # Spread the probe across the training CSV, never inspect test performance.
    ids=torch.randperm(len(rows))[:128].tolist();images=[];selected=[];errors=[]
    resize=transforms.Resize((64,64));to_tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize(.5,.5)])
    for i in ids:
        try:
            pair=[]
            for k in ['current_frame','future_frame']:
                with Image.open(a.project/rows[i][k]) as im:im.load();pair.append(resize(im.convert('RGB')))
            images.append(pair);selected.append(rows[i])
        except (OSError,ValueError,Image.DecompressionBombError) as e:errors.append({'row':i,'error':str(e)})
        if len(images)==16:break
    if len(images)<4:raise ValueError('Not enough readable real training pairs')
    frames=torch.stack([torch.stack([to_tensor(x) for x in pair]) for pair in images])
    aug=transforms.Compose([transforms.RandomResizedCrop(64,scale=(.6,1.)),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),transforms.Normalize(.5,.5)])
    m=JEAP();m.train();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    history=[];start=time.perf_counter()
    for step in range(20):
        idx=torch.randperm(len(images))[:4];x=frames[idx]
        v1=torch.stack([aug(images[i][0]) for i in idx]).unsqueeze(1);v2=torch.stack([aug(images[i][0]) for i in idx]).unsqueeze(1)
        loss,parts,teacher=m.losses(x,v1,v2)
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();m.update_observer(teacher)
        history.append({'step':step+1,**{k:float(v.detach()) for k,v in parts.items()}})
    a.output.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':m.state_dict(),'configuration':m.configuration(),'training_mode':'visual_only_probe_ACTION_MODULE_UNTRAINED'},a.output/'visual_probe.pt')
    r={'mode':'real WalkingTours training images, visual objectives only; NOT joint JEAP training or scientific evaluation','steps':20,'pairs':len(images),'seconds':time.perf_counter()-start,'device':'cpu','history':history,'selected_pairs':selected,'unreadable':errors}
    (a.output/'visual_probe.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['history','selected_pairs','unreadable']},indent=2))
if __name__=='__main__':main()
