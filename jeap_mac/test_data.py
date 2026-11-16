"""Synthetic integration fixture, never used as scientific training data."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_clips import JOINTS
from data import Clips

def main():
    root=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=root,prefix='fixture_') as d:
        d=Path(d)
        for split in ['train','val']:
            folder=d/split;folder.mkdir();body={};camera={'aria01':{'camera_extrinsics':{}}}
            for i in [0,3,6,9]:
                Image.fromarray(np.full((64,64,3),80+i,dtype=np.uint8)).save(folder/f'frame_{i:07d}.jpg')
                body[str(i)]=[{'annotation3D':{name:{'x':j*.01,'y':i*.01,'z':1.} for j,name in enumerate(JOINTS)}}]
                camera['aria01']['camera_extrinsics'][str(i)]=np.eye(4)[:3].tolist()
            (folder/'body.json').write_text(json.dumps(body));(folder/'camera.json').write_text(json.dumps(camera))
            subprocess.run([sys.executable,str(root/'prepare_clips.py'),'--body',str(folder/'body.json'),'--camera',str(folder/'camera.json'),'--frames',str(folder),'--take-uid',f'synthetic_{split}','--fps','30','--output',str(folder/'clips.jsonl')],check=True)
            ds=Clips(folder/'clips.jsonl');assert len(ds)==1
            b=ds[0];assert b['frames'].shape==(4,3,64,64) and b['actions'].shape==(3,17,3)
        subprocess.run([sys.executable,str(root/'train.py'),'--train',str(d/'train/clips.jsonl'),'--val',str(d/'val/clips.jsonl'),'--output',str(d/'run'),'--epochs','1'],check=True)
        assert (d/'run/best.pt').exists()
        subprocess.run([sys.executable,str(root/'evaluate.py'),'--manifest',str(d/'val/clips.jsonl'),'--checkpoint',str(d/'run/best.pt'),'--output',str(d/'evaluation.json')],check=True)
    print('Synthetic manifest preparation, camera transform, dataset loading, joint optimizer and checkpoint checks passed')
if __name__=='__main__':main()
