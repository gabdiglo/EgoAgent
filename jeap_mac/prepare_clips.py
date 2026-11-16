"""Match extracted frame numbers to official Ego-Exo4D body and camera annotations."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=False
JOINTS=['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--body',type=Path,required=True);p.add_argument('--camera',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--take-uid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--camera-name',default='aria01')
    p.add_argument('--fps',type=float,required=True);p.add_argument('--gap',type=int,default=3)
    p.add_argument('--steps',type=int,default=3);p.add_argument('--max-clips',type=int,default=128)
    a=p.parse_args();body=json.loads(a.body.read_text());camera=json.loads(a.camera.read_text())[a.camera_name]['camera_extrinsics']
    clips=[];rejected=[]
    for first in sorted(map(int,body)):
        ids=[first+k*a.gap for k in range(a.steps+1)]
        # One action pose at each observed time; last image is the future target.
        paths=[a.frames/f'frame_{i:07d}.jpg' for i in ids]
        try:
            for f in paths:
                with Image.open(f) as im:im.load()
            poses=[];masks=[]
            for i in ids[:-1]:
                people=body[str(i)]
                if len(people)!=1:raise ValueError('Expected one annotated wearer')
                joints=people[0]['annotation3D'];extr=np.asarray(camera[str(i)],dtype=float)
                if extr.shape!=(3,4):raise ValueError('Invalid world-to-camera extrinsics')
                pose=[];mask=[]
                for name in JOINTS:
                    if name in joints:
                        j=joints[name];xyz=extr@np.array([j['x'],j['y'],j['z'],1.]);pose.append(xyz.tolist());mask.append(1)
                    else:pose.append([0.,0.,0.]);mask.append(0)
                if not any(mask):raise ValueError('No valid joints')
                poses.append(pose);masks.append(mask)
            clips.append({'take_uid':a.take_uid,'frames':[str(x.resolve()) for x in paths],'frame_ids':ids,'fps':a.fps,'coordinate_system':'camera_meters','actions':poses,'action_mask':masks})
        except (OSError,ValueError,KeyError,Image.DecompressionBombError) as e:
            rejected.append({'first_frame':first,'reason':str(e)});continue
        if len(clips)>=a.max_clips:break
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(x)+'\n' for x in clips));a.output.with_suffix('.rejected.json').write_text(json.dumps(rejected,indent=2))
    if not clips:raise ValueError('No synchronized clips; see rejection report')
    print('Saved clips:',len(clips))
if __name__=='__main__':main()
