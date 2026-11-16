"""Download a bounded official Ego-Exo4D subset using an already authorized AWS profile."""
import argparse,json,os
from pathlib import Path
from urllib.parse import urlparse
import boto3
from botocore.config import Config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',default='egoexo');p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'data/egoexo')
    p.add_argument('--max-mb',type=float,default=500);p.add_argument('--download',action='store_true')
    a=p.parse_args();os.environ['AWS_EC2_METADATA_DISABLED']='true'
    s3=boto3.Session(profile_name=a.profile).client('s3',config=Config(connect_timeout=10,read_timeout=60,retries={'max_attempts':2}))
    def read_manifest(part):
        obj=s3.get_object(Bucket='ego4d-consortium-sharing',Key=f'egoexo-public/v2/{part}/manifest.json')
        return json.loads(obj['Body'].read())
    records=read_manifest('downscaled_takes/448');selected=[]
    for split,count in [('train',2),('val',1)]:
        body=a.annotations/'ego_pose'/split/'body/annotation';camera=a.annotations/'ego_pose'/split/'camera_pose'
        candidates=[]
        available={x.stem for x in body.glob('*.json') if not x.name.startswith('._')} & {x.stem for x in camera.glob('*.json') if not x.name.startswith('._')}
        for record in records:
            uid=record['uid']
            if uid not in available:continue
            for path in record['paths']:
                rel=path['relative_path'];views=path.get('views') or []
                if not Path(rel).name.startswith('aria') or not rel.endswith('_214-1.mp4'):continue
                if path.get('size') is None:continue
                candidates.append({'uid':uid,'split':split,**path})
        candidates.sort(key=lambda x:x['size'])
        chosen=[];seen=set()
        for candidate in candidates:
            if candidate['uid'] in seen:continue
            try:
                uid=candidate['uid'];poses=json.loads((body/f'{uid}.json').read_text());cams=json.loads((camera/f'{uid}.json').read_text())
                if len(poses)<16 or not any('aria' in k for k in cams):continue
            except (ValueError,OSError):continue
            chosen.append(candidate);seen.add(candidate['uid'])
            if len(chosen)==count:break
        if len(chosen)<count:raise ValueError(f'Insufficient matching {split} RGB videos; check fresh metadata/annotations')
        selected.extend(chosen)
    total=sum(x['size'] for x in selected)
    if total>a.max_mb*1e6:raise ValueError(f'Subset requires {total/1e6:.1f} MB, above explicit cap {a.max_mb} MB')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'download_plan.json').write_text(json.dumps(selected,indent=2))
    print(json.dumps({'videos':len(selected),'total_MB':total/1e6,'takes':[{'uid':x['uid'],'split':x['split'],'relative_path':x['relative_path']} for x in selected]},indent=2))
    if not a.download:return
    for x in selected:
        rel=Path(x['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
        u=urlparse(x['source_path'])
        if u.scheme!='s3':raise ValueError('Expected official S3 URI')
        target=a.output/rel;target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(target.suffix+'.part')
        s3.download_file(u.netloc,u.path.lstrip('/'),str(temp))
        if temp.stat().st_size!=x['size']:raise ValueError('Incomplete video download')
        temp.replace(target);print('Downloaded',rel,flush=True)
if __name__=='__main__':main()
