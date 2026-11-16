"""Extract original frame IDs from a frame-aligned MP4; no temporal resampling."""
import argparse,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int,default=1024);a=p.parse_args()
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate,avg_frame_rate,width,height','-of','json',str(a.video)]))
    a.output.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-i',str(a.video),'-map','0:v:0','-an','-fps_mode','passthrough','-frames:v',str(a.max_frames),'-start_number','0','-q:v','2',str(a.output/'frame_%07d.jpg')],check=True)
    (a.output/'video_info.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
if __name__=='__main__':main()
