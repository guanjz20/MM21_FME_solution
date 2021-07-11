import os
import os.path as osp
from tqdm import tqdm
from glob import glob

from video_processor import Video_Processor
import params

# OpenFace parameters
save_size = 224
OpenFace_exe = params.OpenFace_exe
quiet = True
nomask = True
grey = False
tracked_vid = False
noface_save = False

# dataset
video_root = params.video_root

# main
video_processor = Video_Processor(save_size, nomask, grey, quiet, tracked_vid,
                                  noface_save, OpenFace_exe)
video_ps = list(glob(osp.join(video_root, '*/*mp4')))
video_ps.extend(list(glob(osp.join(video_root, '*/*avi'))))

for video_p in tqdm(video_ps):
    video_name = os.path.basename(video_p).split('.')[0]
    opface_output_dir = os.path.join(os.path.dirname(video_p),
                                     video_name + "_opface")
    video_processor.process(video_p, opface_output_dir)
