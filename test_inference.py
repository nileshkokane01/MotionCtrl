import argparse
import datetime
import glob
import json
import math
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision
## note: decord should be imported after torch
from omegaconf import OmegaConf

from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from main.evaluation.motionctrl_prompts_camerapose_trajs import (
    both_prompt_camerapose_traj, cmcm_prompt_camerapose, omom_prompt_traj)
from utils.utils import instantiate_from_config
import open_clip 


print('attempting to load the open_clip model ')
model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
print('successful')







