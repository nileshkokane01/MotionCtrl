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





#model_config= {'base_learning_rate': 0.0001, 'scale_lr': False, 'target': 'motionctrl.motionctrl.MotionCtrl', 'params': {'omcm_config': {'pretrained': None, 'target': 'lvdm.modules.encoders.adapter.Adapter', 'params': {'channels': [320, 640, 1280, 1280], 'nums_rb': 2, 'cin': 128, 'sk': True, 'use_conv': False}}, 'linear_start': 0.00085, 'linear_end': 0.012, 'num_timesteps_cond': 1, 'log_every_t': 200, 'timesteps': 1000, 'first_stage_key': 'video', 'cond_stage_key': 'caption', 'cond_stage_trainable': False, 'conditioning_key': 'crossattn', 'image_size': [32, 32], 'channels': 4, 'scale_by_std': False, 'scale_factor': 0.18215, 'use_ema': False, 'uncond_prob': 0.1, 'uncond_type': 'empty_seq', 'empty_params_only': True, 'scheduler_config': {'target': 'utils.lr_scheduler.LambdaLRScheduler', 'interval': 'step', 'frequency': 100, 'params': {'start_step': 0, 'final_decay_ratio': 0.01, 'decay_steps': 20000}}, 'unet_config': {'target': 'lvdm.modules.networks.openaimodel3d_next.UNetModel', 'params': {'in_channels': 4, 'out_channels': 4, 'model_channels': 320, 'attention_resolutions': [4, 2, 1], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4], 'num_head_channels': 64, 'transformer_depth': 1, 'context_dim': 1024, 'use_linear': True, 'use_checkpoint': True, 'temporal_conv': True, 'temporal_attention': True, 'temporal_selfatt_only': True, 'use_relative_position': False, 'use_causal_attention': False, 'temporal_length': 16, 'use_image_dataset': False, 'addition_attention': True}}, 'first_stage_config': {'target': 'lvdm.models.autoencoder.AutoencoderKL', 'params': {'embed_dim': 4, 'monitor': 'val/rec_loss', 'ddconfig': {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}, 'lossconfig': {'target': 'torch.nn.Identity'}}}, 'cond_stage_config': {'target': 'lvdm.modules.encoders.condition2.FrozenOpenCLIPEmbedder', 'params': {'freeze': True, 'layer': 'penultimate'}}}}

model = instantiate_from_config(model_config)
#model = model.cuda(gpu_no)
print('------------ model  - ------- - -------')
print(model ) 


