import torch
import os
import argparse
from tqdm import tqdm
import time
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import numpy as np
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from math import ceil
from torchvision.utils import save_image, make_grid

# 添加config和nocuda参数
parser = argparse.ArgumentParser(
    description='Evaluate a GIRAFFE model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
# 加载配置文件，若没有配置文件就加载默认配置
cfg = config.load_config(args.config, 'configs/default.yaml')
# 判断cuda是否可用
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
# 设置cpu/gpu
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
out_dict_file = os.path.join(out_dir, 'fid_evaluation.npz')
out_img_file = os.path.join(out_dir, 'fid_images.npy')
out_vis_file = os.path.join(out_dir, 'fid_images.jpg')

# 加载模型
model = config.get_model(cfg, device=device)
# 加载checkpoint
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])


# Generate
# 模型评估
model.eval()
gen = model.generator

latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
bg_rotation = gen.get_random_bg_rotation(batch_size)