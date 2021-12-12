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
# 加载FID文件
fid_file = cfg['data']['fid_file']
assert(fid_file is not None)
fid_dict = np.load(cfg['data']['fid_file'])
# 测试图片数
n_images = cfg['test']['n_images']
# 批大小
batch_size = cfg['training']['batch_size']
# 计算迭代次数
n_iter = ceil(n_images / batch_size)
# 输出信息
out_dict = {'n_images': n_images}
# 假图列表
img_fake = []
t0 = time.time()
# 让模型生成图像放入img_fake
for i in tqdm(range(n_iter)):
    with torch.no_grad():
        # [batch_size,h,w,3]
        img_fake.append(model(batch_size).cpu())
# [batch_size*n_iter,h,w,3]
img_fake = torch.cat(img_fake, dim=0)[:n_images]
# 将元素值范围界定到[0,1]
img_fake.clamp_(0., 1.)
# 图像个数变为batch_size*n_iter
n_images = img_fake.shape[0]
# 获取生成+处理的总时间
t = time.time() - t0
# 总时间
out_dict['time_full'] = t
# 每张图片的时间
out_dict['time_image'] = t / n_images
# 图像回到uint8
img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)
# fid_images.npy
np.save(out_img_file[:n_images], img_uint8)

# use uint for eval to fairly compare
img_fake = torch.from_numpy(img_uint8).float() / 255.
# InceptionV3的pool_3 layer输出的均值和方差
mu, sigma = calculate_activation_statistics(img_fake)
out_dict['m'] = mu
out_dict['sigma'] = sigma

# 计算FID分数
fid_score = calculate_frechet_distance(mu, sigma, fid_dict['m'], fid_dict['s'])
out_dict['fid'] = fid_score
# 打印FID分数
print("FID Score (%d images): %.6f" % (n_images, fid_score))
# fid_evaluation.npz
np.savez(out_dict_file, **out_dict)

# 拿出256张图，生成16x16的网格进行可视化展示，保存到fid_images.jpg
save_image(make_grid(img_fake[:256], nrow=16, pad_value=1.), out_vis_file)
