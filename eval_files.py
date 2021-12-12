import torch
import os
import argparse
import numpy as np
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from torchvision.utils import save_image, make_grid

# 添加一些参数
parser = argparse.ArgumentParser(
    description='Evaluate your own generated images (see ReadMe for more\
                 information).'
)
parser.add_argument('--input-file', type=str, help='Path to input file.')
parser.add_argument('--gt-file', type=str, help='Path to gt file.')
parser.add_argument('--n-images', type=int, default=20000,
                    help='Number of images used for evaluation.')

args = parser.parse_args()
n_images = args.n_images


def load_np_file(np_file):
    """
        加载npy文件
    """
    # 获取文件后缀
    ext = os.path.basename(np_file).split('.')[-1]
    assert(ext in ['npy'])
    # 如果是npy文件
    if ext == 'npy':
        # 转换为torch张量
        return torch.from_numpy(np.load(np_file)).float() / 255

# 加载图片的前n_images张
img_fake = load_np_file(args.input_file)[:n_images]
fid_dict = np.load(args.gt_file)
out_dict_file = os.path.join(
    os.path.dirname(args.input_file), 'fid_evaluation.npz')
out_vis_file = os.path.join(
    os.path.dirname(args.input_file), 'fid_evaluation.jpg')
out_dict = {}

# 准备计算n_images张图片的FID分数
print("Start FID calculation with %d images ..." % img_fake.shape[0])
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
