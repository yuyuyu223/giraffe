import torch
import os
import argparse
from im2scene import config
from im2scene.checkpoints import CheckpointIO

# 添加config和nocuda参数
parser = argparse.ArgumentParser(
    description='Render images of a GIRAFFE model.'
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
# 设置输出路径
out_dir = cfg['training']['out_dir']
# 生成渲染路径在输出路径内
render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
# 无此文件夹则新建
if not os.path.exists(render_dir):
    os.makedirs(render_dir)

# 加载模型
model = config.get_model(cfg, device=device)
# 加载checkpoint
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
# 加载渲染器
renderer = config.get_renderer(model, cfg, device=device)
# 模型评估
model.eval()
# 渲染
out = renderer.render_full_visualization(
    render_dir,
    cfg['rendering']['render_program'])
