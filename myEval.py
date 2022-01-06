import torch
import os
from tqdm import tqdm
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import numpy as np
from torchvision.utils import save_image, make_grid
import logging
from im2scene import config


############################################################################################
# 配置文件
config_path = "./configs/64res/cars_64.yaml"
model_path = "backup_model_best/1640775020.696839.pt"
# 加载配置文件，若没有配置文件就加载默认配置
cfg = config.load_config(config_path, 'configs/default.yaml')
# 设置图片保存位置
out_dir = cfg['training']['out_dir']
out_vis_file = os.path.join(out_dir, 'fid_images.jpg')
############################################################################################
## Logger
# 以模块名定义logger名，返回logger对象
logger_py = logging.getLogger(__name__)
config.set_logger(cfg)
############################################################################################
## 运行环境
# 判断cuda是否可用
is_cuda = torch.cuda.is_available()
logger_py.info("CUDA:{}".format(is_cuda))
# 设置cpu/gpu
device = torch.device("cuda" if is_cuda else "cpu",3)
logger_py.info("we will use %s"%device)
#############################################################################################
## 加载模型
model = config.get_model(cfg, device=device)
gen = model.generator
logger_py.info("switch to eval mode")
gen.eval()
# 加载checkpoint
try:
    checkpoint_io = CheckpointIO(checkpoint_dir=out_dir, model=model)
    checkpoint_io.load(model_path, device=device)
except Exception as e:
    logger_py.warning("no check point found!")

##############################################################################################
## 生成参数
def make_gernerate_args(n_iter=25, batch_size=1, change=None, device=None):
    if change == None:
        logger_py.error("please change a input")
        exit(0)
    elif change== "rotate":
        latent_codes = gen.get_latent_codes(batch_size)
        camera_matrices = gen.get_random_camera(batch_size)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        s = gen.bounding_box_generator.get_scale(
            batch_size=batch_size, val=[[0.5,0.5,0.5]])
        t = gen.bounding_box_generator.get_translation(
            batch_size=batch_size, val=[[0.5,0.5,0.5]])
        eval_args_list=[{
            "batch_size":batch_size, 
            "latent_codes":latent_codes, 
            "camera_matrices":camera_matrices,
            "transformations":[s.to(device),t.to(device),gen.bounding_box_generator.get_rotation(batch_size=batch_size, val=[i/n_iter]).to(device)], 
            "bg_rotation":bg_rotation, 
            "mode":"eval", 
            "it":0,
            "return_alpha_map":False,
            "not_render_background":False,
            "only_render_background":False
        } for i in range(n_iter)]
        return eval_args_list
    else:
        logger_py.error("%s can not be changed!"%change)
        exit(0)

#############################################################################################
## 生成
def Generate(num=1):
    for i in range(num):
        out_vis_file = os.path.join(out_dir, 'eval_out', 'fid_images%d.jpg'%i)
        img_fake = []
        argsGen = make_gernerate_args(n_iter=60, batch_size=1, change="rotate", device=device)
        for i in tqdm(argsGen):
            with torch.no_grad():
                # [batch_size,h,w,3]
                img_fake.append(gen(**i).cpu())
        # [batch_size*n_iter,h,w,3]
        img_fake = torch.cat(img_fake, dim=0)
        # 将元素值范围界定到[0,1]
        img_fake.clamp_(0., 1.)
        # 图像个数变为batch_size*n_iter
        n_images = img_fake.shape[0]
        # 图像回到uint8
        img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)

        # use uint for eval to fairly compare
        img_fake = torch.from_numpy(img_uint8).float() / 255.

        #######################################################################
        ## 保存结果
        logger_py.info("save result to '%s'"%out_dir)
        # 拿出256张图，生成16x16的网格进行可视化展示，保存到fid_images.jpg
        save_image(make_grid(img_fake, nrow=10, pad_value=1.), out_vis_file)

Generate(20)