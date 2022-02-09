import tkinter as tk
from im2scene import config
import torch
import os
import logging
from im2scene.checkpoints import CheckpointIO
from torchvision import transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import time


unloader = transforms.ToPILImage()
normalize = transforms.Normalize(0.5, 0.5)


class RotateGUI:
    def __init__(self):
        self.win = tk.Tk()
        self.win.geometry("960x600")
        ############################################################################################
        # 配置文件
        config_path = "./configs/256res/cars_256_pretrained.yaml"
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
        device = torch.device("cuda" if is_cuda else "cpu")
        logger_py.info("we will use %s"%device)
        #############################################################################################
        ## 加载模型
        self.model = config.get_model(cfg, device=device)
        self.gen = self.model.generator
        self.gen.resolution_vol=16
        logger_py.info("switch to eval mode")
        self.gen.eval()
        # 加载checkpoint
        try:
            checkpoint_io = CheckpointIO(model=self.model)
            checkpoint_io.load("checkpoint_cars256-d9ea5e11.pt")
        except Exception as e:
            logger_py.warning("no check point found!")
        #############################################################################
        self.device = torch.device("cuda",0)
        self.latent_codes = self.gen.get_latent_codes(1)
        self.camera_matrices = self.gen.get_random_camera(1)
        self.bg_rotation = self.gen.get_random_bg_rotation(1)
        self.R = 0
        self.s = self.gen.bounding_box_generator.get_scale(
            batch_size=1, val=[[0.5,0.5,0.5]])
        self.t = self.gen.bounding_box_generator.get_translation(
            batch_size=1, val=[[0.5,0.5,0.5]])
        #############################################################################
        self.scale = tk.Scale(self.win, from_=0, to=360, orient=tk.HORIZONTAL, command=self.freshR,width=30,length=500)
        self.scale.set(0)  # 设置初始值
        self.scale.place(x=0,y=0)

        self.label1 = tk.Label(self.win, image=None)
        self.label1.place(x=10, y=100)
        self.label2 = tk.Label(self.win, image=None)
        self.label2.place(x=400, y=100)
        self.label3 = tk.Label(self.win, image=None)
        self.label3.place(x=700, y=100)

        self.button = tk.Button(self.win, text="播放", command=self.play)
        self.button.place(x=600,y=10)

        

    
    def freshR(self, ev=None):
        self.R = self.scale.get()/360
        self.Generate()
        time.sleep(1)
    
    def make_gernerate_args(self):
        batch_size = 1
        device = self.device
        eval_args={
            "batch_size":batch_size, 
            "latent_codes":self.latent_codes, 
            "camera_matrices":self.camera_matrices,
            "transformations":[self.s.to(device),self.t.to(device),self.gen.bounding_box_generator.get_rotation(batch_size=batch_size, val=[self.R]).to(device)], 
            "bg_rotation":self.bg_rotation, 
            "mode":"eval", 
            "it":0,
            "return_alpha_map":False,
            "return_feature_map":False,
            "return_all": True,
            "not_render_background":False,
            "only_render_background":False
        }
        return eval_args
    
    def Generate(self):
        args = self.make_gernerate_args()
        rgb_v, alpha_map, rgb = self.gen(**args)
        rgb_v, alpha_map, rgb = rgb_v.cpu(), alpha_map.cpu(), rgb.cpu()
        # 热力图
        rgb_v = rgb_v[0]
        # rgb_v = torch.sum(rgb_v, dim=0)
        rgb_v = rgb_v.max(0).values
        # print(type(rgb_v))
        rgb_v = unloader(rgb_v)
        buffer = io.BytesIO()
        sns.set()
        heatmap = sns.heatmap(np.array(rgb_v))
        plt.savefig(buffer,format = 'png')
        plt.close()
        hot = Image.open(buffer)
        hot = hot.resize((356,256))
        self.tkImage1 = ImageTk.PhotoImage(image=hot)
        self.label1['image'] = self.tkImage1
        buffer.close()
        hot.close()
        # alpha图
        alpha_map = unloader(alpha_map[0])
        alpha_map = alpha_map.resize((256,256))
        self.tkImage2 = ImageTk.PhotoImage(image=alpha_map)
        self.label2['image'] = self.tkImage2
        # rgb图
        rgb = unloader(rgb[0])
        rgb = rgb.resize((256,256))
        self.tkImage3 = ImageTk.PhotoImage(image=rgb)
        self.label3['image'] = self.tkImage3
    
    def play(self):
        for r in range(0,361,10):
            print("render %d"%r)
            self.scale.set(r)
            self.win.update()
            # time.sleep(2)
        

    
    def render(self):
        self.win.mainloop()

if __name__ == '__main__':
    gui = RotateGUI()
    gui.render()
    exit(0)

        
