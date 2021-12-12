import os
from im2scene.discriminator import discriminator_dict
from im2scene.gan2d import models, training
from torch import randn
from copy import deepcopy
import numpy as np


def get_model(cfg, device=None, len_dataset=0, **kwargs):
    ''' Returns the model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        len_dataset (int): length of dataset
    '''
    # 读取配置文件
    discriminator = cfg['model']['discriminator']
    generator = cfg['model']['generator']
    z_dim = cfg['model']['z_dim']
    discriminator_kwargs = cfg['model']['discriminator_kwargs']
    generator_kwargs = cfg['model']['generator_kwargs']
    img_size = cfg['data']['img_size']
    # 如果判别器配置不为空
    if discriminator is not None:
        # 'dc': conv.DCDiscriminator,
        # 'resnet': conv.DiscriminatorResnet,
        # 获取判别器网络
        discriminator = discriminator_dict[discriminator](
            image_size=img_size, **discriminator_kwargs)
    # 如果生成器配置不为空
    if generator is not None:
        # 定义一个随机生成噪声的函数
        def prior_dist(): return randn(cfg['training']['batch_size'], z_dim)
        # 获取生成器网络
        generator = models.generator_dict[generator](
            device, prior_dist=prior_dist, z_dim=z_dim, img_size=img_size,
            **generator_kwargs
        )
    # 如果开启take_generator_average
    if cfg['test']['take_generator_average']:
        # 深复制生成器
        generator_test = deepcopy(generator)
    else:
        generator_test = None
    # 获取GAN2D模型
    model = models.GAN2D(
        device=device,
        discriminator=discriminator, generator=generator,
        generator_test=generator_test,
    )
    return model


def get_trainer(model, optimizer, optimizer_d, cfg, device,
                **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the 2DGAN model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    # 读取配置
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    overwrite_visualization = cfg['training']['overwrite_visualization']
    multi_gpu = cfg['training']['multi_gpu']
    n_eval_iterations = 10000 // cfg['training']['batch_size']
    # 确保fidfile存在
    assert(cfg['data']['fid_file'] is not None)
    # 读取fidfile
    fid_dict = np.load(cfg['data']['fid_file'])
    # 获取trainer
    trainer = training.Trainer(
        model, optimizer, optimizer_d, device=device, vis_dir=vis_dir,
        overwrite_visualization=overwrite_visualization, multi_gpu=multi_gpu,
        fid_dict=fid_dict,
        n_eval_iterations=n_eval_iterations)

    return trainer
