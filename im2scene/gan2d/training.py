from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average)
from torchvision.utils import save_image, make_grid
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
import os
import torch
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
logger_py = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    ''' Trainer object for the 2D-GAN.

    Args:
        model (nn.Module): 2D-GAN model
        optimizer (optimizer): generator optimizer
        optimizer_d (optimizer): discriminator optimizer
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): FID GT dictionary
        n_eval_iterations (int): number of evaluation iterations
    '''

    def __init__(self, model, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 generator=None,
                 multi_gpu=False,  fid_dict={},
                 n_eval_iterations=10, **kwargs):
        self.model = model
        # 开启multi_gpu会使用DP，DDP模式要禁用
        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            # 否则使用正常传进来的model
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test
        # 生成器和判别器的优化器
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        # 可视化存储路径
        self.vis_dir = vis_dir
        # 是否覆盖可视化
        self.overwrite_visualization = True
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations

        self.visualize_z = torch.randn(
            16, self.generator.z_dim).to(device)
        # 可视化存储路径不存在就创建
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        ''' 
        训练的一步
        Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        # 训练生成器
        loss_g = self.train_step_generator(data, it)
        # 训练判别器
        loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)
        return {
            'generator': loss_g,
            'discriminator': loss_d,
            'regularizer': reg_d,
            'd_real': real_d,
            'd_fake': fake_d,
        }

    def eval_step(self, data=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        # gen是生成器
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        # eval模式
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations
        # 生成器生成n_iter个fake图
        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen(None).cpu())
        # [n_iter,h,w,3]
        x_fake = torch.cat(x_fake, dim=0)
        # [-1,1]-->[0,1]
        x_fake = x_fake * 0.5 + 0.5
        # 计算FID所需要的mu和sigma
        mu, sigma = calculate_activation_statistics(x_fake)
        # 计算FID
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        # 返回结果
        eval_dict = {
            'fid_score': fid_score
        }

        return eval_dict

    def train_step_generator(self, data, it=None, z=None):
        """
            训练一步生成器
        """
        # 生成器和判别器
        generator = self.generator
        discriminator = self.discriminator
        # 为模型所有参数开启自动微分（保存梯度信息）
        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        # 开启train模式
        generator.train()
        discriminator.train()
        # 清空梯度
        self.optimizer.zero_grad()
        # 随机生成噪声
        z = generator.sample_z()
        # 生成假图
        x_fake = generator(z)
        # 判别器为假图打分
        d_fake = discriminator(x_fake)
        # 计算bceloss
        gloss = compute_bce(d_fake, 1)
        # 反向传播
        gloss.backward()
        # 更新参数
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)
        # 返回loss
        return gloss.item()

    def train_step_discriminator(self, data, it=None, z=None):
        """
            训练一步判别器
        """
        # 生成器和判别器
        generator = self.generator
        discriminator = self.discriminator
        # 为模型所有参数开启自动微分（保存梯度信息）
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        # 开启train模式
        generator.train()
        discriminator.train()
        # 清空梯度
        self.optimizer_d.zero_grad()
        # 从数据集读取真图片
        x_real = data.get('image').to(self.device)
        loss_d_full = 0.

        x_real.requires_grad_()
        # 判别器为真图片打分
        d_real = discriminator(x_real)
        # 计算判别器判定真图的loss
        d_loss_real = compute_bce(d_real, 1)
        # 累加loss
        loss_d_full += d_loss_real
        # 通过自动微分求惩罚项
        reg = 10. * compute_grad2(d_real, x_real).mean()
        # 累加loss
        loss_d_full += reg
        # 生成假图
        with torch.no_grad():
            x_fake = generator(z)

        x_fake.requires_grad_()
        # 判别器为假图打分
        d_fake = discriminator(x_fake)
        # 计算判别器判别假图的loss
        d_loss_fake = compute_bce(d_fake, 0)
        # 累加loss
        loss_d_full += d_loss_fake
        # 反向传播
        loss_d_full.backward()
        # 更新参数
        self.optimizer_d.step()
        # 判别器的总loss
        d_loss = (d_loss_fake + d_loss_real)
        # 返回各种loss信息
        return (d_loss.item(), reg.item(), d_loss_fake.item(),
                d_loss_real.item())

    def visualize(self, it=0, **kwargs):
        ''' 
        可视化
        Visualize the data.

        '''
        # eval模式
        self.model.generator.eval()
        # gen是生成器
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        # 生成假图并标准化
        with torch.no_grad():
            image_fake = self.generator(self.visualize_z).cpu()
            # rescale
            image_fake = image_fake * 0.5 + 0.5
        # 如果开启覆盖
        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_%010d.png' % it
        # 4x4的可视化结果
        image_grid = make_grid(image_fake.clamp_(0., 1.), nrow=4)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid
