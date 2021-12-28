import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import time
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler 
# 以模块名定义logger名，返回logger对象
logger_py = logging.getLogger(__name__)
# numpy设置随机数种子
np.random.seed(0)
# 设置生成随机数的种子，返回generate对象
torch.manual_seed(0)

# 创建解析器
parser = argparse.ArgumentParser(
    description='Train a GIRAFFE model.'
)
# 添加命令行参数
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')
parser.add_argument("--local_rank", default=-1, type=int)
# 拿出参数集
args = parser.parse_args()
# 加载指定的配置文件（指定path，默认path）
cfg = config.load_config(args.config, 'configs/default.yaml')
# 可以使用cuda的条件是torch cuda环境正确且命令行no_cuda未开启
is_cuda = (torch.cuda.is_available() and not args.no_cuda)



"""
    DDP初始化：要放在所有DDP代码前
"""

# nccl是GPU设备上最快、最推荐的后端
dist.init_process_group(backend='nccl')  

# local_rank参数
local_rank = dist.get_rank()

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)

is_master = True if local_rank == 0 else False

# 如果cuda可以使用，device为cuda，否则是cpu
device = torch.device("cuda" if is_cuda else "cpu",local_rank)

# torch.cuda.set_per_process_memory_fraction(1.0, device)

"""
    配置文件读取
"""
# 输出文件夹
out_dir = cfg['training']['out_dir']
# 
backup_every = cfg['training']['backup_every']
# 程序退出时间（定时器到时间结束训练，保存checkpoint并退出）
exit_after = args.exit_after
# 学习率
lr = cfg['training']['learning_rate']
# 判别器学习率???
lr_d = cfg['training']['learning_rate_d']
# 批大小
batch_size = cfg['training']['batch_size']
# 进程数
n_workers = cfg['training']['n_workers']
# 定时开始时间
t0 = time.time()
# 交叉验证模式???
model_selection_metric = cfg['training']['model_selection_metric']
# 选择最大最小模型
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# 输出文件夹不存在就新建
if not os.path.exists(out_dir) and dist.get_rank()==0:
    os.makedirs(out_dir)

# 根据配置文件读取数据集
train_dataset = config.get_dataset(cfg)

##TODO: DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
train_sampler = DistributedSampler(train_dataset)

# 使用dataloader读取数据集
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=n_workers,
    pin_memory=True, drop_last=True,
    sampler=train_sampler ## TODO:DDP
)
# 根据配置文件获取模型
# TODO:DDP
model = config.get_model(cfg, device=device, len_dataset=len(train_dataset))

# DDP将主节点的param和buffer分发到各进程，让他们保持参数一致
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
model = model.module

# Initialize training
# RMSProp优化算法或者是Adam优化算法
op = optim.RMSprop if cfg['training']['optimizer'] == 'RMSprop' else optim.Adam
# 一些优化器参数
optimizer_kwargs = cfg['training']['optimizer_kwargs']
# 如果模型中有生成器且生成器非空
if hasattr(model, "generator") and model.generator is not None:
    # 获取生成器参数
    parameters_g = model.generator.parameters()
else:
    # 获取decoder的参数
    parameters_g = list(model.decoder.parameters())
# 定义优化器（优化生成器参数/decoder参数）
optimizer = op(parameters_g, lr=lr, **optimizer_kwargs)
# 如果模型中有判别器且判别器非空
if hasattr(model, "discriminator") and model.discriminator is not None:
    # 获取判别器参数
    parameters_d = model.discriminator.parameters()
    # 定义判别器的优化器
    optimizer_d = op(parameters_d, lr=lr_d)
else:
    optimizer_d = None

# 根据配置文件获取指定模型的训练器
trainer = config.get_trainer(model, optimizer, optimizer_d, cfg, device=device, 
                            use_DDP=False, device_ids=[local_rank], output_device=local_rank)

# model checkpoint读取对象
checkpoint_io = CheckpointIO(out_dir, local_rank, model=model, optimizer=optimizer,
                             optimizer_d=optimizer_d)


try:
    # 读取checkpoint
    load_dict = checkpoint_io.load('model.pt', device)
    print("Loaded model checkpoint.")
# 找不到pt文件
except FileExistsError:
    load_dict = dict()
    print("No model checkpoint found.")

# 从checkpoint读取一些参数
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)
# loss_val_best如果是正负无穷
if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
        % (model_selection_metric, metric_val_best))

# pytorch的logger，将event写入文件
logger = SummaryWriter(os.path.join(out_dir, 'logs'))
# Shorthands
# 打印频次
print_every = cfg['training']['print_every']
# checkpoint保存频率
checkpoint_every = cfg['training']['checkpoint_every']
# 验证频率
validate_every = cfg['training']['validate_every']
# 可视化频率
visualize_every = cfg['training']['visualize_every']

# 打印参数个数
# 统计参数个数
nparameters = sum(p.numel() for p in model.parameters())
logger_py.info('Total number of parameters: %d' % nparameters)
# 统计打印判别器参数个数
if hasattr(model, "discriminator") and model.discriminator is not None:
    nparameters_d = sum(p.numel() for p in model.discriminator.parameters())
    logger_py.info(
        'Total number of discriminator parameters: %d' % nparameters_d)
# 统计打印生成器参数个数
if hasattr(model, "generator") and model.generator is not None:
    nparameters_g = sum(p.numel() for p in model.generator.parameters())
    logger_py.info('Total number of generator parameters: %d' % nparameters_g)

t0b = time.time()

print("开始训练......")

dist.barrier()

while (True):
    epoch_it += 1
    # TODO: DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    train_loader.sampler.set_epoch(epoch_it)

    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch, it)
        for (k, v) in loss.items():
            logger.add_scalar(k, v, it)
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            info_txt = '[Epoch %02d] it=%03d, time=%.3f ,pid=%d'% (
                epoch_it, it, time.time() - t0b,local_rank)
            for (k, v) in loss.items():
                info_txt += ', %s: %.4f' % (k, v)
            logger_py.info(info_txt)
            t0b = time.time()

        # # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0 and is_master:
            logger_py.info('Visualizing')
            image_grid = trainer.visualize(it=it)
            if image_grid is not None:
                logger.add_image('images', image_grid, it)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0) and is_master:
            logger_py.info('Saving checkpoint')
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0) and is_master:
            logger_py.info('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Run validation
        if validate_every > 0 and (it % validate_every) == 0 and (it > 0) and is_master:
            print("Performing evaluation step.")
            eval_dict = trainer.evaluate()
            metric_val = eval_dict[model_selection_metric]
            logger_py.info('Validation metric (%s): %.4f'
                           % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                logger_py.info('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.backup_model_best('model_best.pt')
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
            

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            logger_py.info('Time limit reached. Exiting.')
            if is_master:
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)
            exit(3)
        
        dist.barrier()
