import yaml
from im2scene import data
from im2scene import gan2d, giraffe
import logging
import os


# method directory; for this project we only use giraffe
method_dict = {
    'gan2d': gan2d,
    'giraffe': giraffe,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
        将dict2的内容更新到dict1
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        # 2中的key不在1中
        if k not in dict1:
            # 在1中新建一个key并初始化value为一个dict
            dict1[k] = dict()
        # 如果v是一个字典
        if isinstance(v, dict):
            # 递归update
            update_recursive(dict1[k], v)
        else:
            # 不是字典直接赋值
            dict1[k] = v


# Models
def get_model(cfg, device=None, len_dataset=0):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    # 获取方法名
    method = cfg['method']
    # 根据方法名找到对应模型模块，获取模型
    model = method_dict[method].config.get_model(
        cfg, device=device, len_dataset=len_dataset)
    return model


def set_logger(cfg):
    """
        根据配置文件设置logger
    """
    logfile = os.path.join(cfg['training']['out_dir'],
                           cfg['training']['logfile'])
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logfile,
        filemode='a',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)


# Trainer
def get_trainer(model, optimizer, optimizer_d, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    # 获取方法名
    method = cfg['method']
    # 设置log
    set_logger(cfg)
    # 获取指定模型的训练器
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, optimizer_d, cfg, device)
    return trainer


# Renderer
def get_renderer(model, cfg, device):
    ''' Returns a render instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    # 获取方法名
    method = cfg['method']
    # 获取指定渲染器
    renderer = method_dict[method].config.get_renderer(model, cfg, device)
    return renderer


def get_dataset(cfg, **kwargs):
    ''' Returns a dataset instance.

    Args:
        cfg (dict): config dictionary
        mode (string): which mode is used (train / val /test / render)
        return_idx (bool): whether to return model index
        return_category (bool): whether to return model category
    '''
    # Get fields with cfg
    # 数据集名称
    dataset_name = cfg['data']['dataset_name']
    # 数据集位置
    dataset_folder = cfg['data']['path']
    # 数据集分类
    categories = cfg['data']['classes']
    # 数据集图片尺寸
    img_size = cfg['data']['img_size']
    # 如果是lsun数据集
    if dataset_name == 'lsun':
        # 调用im2scene.data.datasets中的方法读取数据集
        dataset = data.LSUNClass(dataset_folder, categories, size=img_size,
                                 random_crop=cfg['data']['random_crop'],
                                 use_tanh_range=cfg['data']['use_tanh_range'],
                                 )
    # 如果是其他数据集
    else:
        # 调用im2scene.data.datasets中的方法读取数据集
        dataset = data.ImagesDataset(
            dataset_folder, size=img_size,
            use_tanh_range=cfg['data']['use_tanh_range'],
            celebA_center_crop=cfg['data']['celebA_center_crop'],
            random_crop=cfg['data']['random_crop'],
        )
    return dataset
