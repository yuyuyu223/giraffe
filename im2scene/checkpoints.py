import os,gc
import urllib
import torch
from torch.utils import model_zoo
import shutil
import datetime


class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''

    def __init__(self, checkpoint_dir='./chkpts', local_rank=-1, **kwargs):
        self.module_dict = kwargs
        self.local_rank=local_rank
        self.checkpoint_dir = checkpoint_dir
        # checkpoint_dir不存在去创建
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        # 更新模组字典
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        # 如果不是绝对路径，生成绝对路径
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        # 保存模型参数
        torch.save(outdict, filename)

    def backup_model_best(self, filename, **kwargs):
        # 如果不是绝对路径，生成绝对路径
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        # 如果最佳模型文件存在
        if os.path.exists(filename):
            # 生成备份目录路径
            backup_dir = os.path.join(self.checkpoint_dir, 'backup_model_best')
            # 路径不存在就创建
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            # 获取时间戳
            ts = datetime.datetime.now().timestamp()
            # 生成备份文件路径
            filename_backup = os.path.join(backup_dir, '%s.pt' % ts)
            # 拷贝
            shutil.copy(filename, filename_backup)

    def load(self, filename, device):
        '''Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        '''
        # 如果文件路径是url
        if is_url(filename):
            return self.load_url(filename, device)
        # 如果文件路径是本地文件
        else:
            return self.load_file(filename, device)

    def load_file(self, filename, device):
        '''Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        '''
        # 如果不是绝对路径，生成绝对路径
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        # 如果文件存在
        if os.path.exists(filename):
            print(filename)
            print('=>GPU{} : Loading checkpoint from local file...'.format(device))
            # 加载模型参数
            state_dict = torch.load(filename, map_location=device)
            scalars = self.parse_state_dict(state_dict)
            del state_dict
            gc.collect()
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, device):
        '''Load a module dictionary from url.

        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        # 用pytorch model zoo下载模型
        state_dict = model_zoo.load_url(url, progress=True, map_location=device)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        '''Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
    '''
        # 返回冗余下来的键值对
        scalars = {k: v for k, v in state_dict.items()
                   if k not in self.module_dict}
        # 把state_dict的内容加载到各个model里
        # "model":model 
        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        
        return scalars

    


def is_url(url):
    ''' Checks if input string is a URL.

    Args:
        url (string): URL
    '''
    # 检查协议名称
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
