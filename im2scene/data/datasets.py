import os
import logging
from torch.utils import data
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import lmdb
import pickle
import string
import io
import random
# fix for broken images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class LSUNClass(data.Dataset):
    ''' LSUN Class Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        classes (str): class name
        size (int): image output size
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder,
                 classes='scene_categories/church_outdoor_train_lmdb',
                 size=64, random_crop=False, use_tanh_range=False):
        root = os.path.join(dataset_folder, classes)

        # Define transforms
        # 如果开启随机裁剪
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
            ]
        self.transform += [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        # 使用tanh
        if use_tanh_range:
            # 在[-1,1]范围内归一化
            self.transform += [transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)

        import time
        t0 = time.time()
        print('Start loading lmdb file ...')
        # lmdb读取
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # 生成cache路径
        cache_file = '_cache_' + ''.join(
            c for c in root if c in string.ascii_letters)
        # 有cache读cache
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            # 没有cache制作cache
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(
                    keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))
        print('done!')
        t = time.time() - t0
        print('time', t)
        print("Found %d files." % self.length)

    def __getitem__(self, idx):
        try:
            img = None
            # lmdb读取
            env = self.env
            with env.begin(write=False) as txn:
                imgbuf = txn.get(self.keys[idx])
            # 用buf的方法读取图片
            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            # 做图像变换
            if self.transform is not None:
                img = self.transform(img)
            # 把图片封装到字典里
            data = {
                'image': img
            }
            return data

        except Exception as e:
            # 异常可能是index溢出导致，这时候随机生成index返回图片
            print(e)
            idx = np.random.randint(self.length)
            return self.__getitem__(idx)

    def __len__(self):
        # 返回数据集长度
        return self.length


class ImagesDataset(data.Dataset):
    ''' Default Image Dataset Class.

    Args:
        dataset_folder (str): path to LSUN dataset
        size (int): image output size
        celebA_center_crop (bool): whether to apply the center
            cropping for the celebA and celebA-HQ datasets.
        random_crop (bool): whether to perform random cropping
        use_tanh_range (bool): whether to rescale images to [-1, 1]
    '''

    def __init__(self, dataset_folder,  size=64, celebA_center_crop=False,
                 random_crop=False, use_tanh_range=False):

        self.size = size
        assert(not(celebA_center_crop and random_crop))
        # 如果开启随机裁剪
        if random_crop:
            self.transform = [
                transforms.Resize(size),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        # celebA裁剪
        elif celebA_center_crop:
            if size <= 128:  # celebA
                crop_size = 108
            else:  # celebAHQ
                crop_size = 650
            self.transform = [
                transforms.CenterCrop(crop_size),
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        else:
            self.transform = [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        if use_tanh_range:
            self.transform += [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(self.transform)
        # 数据类型
        self.data_type = os.path.basename(dataset_folder).split(".")[-1]
        assert(self.data_type in ["jpg", "png", "npy"])

        import time
        t0 = time.time()
        # print('Start loading file addresses ...')
        # 获取图片列表
        images = glob.glob(dataset_folder)
        # 随机打乱图片
        random.shuffle(images)
        t = time.time() - t0
        # print('done! time:', t)
        print("Number of images found: %d" % len(images))

        self.images = images
        self.length = len(images)

    def __getitem__(self, idx):
        try:
            buf = self.images[idx]
            # 如果数据类型是npy
            if self.data_type == 'npy':
                # np读取
                img = np.load(buf)[0].transpose(1, 2, 0)
                # PIL读取
                img = Image.fromarray(img).convert("RGB")
            else:
                # 否则直接读图
                img = Image.open(buf).convert('RGB')
            # 做图像变换
            if self.transform is not None:
                img = self.transform(img)
            # 把图片封装到字典里
            data = {
                'image': img
            }
            return data
        except Exception as e:
            # 异常可能是index溢出导致，这时候随机生成index返回图片
            print(e)
            print("Warning: Error occurred when loading file %s" % buf)
            return self.__getitem__(np.random.randint(self.length))

    def __len__(self):
        # 返回数据集长度
        return self.length
