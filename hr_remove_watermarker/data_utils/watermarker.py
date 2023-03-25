import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader
from metrics import *
from option import opt



class Dataset(data.Dataset):
    def __init__(self, path, train, size=500, format='.png'):
        super(Dataset,self).__init__()
        self.size = size
        self.train=train
        self.path = path
        print(f'path={path}')
        print('crop size',size)
        self.hazy_names = os.listdir(self.path+'hazy')

    def __getitem__(self, index):
        hazy_name = self.hazy_names[index]
        img_idx = hazy_name.split('_')[0]
        hazy_path = self.path + 'hazy/' + hazy_name
        # clear_path = self.path + 'clear/' + img_idx + '.png'
        clear_path = self.path + 'clear/' + hazy_name
        hazy_img = Image.open(hazy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        hazy_img = hazy_img.resize((500, 500))
        clear_img = clear_img.resize((500, 500))
        haze, clear = self.augData(hazy_img, clear_img)
        return haze, clear

    def augData(self, data, target):

        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90*rand_rot)
                target = FF.rotate(target, 90*rand_rot)

        data = tfs.ToTensor()(data)
        #data = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data) # 加上归一化
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.hazy_names)

root = '/mnt/cfs/user/lzy/data/gene/watermarker/'

train_loader=DataLoader(dataset=Dataset(root+'train/', train=True,size='whole img'),batch_size=opt.bs,shuffle=True)
test_loader=DataLoader(dataset=Dataset(root+'test/',train=False,size='whole img'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass
