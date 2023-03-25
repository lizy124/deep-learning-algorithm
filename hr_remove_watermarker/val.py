import torch, os, sys, torchvision, argparse
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from option import opt, log_dir
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as tfs
from PIL import Image
from models.WHRNet import *
from models.config import config
from models.config import update_config
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import json


warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        default='models/seg_hrnet_w18_small_v2_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

args = parse_args()
models_={
	'whrnet': RemoveWatermarker(config, 3, 3)
}


model_name = opt.model_name
steps = opt.eval_step * opt.epochs
T = steps

def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr


class Inference:
	def __init__(self, path, model_dir, view_path) -> None:
		self.path = path
		self.hazy_names = os.listdir(self.path)
		self.view_path = view_path
		self.model_dir = model_dir
	
	def infer(self, net):
		opt.model_dir = self.model_dir
		print(os.path.exists(opt.model_dir))

		ckp = torch.load(opt.model_dir, map_location=torch.device('cuda:0'))
		print(f'resume from {opt.model_dir}')
		net.load_state_dict(ckp['model'])
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		print(f'max_psnr: {max_psnr} max_ssim: {max_ssim}')

		with torch.no_grad():
			self.test(net)


	def test(self, net):
		net.eval()
		img_size = 800
		torch.cuda.empty_cache()
		t0 = time.time()
		t_sum = 0
		for index, hazy_name in enumerate(self.hazy_names):
			hazy_path = self.path + hazy_name
			hazy_img = Image.open(hazy_path).convert('RGB')
			print('*****************', hazy_img.size)
			w, h = hazy_img.size
			ww, hh = w, h
			if max(w, h) > img_size:
				m = max(w, h)
				w = int(w/(m/img_size))
				h = int(h/(m/img_size))
			w_new = w - w%32 + 32
			h_new = h - h%32 + 32
			hazy_img = hazy_img.resize((w_new, h_new))
			T0 = time.time()
			hazy_img = tfs.ToTensor()(hazy_img)
			inputs = hazy_img.to(opt.device)
			with torch.no_grad():
				inputs = inputs.reshape([1, inputs.shape[0], inputs.shape[1], inputs.shape[2]])
				pred = net(inputs)
				pred = pred[0]
				pred = pred.permute(1, 2, 0)
				pred_arr = pred.cpu().numpy()*255
				pred_arr = cv2.cvtColor(pred_arr, cv2.COLOR_BGR2RGB)
				hazy_name = hazy_name.replace('perspective', 'perspective_2')
				tar_path = self.view_path + str(hazy_name)
				pred_arr = cv2.resize(pred_arr, [ww, hh])
				T1 = time.time() - T0
				t_sum += T1
				cv2.imwrite(tar_path, pred_arr)
		t1 = time.time() - t0
		print('t', t_sum)
		print('time resume:', t1)


if __name__ == "__main__":

	with open(f'./logs_train/args_{opt.model_name}.txt', 'w') as f:
		json.dump(opt.__dict__, f, indent=2)

	net = models_[opt.net]
	opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	net = net.to(opt.device)

	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total_params: ==> {}".format(pytorch_total_params))

	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	img_path = '/mnt/cfs/user/lzy/data/gene/ITS/watermarker_test/'
	model_dir = '/mnt/cfs/user/lzy/code/gene/hr_remove_watermarker/trained_models/watermarker_train_whrnet_test.pk.best'
	view_path = '/mnt/cfs/user/lzy/data/gene/ITS/view6/'
	inference = Inference(img_path, model_dir, view_path)
	inference.infer(net)
	

