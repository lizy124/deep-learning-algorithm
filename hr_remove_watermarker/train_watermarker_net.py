import torch, os, sys, torchvision, argparse
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from option import opt
from metrics import psnr, ssim
from models.WHRNet import RemoveWatermarker
from data_utils.watermarker import train_loader
from data_utils.watermarker import test_loader
import json
from models.config import config
from models.config import update_config
from models.tripletnet import Tripletnet

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class BasicOperate:
    def __init__(self) -> None:
        self.args, self.config = self.parse_args()
        self.model_name = opt.model_name
        self.models, self.loaders = self.get_model_dataset()

    def set_seed_torch(self, seed=2023):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def parse_args(self):
        yaml_file = 'models/seg_hrnet_w18_small_v2_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
        parser = argparse.ArgumentParser(description='Train segmentation network')
        parser.add_argument('--cfg', default=yaml_file, type=str)
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
        args = parser.parse_args()
        update_config(config, args)
        return args, config
    
    def get_model_dataset(self):
        models_ = {'whrnet': RemoveWatermarker(self.config, 3, 3)}
        loaders_ = {'watermarker_train': train_loader, 'watermarker_test': test_loader}
        return models_, loaders_

class TrainOprate(BasicOperate):
    def __init__(self) -> None:
        super().__init__()
        self.T = opt.eval_step * opt.epochs
        self.start_time = time.time()
        
    def lr_schedule_cosdecay(self, t, T, init_lr=opt.lr):
        lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
        return lr

    def res_lr_schedule_cosdecay(self, t, T, init_lr=opt.lr_res):
        lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
        return lr

    def train(self, net, loader_train, loader_test, optim, optim_res, criterion):
        losses, ssims, psnrs = [], [], []
        start_step, max_ssim, max_psnr = 0, 0, 0
        print(os.path.exists(opt.model_dir))

        if opt.resume:
            ckp = torch.load(opt.model_dir)
            print(f'resume from {opt.model_dir}')

            net.load_state_dict(ckp['model'])
            optim.load_state_dict(ckp['optimizer'])
            start_step, max_ssim, max_psnr = ckp['step'], ckp['max_ssim'], ckp['max_psnr']
            psnrs, ssims, losses = ckp['psnrs'], ckp['ssims'], ckp['losses']
            print(f'max_psnr: {max_psnr} max_ssim: {max_ssim}')
            print(f'start_step:{start_step} start training ---')
        else:
            print('train from scratch *****************')

        for step in range(start_step + 1, self.T + 1):
            net.train()

            lr = opt.lr
            if not opt.no_lr_sche:
                lr = self.lr_schedule_cosdecay(step, self.T)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr
            # lr_r = res_lr_schedule_cosdecay(step,T)
            # for param_group in optim_res.param_groups:
            # param_group["lr"] = lr_r

            x, y = next(iter(loader_train))
            x, y = x.to(opt.device), y.to(opt.device)
            out = net(x)

            loss_constast, all_ap, all_an, loss_rec = 0, 0, 0, 0
            if opt.w_loss_l1 > 0:
                loss_rec = criterion[0](out, y)
            if opt.w_loss_constast > 0:
                criterion[1].train()
                loss_constast = criterion[1](out, y, x)

            loss = opt.w_loss_l1 * loss_rec + opt.w_loss_constast * loss_constast
            loss.backward()

            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            
            l1 = opt.w_loss_l1*loss_rec
            constast = opt.w_loss_constast*loss_constast
            t = (time.time() - self.start_time) / 60
            print(
                f'\rloss:{loss.item():.5f} l1:{l1:.5f} contrast: {constast:.5f}| step :{step}/{self.T}|lr :{lr :.7f} |time_used :{t :.1f}',
                end='', flush=True)

            if step % opt.eval_step == 0:
                epoch = int(step / opt.eval_step)
                save_model_dir = opt.model_dir
                with torch.no_grad():
                    ssim_eval, psnr_eval = self.test(net, loader_test)
                log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'

                print(log)
                with open(f'./logs_train/{opt.model_name}.txt', 'a') as f:
                    f.write(log + '\n')
                ssims.append(ssim_eval)
                psnrs.append(psnr_eval)

                if psnr_eval > max_psnr:
                    max_ssim = max(max_ssim, ssim_eval)
                    max_psnr = max(max_psnr, psnr_eval)
                    save_model_dir = opt.model_dir + '.best'
                    print(
                        f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')

                torch.save({
                    'epoch': epoch, 'step': step, 'max_psnr': max_psnr, 'max_ssim': max_ssim,
                    'ssims': ssims, 'psnrs': psnrs, 'losses': losses, 'model': net.state_dict(),
                    'optimizer': optim.state_dict()}, save_model_dir)

    def test(self, net, loader_test):
        net.eval()
        torch.cuda.empty_cache()
        ssims, psnrs = [], []

        for i, (inputs, targets) in enumerate(loader_test):
            inputs = inputs.to(opt.device);
            targets = targets.to(opt.device)
            with torch.no_grad():
                pred = net(inputs)
            ssim1 = ssim(pred, targets).item()
            psnr1 = psnr(pred, targets)
            ssims.append(ssim1)
            psnrs.append(psnr1)

        return np.mean(ssims), np.mean(psnrs)


class Remove:
    def __init__(self) -> None:
        self.check_log()

    def check_log(self):
        log_path = '/mnt/cfs/user/lzy/code/remote/deep-learning-for-image-processing/aigc/hr_remove_watermarker'
        if not opt.resume and os.path.exists(f'/logs_train/{opt.model_name}.txt'):
            print(log_path + f'/logs_train/{opt.model_name}.txt 已存在，请删除该文件……')
            exit()

        with open(log_path + f'/logs_train/args_{opt.model_name}.txt', 'w') as f:
            json.dump(opt.__dict__, f, indent=2)
        
    def create_criterion(self):
        criterion = []
        criterion.append(nn.L1Loss().to(opt.device))
        criterion.append(Tripletnet(margin=opt.margin))
        return criterion
    
    def print_params(self, loader_train, net, criterion):
        epoch_size = len(loader_train)
        print("epoch_size: ", epoch_size)

        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: ==> {}".format(pytorch_total_params))

        pytorch_total_params_resnet = sum(p.numel() for p in criterion[1].parameters() if p.requires_grad)
        print("Total_params_resnet: ==> {}".format(pytorch_total_params_resnet))
    
    def run(self):

        watermarker = TrainOprate()
        watermarker.set_seed_torch(666)
        criterion = self.create_criterion()
        
        loader_train = watermarker.loaders[opt.trainset]
        loader_test = watermarker.loaders[opt.testset]
        net = watermarker.models[opt.net]
        opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(opt.device)

        self.print_params(loader_train, net, criterion)

        params=filter(lambda x: x.requires_grad, net.parameters())
        optimizer = optim.Adam(params=params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
        optimizer.zero_grad()

        watermarker.train(net, loader_train, loader_test, optimizer, '', criterion)


if __name__ == "__main__":
    remove = Remove()
    remove.run()


