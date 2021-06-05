# -*- utf-8 ----
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from util import util
import torch
from torch.utils.data import DataLoader
from skimage import data, io
from skimage.measure import mean_squared_error
from skimage.measure import peak_signal_noise_ratio
from util.config import cfg
from models.networks import RainNet
from models.normalize import RAIN
from data.iharmony4_dataset import Iharmony4Dataset


def calculateMean(vars):
    return sum(vars) / len(vars)

def load_network(cfg):
    net = RainNet(input_nc=cfg.input_nc, 
                output_nc=cfg.output_nc, 
                ngf=cfg.ngf, 
                norm_layer=RAIN, 
                use_dropout=not cfg.no_dropout)
    
    load_path = os.path.join(cfg.checkpoints_dir, cfg.name, 'net_G.pth')
    if not os.path.exists(load_path):
        raise FileExistsError, print('%s not exists. Please check the file'%(load_path))
    print(f'loading the model from {load_path}')
    state_dict = torch.load(load_path)
    util.copy_state_dict(net.state_dict(), state_dict)
    # net.load_state_dict(state_dict)
    return net

def save_img(path, img):
    fold, name = os.path.split(path)
    os.makedirs(fold, exist_ok=True)
    io.imsave(path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='none', required=False, type=str, help='the path of the dataset for evaluation')
    parser.add_argument('--batch_size', default=16, required=False, type=int, help='batchsize of inference')
    parser.add_argument('--save_dir', default='evaluate', required=False, type=str, help='directory to save evaluating results')
    parser.add_argument('--store_image', action='store_true', required=False, help='whether store the result images')
    parser.add_argument('--device', default='cuda', type=str, help='device to running the code | default cuda')
    user_args = parser.parse_args()
    if user_args.dataset_root is not 'none':
        cfg.dataset_root = user_args.dataset_root

    # ----------------- main --------------------
    device = user_args.device
    assert device.startswith('cpu') or device.startswith('cuda'), 'Device setting error. Please check --device in the arguments'

    testdata = Iharmony4Dataset(cfg, is_for_train=False)
    testloader = DataLoader(testdata, batch_size=user_args.batch_size, shuffle=False, drop_last=False)
    net = load_network(cfg)
    net = net.to(device)
    net.eval()

    os.makedirs(user_args.save_dir, exist_ok=True)
    fsave_results = open(os.path.join(user_args.save_dir, 'test_results.csv'), 'w')
    fsave_results.writelines('image_path,foreground_ratio,MSE,PSNR\n')
    all_psnr, all_mse = [], []
    for i, batch_data in enumerate(tqdm(testloader)):
        comp = batch_data['comp'].to(device)
        mask = batch_data['mask'].to(device)
        real = batch_data['real'].to(device)
        pred = net.processImage(comp, mask)
        for img_idx in range(comp.size(0)):
            img_path = str(batch_data['img_path'][img_idx])
            comp_rgb = util.tensor2im(comp[img_idx:img_idx+1]) # input shape should be (1, 3, H, W) or (1, 1, H, W)
            pred_rgb = util.tensor2im(pred[img_idx:img_idx+1])
            mask_rgb = util.tensor2im(mask[img_idx:img_idx+1])
            real_rgb = util.tensor2im(real[img_idx:img_idx+1])
            mse_score_op = mean_squared_error(pred_rgb, real_rgb)
            psnr_score_op = peak_signal_noise_ratio(pred_rgb, real_rgb)
            all_psnr.append(psnr_score_op)
            all_mse.append(mse_score_op)

            fsave_results.writelines('%s,%.2f,%.2f,%.2f\n' % (img_path, mask[img_idx].mean().item(), mse_score_op, psnr_score_op))
            if user_args.store_image:
                basename, imagename = os.path.split(img_path)
                basename = basename.split('/')[-2] # HAdobe, HCOCO etc...
                save_img(os.path.join(user_args.save_dir, basename, imagename.split('.')[0] + '.png'), 
                        np.hstack([comp_rgb, mask_rgb, real_rgb, pred_rgb]))
            
        if i+1 % 50 == 0:
            fsave_results.flush()
    print('MSE: %.4f  PSNR: %.4f' % (calculateMean(all_psnr), calculateMean(all_mse)))
    fsave_results.flush()
    fsave_results.close()
    
