import os
from os.path import realpath
import torch
from skimage import io
import numpy as np
from util.config import cfg as test_cfg
from data.test_dataset import TestDataset
from util import util
from models.networks import RainNet
from models.normalize import RAIN

def load_network(cfg):
    net = RainNet(input_nc=cfg.input_nc, 
                output_nc=cfg.output_nc, 
                ngf=cfg.ngf, 
                norm_layer=RAIN, 
                use_dropout=not cfg.no_dropout)
    
    load_path = os.path.join(cfg.checkpoints_dir, cfg.name, 'net_G_last.pth')
    assert os.path.exists(load_path), print('%s not exists. Please check the file'%(load_path))
    print(f'loading the model from {load_path}')
    state_dict = torch.load(load_path, map_location='cpu')
    util.copy_state_dict(net.state_dict(), state_dict)
    # net.load_state_dict(state_dict)
    return net

def save_img(path, img):
    fold, name = os.path.split(path)
    if not os.path.exists(fold):
        os.makedirs(fold)
    io.imsave(path, img)

if __name__ == '__main__':
    comp_path = 'examples/1.png' # ['examples/1.png', 'examples/2.png']
    mask_path = 'examples/1-mask.png' # ['examples/1-mask.png', 'examples/2-mask.png']
    real_path = 'examples/1-gt.png' # ['examples/1-gt.png', 'examples/2-gt.png']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testdata = TestDataset(foreground_paths=comp_path, mask_paths=mask_path, background_paths=real_path, load_size=256)
    rainnet = load_network(test_cfg)
    rainnet = rainnet.to(device) # add
    
    for i in range(len(testdata)):
        sample = testdata[i]
        # inference
        comp = sample['comp'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        real = sample['real'].unsqueeze(0).to(device)
        img_path = sample['img_path']
        pred = rainnet.processImage(comp, mask, real)
        # save
        pred_rgb = util.tensor2im(pred[0:1])
        comp_rgb = util.tensor2im(comp[:1])
        mask_rgb = util.tensor2im(mask[:1])
        real_rgb = util.tensor2im(real[:1])
        print(img_path)
        save_img(img_path.split('.')[0] + '-results.png', np.hstack([comp_rgb, mask_rgb, pred_rgb]))

