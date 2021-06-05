# This config file is inspired by https://github.com/Yadiraf/DECA/decalib/utils/config.py
# Testing config for our RainNet

from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()
# ------ dataloader -------------
cfg.dataset_root = '../dataset/iHarmony4'
cfg.dataset_mode = 'iharmony4'
cfg.batch_size = 10
cfg.beta1 = 0.5
cfg.checkpoints_dir = './checkpoints'
cfg.crop_size = 256
cfg.load_size = 256
cfg.num_threads = 11
cfg.preprocess = 'none' # 
# ------ model -------------
cfg.gan_mode = 'wgangp'
cfg.model = 'rainnet'
cfg.netG = 'rainnet'
cfg.normD = 'instance'
cfg.normG = 'RAIN'
cfg.is_train = False
cfg.input_nc = 3
cfg.output_nc = 3
cfg.ngf = 64
cfg.no_dropout = False
# ------ training -------------
cfg.name = 'experiment_train'
cfg.gpu_ids = 0
cfg.lambda_L1 = 100
cfg.print_freq = 400
cfg.continue_train = False
cfg.load_iter = 0
cfg.niter = 100
cfg.niter_decay = 0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
