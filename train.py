import time
from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def calculateMean(vars):
    return sum(vars) / len(vars)

def evaluateModel(model, opt, test_dataset, epoch, iters=None):
    model.netG.eval()
    if iters is not None:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s_iter%d.csv' % (epoch, iters))  # define the website directory
    else:
        eval_path = os.path.join(opt.checkpoints_dir, opt.name, 'Eval_%s.csv' % (epoch))  # define the website directory
    eval_results_fstr = open(eval_path, 'w')
    eval_results = {'mask': [], 'mse': [], 'psnr': []}

    for i, data in tqdm(enumerate(test_dataset)):
        model.set_input(data)  # unpack data from data loader
        model.test()  # inference
        visuals = model.get_current_visuals()  # get image results

        output = visuals['attentioned']
        real = visuals['real']
        # comp = visuals['comp']
        for i_img in range(real.size(0)):
            gt, pred = real[i_img:i_img+1], output[i_img:i_img+1]
            mse_score_op = mean_squared_error(util.tensor2im(pred), util.tensor2im(gt))
            psnr_score_op = peak_signal_noise_ratio(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
            # update calculator
            eval_results['mse'].append(mse_score_op)
            eval_results['psnr'].append(psnr_score_op)
            eval_results['mask'].append(data['mask'][i_img].mean().item())
            eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f\n' % (data['img_path'][i_img], eval_results['mask'][-1],
                                                                  mse_score_op, psnr_score_op))
        if i + 1 % 100 == 0:
            # print('%d images have been processed' % (i + 1))
            eval_results_fstr.flush()
    eval_results_fstr.flush()
    eval_results_fstr.close()

    all_mse, all_psnr = calculateMean(eval_results['mse']), calculateMean(eval_results['psnr'])
    print('MSE:%.3f, PSNR:%.3f' % (all_mse, all_psnr))
    model.netG.train()
    return all_mse, all_psnr, resolveResults(eval_results)

def resolveResults(results):
    interval_metrics = {}
    mask, mse, psnr = np.array(results['mask']), np.array(results['mse']), np.array(results['psnr'])
    interval_metrics['0.00-0.05'] = [np.mean(mse[np.logical_and(mask <= 0.05, mask > 0.0)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.05, mask > 0.0)])]
    interval_metrics['0.05-0.15'] = [np.mean(mse[np.logical_and(mask <= 0.15, mask > 0.05)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.15, mask > 0.05)])]
    interval_metrics['0.15-0.25'] = [np.mean(mse[np.logical_and(mask <= 0.25, mask > 0.15)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.25, mask > 0.15)])]
    interval_metrics['0.25-0.50'] = [np.mean(mse[np.logical_and(mask <= 0.5, mask > 0.25)]),
                                     np.mean(psnr[np.logical_and(mask <= 0.5, mask > 0.25)])]
    interval_metrics['0.50-1.00'] = [np.mean(mse[mask > 0.5]), np.mean(psnr[mask > 0.5])]
    return interval_metrics

def updateWriterInterval(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar('interval/{}-MSE'.format(k), v[0], epoch)
        writer.add_scalar('interval/{}-PSNR'.format(k), v[1], epoch)

if __name__ == '__main__':
    # setup_seed(6)
    opt = TrainOptions().parse()   # get training 
    train_dataset = CustomDataset(opt, is_for_train=True)
    test_dataset = CustomDataset(opt, is_for_train=False)
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    test_dataset_size = len(test_dataset)
    print('The number of training images = %d' % train_dataset_size)
    print('The number of testing images = %d' % test_dataset_size)
    
    train_dataloader = train_dataset.load_data()
    test_dataloader = test_dataset.load_data()
    print('The total batches of training images = %d' % len(train_dataset.dataloader))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))

    for epoch in range(opt.load_iter+1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in tqdm(enumerate(train_dataloader)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # evaluate for every epoch
        epoch_mse, epoch_psnr, epoch_interval_metrics = evaluateModel(model, opt, test_dataloader, epoch)
        writer.add_scalar('overall/MSE', epoch_mse, epoch)
        writer.add_scalar('overall/PSNR', epoch_psnr, epoch)
        updateWriterInterval(writer, epoch_interval_metrics, epoch)

        torch.cuda.empty_cache()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks('%d' % epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        print('Current learning rate: {}, {}'.format(model.schedulers[0].get_lr(), model.schedulers[1].get_lr()))

    writer.close()
