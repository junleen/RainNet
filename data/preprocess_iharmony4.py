import os
import cv2
from tqdm import tqdm
import argparse

def parsePaths(path):
    name_parts=path.split('_')
    mask_path = path.replace('composite_images','masks')
    mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
    target_path = path.replace('composite_images','real_images')
    target_path = target_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
    return path, mask_path, target_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_iharmony4', type=str, required=True, help='data directory to iHarmony4 dataset')
    parser.add_argument('--save_dir', type=str, default='none', help='data directory to save the resized images. Default none')
    parser.add_argument('--image_size', type=int, default=512, help='image size to save in the local device. Default 512')
    parser.add_argument('--keep_aspect_ratio', action='store_true', help='if keep the height-width aspect ratio unchanged. Default False')
    args = parser.parse_args()

    with open(os.path.join(args.dir_iharmony4, 'IHD_train.txt'), 'r') as f:
        train_list = f.readlines()
        train_list = [item.strip() for item in train_list]

    with open(os.path.join(args.dir_iharmony4, 'IHD_test.txt'), 'r') as f:
        test_list = f.readlines()
        test_list = [item.strip() for item in test_list]

    sub_dataset = ['HAdobe5k', 'HCOCO', 'Hday2night', 'HFlickr']
    data_root = args.dir_iharmony4
    if args.save_dir == 'none':
        save_path = os.path.join(os.path.split(data_root)[0], 'iHarmony4Resized')
    else:
        save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    os.system('cp %s %s' % (os.path.join(args.dir_iharmony4, 'IHD_*'), save_path))

    target_size = args.image_size
    keep_aspect_ratio = args.keep_aspect_ratio

    for item in sub_dataset:
        i_save_path = os.path.join(save_path, item)
        os.makedirs(i_save_path, exist_ok=True)
        os.makedirs(os.path.join(i_save_path, 'composite_images'), exist_ok=True)
        os.makedirs(os.path.join(i_save_path, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(i_save_path, 'real_images'), exist_ok=True)

    # start to read and rewrite the images
    for running_list in [train_list, test_list]:
        for item in tqdm(running_list):
            comp_path, mask_path, real_path = parsePaths(item)
            save_comp_path = os.path.join(save_path, comp_path) # target path of the composite image
            save_mask_path = os.path.join(save_path, mask_path) # target path of the mask image
            save_real_path = os.path.join(save_path, real_path) # target path of the real image
            # if you have saved the image before, then skip this image
            if os.path.exists(save_comp_path):
                continue
            # read the images
            comp = cv2.imread(os.path.join(data_root, comp_path))
            size = comp.shape[:2]
            scale = target_size / min(size)
            if keep_aspect_ratio:
                new_size = (int(scale * size[0]), int(scale * size[1])) # the new image keep its width-height ratio
            else:
                new_size = (target_size, target_size)
                
            # resize the image
            comp = cv2.resize(comp, (new_size[1], new_size[0]), cv2.INTER_CUBIC)
            mask = cv2.imread(os.path.join(data_root, mask_path))
            mask = cv2.resize(mask, (new_size[1], new_size[0]), cv2.INTER_NEAREST)
            real = cv2.imread(os.path.join(data_root, real_path))
            real = cv2.resize(real, (new_size[1], new_size[0]), cv2.INTER_CUBIC)

            cv2.imwrite(save_comp_path, comp)
            cv2.imwrite(save_mask_path, mask)
            cv2.imwrite(save_real_path, real)