import os.path
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TestDataset(Dataset):
    def __init__(self, 
                 foreground_paths,
                 mask_paths, 
                 background_paths=None,
                 load_size=256):
        '''
        foreground_paths: [folder, imagepath_list, image path]
        mask_paths: [folter, imagepath_list, image path]
        background_paths: [folter, imagepath_list, image path]
        '''
        self.foreg_paths, self.mask_paths, self.backg_paths = \
            foreground_paths, mask_paths, background_paths
        
        self._load_images_paths()
        self._load_sizez = load_size
        self.transform_image = transforms.Compose([
            transforms.Resize([load_size, load_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize([load_size, load_size]),
            transforms.ToTensor()
        ])

    def _load_images_paths(self, ):
        if isinstance(self.foreg_paths, list):
            pass
        elif os.path.isdir(self.foreg_paths): 
            self.foreg_paths = glob(self.foreg_paths + '/*.jpg') + \
                               glob(self.foreg_paths + '/*.png') + \
                               glob(self.foreg_paths + '/*.bmp')
            self.mask_paths = glob(self.mask_paths + '/*.png')
            if self.backg_paths is not None:
                self.backg_paths = glob(self.mask_paths + '/*.png') + \
                                   glob(self.backg_paths + '/*.jpg') + \
                                   glob(self.backg_paths + '/*.bmp')
        elif os.path.isfile(self.foreg_paths) and (self.foreg_paths[-3:] in ['jpg', 'png', 'bmp']):
            self.foreg_paths = [self.foreg_paths]
            self.mask_paths = [self.mask_paths]
            if self.backg_paths is not None:
                self.backg_paths = [self.backg_paths]
        else:
            print(f'please check the test path: {self.foreg_paths} {self.mask_paths} {self.backg_paths}')
            exit()
        self._sort()
        print(f'total {len(self.foreg_paths)} images')
    
    def _sort(self):
        self.foreg_paths.sort()
        self.mask_paths.sort()
        if self.backg_paths is not None:
            self.backg_paths.sort()

    def __getitem__(self, index):
        comp = self.transform_image(Image.open(self.foreg_paths[index]).convert('RGB'))
        mask = self.transform_mask(Image.open(self.mask_paths[index]).convert('1'))
        
        if self.backg_paths is not None:
            real = self.transform_image(Image.open(self.backg_paths[index]).convert('RGB'))
            comp = self._compose(comp, mask, real)
        else:
            real = self.transform_image(Image.open(self.foreg_paths[index]).convert('RGB'))
        
        return {'comp': comp, 'mask': mask, 'real': real, 'img_path':self.foreg_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.foreg_paths)

    def _compose(self, foreground_img, foreground_mask, background_img):
        return foreground_img * foreground_mask + background_img * (1 - foreground_mask)
