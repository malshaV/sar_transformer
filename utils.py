import os
import numpy as np
import torch

from skimage import io,color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable
import os
import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict

from scipy.io import loadmat


class BSD_SAR(Dataset):
    """
    Reads the synthetic images (created useing create_synthetic_data.py) saved as .mat files .
    """

    def __init__(self, dataset_path, crop_size, training_set=True) -> None:
        self.dataset_path = dataset_path
        # self.input_path = os.path.join(dataset_path, 'noisy')
        # self.output_path = os.path.join(dataset_path, 'clean')
        self.images_list = os.listdir(self.dataset_path)
        self.training_set = training_set

        self.crop = crop_size
        


    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        

        # read .dat file
        data_SAR = loadmat(os.path.join(self.dataset_path, image_filename))

        # get noisy image and numpy to tensor
        image = data_SAR['noisy']
        image = np.sqrt(image + 1e-10)
        image = F.to_pil_image(image)
        
    

        # get clean image and numpy to tensor
        mask = data_SAR['clean']
        mask = np.sqrt(mask + 1e-10)
        mask = F.to_pil_image(mask)
        
        # print(image.shape)
        # print(mask.shape)

        # # read noisy image
        # image = cv2.imread(os.path.join(self.input_path, image_filename),0)
        
        # # read clean image
        # mask = cv2.imread(os.path.join(self.output_path, image_filename),0)
        
        # # transforming to PIL image
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        if self.training_set:
            # # random resized crop
            # i, j, h, w = T.RandomResizedCrop.get_params(image,scale= (0.12, 1.0), ratio=(1, 1))
            # image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

            # random crop
            i, j, h, w = T.RandomCrop.get_params(image, output_size= self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        

            # # resize
            # image, mask = F.resize(image,self.crop), F.resize(mask,self.crop)

            # rotation
            a = T.RandomRotation.get_params((-90, 90))
            image, mask = F.rotate(image, a), F.rotate(mask, a)


            # random horizontal flipping
            if np.random.rand() < 0.5:
                image, mask = F.hflip(image), F.hflip(mask)

            # random affine transform
            if np.random.rand() < 0.5:
                affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
                image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        else:
            # random crop
            i, j, h, w = T.RandomCrop.get_params(image, output_size= self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)


        # transforming to tensor
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        

        return image, mask, image_filename



