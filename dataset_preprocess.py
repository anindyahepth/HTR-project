import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torchvision import transforms

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from functools import partial
import argparse
from collections import OrderedDict
import ast
from torch.utils.data import Dataset

import torch
import torchvision.transforms.functional as TF
import random
import cv2
import numpy as np
from PIL import Image


from scipy.ndimage import map_coordinates


import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageFilter
import random
from datasets import load_dataset

#--------Augmentation pipeline---------
def augmentation_pipeline(image, p=0.5):
    """
    Applies a series of image augmentations to a PIL Image with a given probability.

    Args:
        image (PIL.Image.Image): The input PIL Image.
        p (float, optional): The probability of applying each augmentation. Defaults to 0.5.

    Returns:
        PIL.Image.Image: The augmented PIL Image.
    """
    # 1. Random Affine Transformation
    if random.random() < p:
        affine_params = {
            'scale': (0.8, 2.0),  # Scale factor range
            'translation': (-10, 10),  # Translation range in pixels
            'rotation': (0, 0),  # Rotation range in degrees
            'shear': (-10, 10),  # Shear range in degrees
            'fillcolor': 255  # Fill color for the areas outside the transformed image
        }
        scale = random.uniform(affine_params['scale'][0], affine_params['scale'][1])
        translation_x = random.randint(affine_params['translation'][0], affine_params['translation'][1])
        translation_y = random.randint(affine_params['translation'][0], affine_params['translation'][1])
        rotation = random.uniform(affine_params['rotation'][0], affine_params['rotation'][1])
        shear_x = random.uniform(affine_params['shear'][0], affine_params['shear'][1])
        shear_y = random.uniform(affine_params['shear'][0], affine_params['shear'][1])

        image = image.transform(
            (1510, 128),  # Output size
            PIL.Image.AFFINE,
            (
                scale * 1, 0, translation_x,
                0, scale * 1, translation_y,
                shear_x, shear_y, 1
            ),
            resample=PIL.Image.Resampling.BILINEAR,
            fillcolor=affine_params['fillcolor']
        )

    # 2. Color Jitter
    if random.random() < p:
        color_params = {
            'brightness': (0.4, 0.4),  # Brightness factor range
            'contrast': (0.4, 0.4),  # Contrast factor range
            'saturation': (0.4, 0.4),  # Saturation factor range
            'hue': (0.2, 0.2),  # Hue factor range
        }
        brightness = random.uniform(color_params['brightness'][0], color_params['brightness'][1])
        contrast = random.uniform(color_params['contrast'][0], color_params['contrast'][1])
        saturation = random.uniform(color_params['saturation'][0], color_params['saturation'][1])
        hue = random.uniform(color_params['hue'][0], color_params['hue'][1])

        image = PIL.ImageEnhance.Brightness(image).enhance(brightness)
        image = PIL.ImageEnhance.Contrast(image).enhance(contrast)
        image = PIL.ImageEnhance.Color(image).enhance(saturation)
        #image = image.rotate(hue * 180)  # Hue is a rotation on the color wheel

    # 3. Gaussian Blur
    if random.random() < p:
        blur_radius = random.uniform(0.1, 1.5)  # Blur radius range
        image = image.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius))

    # 4. Resize (Always applied, but you can add probability if needed)
    image = image.resize((1024, 64), resample=PIL.Image.Resampling.BILINEAR)

    return image

#-----Torch Dataset class ---------

class IAMDataset(Dataset):

    def __init__(self, dataset, converter, transform=transforms.ToTensor(), augment=False):
        self.dataset = dataset
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']  # Assuming 'image' column contains PIL Images
        label = item['text']  # Assuming 'label' column

        if self.augment:
            image = augmentation_pipeline(image)

        image = image.convert("RGB")

        if self.transform:
            pixel_values = self.transform(image)



        # Tokenize the label
        labels, length = converter.encode(label)

        encoding = {"pixel_values": pixel_values, "labels": labels}

        return encoding
