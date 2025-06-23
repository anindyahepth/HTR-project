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


#-------Simpler Augmentation ---------

class ErosionDilationColorJitterTransform:
    """
    Applies erosion, dilation, and color jitter augmentations to PIL Images or torch tensors.
    """

    def __init__(self, erosion_kernel_size=(3, 3), erosion_iterations=1,
                 dilation_kernel_size=(3, 3), dilation_iterations=1,
                 brightness=0, contrast=0, saturation=0, hue=0):
        """
        Initializes the augmentations.

        Args:
            erosion_kernel_size (tuple): Size of the erosion kernel (width, height).
            erosion_iterations (int): Number of erosion iterations.
            dilation_kernel_size (tuple): Size of the dilation kernel (width, height).
            dilation_iterations (int): Number of dilation iterations.
            brightness (float): Brightness jitter factor.
            contrast (float): Contrast jitter factor.
            saturation (float): Saturation jitter factor.
            hue (float): Hue jitter factor.
        """
        self.erosion_kernel = np.ones(erosion_kernel_size, np.uint8)
        self.erosion_iterations = erosion_iterations
        self.dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
        self.dilation_iterations = dilation_iterations
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        """
        Applies augmentations to the input PIL Image or torch tensor.

        Args:
            img (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            PIL.Image.Image or torch.Tensor: Augmented image.
        """
        if isinstance(img, torch.Tensor):
            img_pil = TF.to_pil_image(img)
        else:
            img_pil = img

        img_np = np.array(img_pil)

        if random.random() < 0.7:
            img_np = cv2.erode(img_np, self.erosion_kernel, iterations=self.erosion_iterations)

        if random.random() < 0.7:
            img_np = cv2.dilate(img_np, self.dilation_kernel, iterations=self.dilation_iterations)

        img_pil = Image.fromarray(img_np)

        img_pil = TF.adjust_brightness(img_pil, 1 + random.uniform(-self.brightness, self.brightness))
        img_pil = TF.adjust_contrast(img_pil, 1 + random.uniform(-self.contrast, self.contrast))
        img_pil = TF.adjust_saturation(img_pil, 1 + random.uniform(-self.saturation, self.saturation))
        img_pil = TF.adjust_hue(img_pil, random.uniform(-self.hue, self.hue))

        if isinstance(img, torch.Tensor):
            return TF.to_tensor(img_pil)
        else:
            return img_pil


#-----Torch Dataset class ---------

class IAMDataset(Dataset):

    def __init__(self, dataset, converter, transform=transforms.ToTensor(), augment=False):
        self.dataset = dataset
        self.augment = augment
        self.transform = transform
        self.converter = converter

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
        labels, length = self.converter.encode(label)

        encoding = {"pixel_values": pixel_values, "labels": labels}

        return encoding
