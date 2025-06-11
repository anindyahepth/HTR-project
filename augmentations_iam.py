import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
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

class ErosionDilationElasticRandomTransform:
    """
    Applies erosion, dilation, elastic distortion, random transformation, and color jitter augmentations.
    """

    def __init__(self, erosion_kernel_size=(2, 2), erosion_iterations=1,
                 dilation_kernel_size=(2, 2), dilation_iterations=1,
                 elastic_alpha=0.8, elastic_sigma=5,
                 random_angle=0, random_translate=(0.1, 0.1), random_scale=(0.9, 1.1),
                 brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
        """
        Initializes the augmentations.

        Args:
            erosion_kernel_size (tuple): Size of the erosion kernel (width, height).
            erosion_iterations (int): Number of erosion iterations.
            dilation_kernel_size (tuple): Size of the dilation kernel (width, height).
            dilation_iterations (int): Number of dilation iterations.
            elastic_alpha (float): Elastic distortion alpha parameter.
            elastic_sigma (float): Elastic distortion sigma parameter.
            random_angle (float): Max random rotation angle in degrees.
            random_translate (tuple): Max random translation fraction (width, height).
            random_scale (tuple): Random scale range (min, max).
            brightness (float): Brightness jitter factor.
            contrast (float): Contrast jitter factor.
            saturation (float): Saturation jitter factor.
            hue (float): Hue jitter factor.
        """
        kernel_h = utils.randint(1, 3)
        kernel_w = utils.randint(1, 3)             
        self.erosion_kernel = (kernel_h, kernel_w)    #np.ones(erosion_kernel_size, np.uint8)
        self.erosion_iterations = erosion_iterations
        self.dilation_kernel = (kernel_h, kernel_w)   #np.ones(dilation_kernel_size, np.uint8)
        self.dilation_iterations = dilation_iterations
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.random_angle = random_angle
        self.random_translate = random_translate
        self.random_scale = random_scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _elastic_transform(self, image, alpha, sigma):
        """Applies elastic distortion to the input NumPy array."""
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        dx = cv2.GaussianBlur(dx, (sigma | 1, sigma | 1), 0) * alpha
        dy = cv2.GaussianBlur(dy, (sigma | 1, sigma | 1), 0) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

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

        if random.random() < 0.7:
            img_np = self._elastic_transform(img_np, self.elastic_alpha, self.elastic_sigma)

        if random.random() < 0.7:
            angle = random.uniform(-self.random_angle, self.random_angle)
            translate = (random.uniform(-self.random_translate[0], self.random_translate[0]),
                         random.uniform(-self.random_translate[1], self.random_translate[1]))
            scale = random.uniform(self.random_scale[0], self.random_scale[1])
            img_pil = TF.affine(img_pil, angle, [int(img_pil.width * translate[0]), int(img_pil.height * translate[1])], scale, 0)

        img_pil = Image.fromarray(img_np)

        img_pil = TF.adjust_brightness(img_pil, 1 + random.uniform(-self.brightness, self.brightness))
        img_pil = TF.adjust_contrast(img_pil, 1 + random.uniform(-self.contrast, self.contrast))
        img_pil = TF.adjust_saturation(img_pil, 1 + random.uniform(-self.saturation, self.saturation))
        img_pil = TF.adjust_hue(img_pil, random.uniform(-self.hue, self.hue))

        if isinstance(img, torch.Tensor):
            return TF.to_tensor(img_pil)
        else:
            return img_pil



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
