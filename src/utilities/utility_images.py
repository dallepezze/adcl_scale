from torchvision import transforms as transforms
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from matplotlib import image as image_functions
from torchvision.transforms import InterpolationMode
import time
import json

from src.datasets import *


#assuming a numpy matrix (img_size,img_size) in input
def matrix2d_to_image(matrix2d):
    mask_img =  Image.fromarray(np.uint8(matrix2d) , 'L')
    mask_img = mask_img.convert('RGB')
    mask_img = np.asarray(mask_img)
    return mask_img

#take numpy img and scale it and return it as numpy object
def resize_numpy_img(img,scale_size, channels=3):
    img = img.astype(np.uint8)
    # if channels==1:
    #       img = Image.fromarray(img, "L")
    #       img = img.resize((scale_size,scale_size), Image.ANTIALIAS)
    # else:
    img = Image.fromarray(img)
    img = img.resize((scale_size,scale_size), Image.ANTIALIAS)
    return np.asarray(img)


import torch.nn.functional as F
#assuming tensor img as (#channels,h,w)
def resize_torch_img(img,scale_size, gray=False):
    num_channels = 3 if gray is False else 1
    #print(img.shape)
    original_size = img.shape[0]
    if gray is False:
        img = torch.permute(img,(1, 2, 0))
    img = img.reshape(1,num_channels,original_size,original_size)
    img = F.interpolate(img, size=scale_size, mode="bicubic") #Image.ANTIALIAS
    return img[0][0]

#assuming a numpy matrix (img_size,img_size) in input
def matrix2d_to_heatmap(matrix2d,img_size):
    heatmap_img = cv2.applyColorMap(np.uint8(matrix2d), cv2.COLORMAP_JET)
    return heatmap_img

def load_img(filepath):
    # load image as pixel array
    image = image_functions.imread(filepath)
    return image

def create_transform_img(scale_size):
    transform = []
    #transform.append(transforms.ColorJitter(brightness=opt.brightness))
    transform.append(transforms.Resize((scale_size,scale_size), Image.BICUBIC))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform)

from torchvision.transforms.functional import InterpolationMode
def create_transform_x(crp_size):
    transform = []
    transform.append(transforms.Resize((crp_size, crp_size), interpolation=2))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform)

def create_transform_x_with_rotation(crp_size,rotation_degree,fill=0):
    transform = []
    transform.append(transforms.Resize((crp_size, crp_size), interpolation=2))
    random_rotation = torchvision.transforms.RandomAffine(degrees=rotation_degree,resample=PIL.Image.BICUBIC,fill=fill)
    #random_rotation = torchvision.transforms.functional.rotate(rotation_degree,interpolation=InterpolationMode.BICUBIC, fill=fill)
    transform.append( random_rotation )
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform)


def convert2img(image,gray=False, normalize=True,imtype=np.uint8):
    if not isinstance(image, np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.data
        else:
            return image
        image = image.cpu().numpy()
        assert len(image.squeeze().shape) < 4
    if len(image.shape)==2:
        image = np.expand_dims(image, axis=-1)
    if (len(image.shape)==3 and image.shape[2]==1):
        min_score_value = image.min()
        max_score_value = image.max()
        anomaly_map = (image.copy() - min_score_value) / (max_score_value -  min_score_value)
        heatmap = cv2.resize(anomaly_map, (256,256))
        plt.matshow(heatmap)
        plt.show()
        heatmapshow = None
        heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        im_color = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        im_pil = Image.fromarray(im_color)
        im_np = np.asarray(im_pil)
        image = im_np
    if image.dtype != np.uint8:
        if normalize == True:
            #denormalize image if the image is normalized
            image = np.transpose(image, (1,2,0))
            image = ((image * normalazition_parameters_mvtec["std"]) + normalazition_parameters_mvtec["mean"]) * 255
            # image = (np.transpose(image, (1, 2, 0)) * 0.5 + 0.5) * 255
        else:
            #not denormalize
           image = (np.transpose(image, (1, 2, 0))) * 255
           print(f"5)image.shape: {image.shape}")
    plt.close("all")
    return image.astype(imtype) 

def renormalize_img(img):
    img = (img* 0.5 + 0.5) * 255
    return img

def scale_img(x,size):
     blurred_img = transforms.Resize(size,InterpolationMode.BICUBIC)(x)
     blurred_img = transforms.Resize(256,InterpolationMode.BICUBIC)(blurred_img)
     return blurred_img



