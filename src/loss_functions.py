import numpy as np 
import pandas as pd 
import math
import random 
import os 
from tqdm import tqdm # this module is useful to plot progress bars
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch


def euclidian_distance(weight, x):
    return np.sqrt( np.power(weight-x,2).sum() )

def euclidian_distance_torch(weight, x):
    return torch.sqrt( torch.pow(weight-x,2).sum() )


def distance_classes_loss_function(mus,labels,batch_size):
    high_value = max(0,len(mus))
    a = torch.randint(low=0, high=high_value, size=(32,))
    b = torch.randint(low=0, high=high_value, size=(32,))
    distances = [ 1/euclidian_distance_torch(mus[a[i]], mus[b[i]]) for i in range(a.shape[0]) if labels[a[i]]!=labels[b[i]]  ]
    loss = torch.stack(distances).mean()
    return loss    
    
def create_loss_function_by_pixel(thresholds_by_pixel):
    def threshold_by_pixel_loss_function(x_hat, x):
        diff = x_hat-x
        errors_by_pixel = torch.abs(diff) 
        errors_by_pixel = errors_by_pixel[:,0]
        loss = torch.stack([ torch.sum(errors_by_pixel[i]>thresholds_by_pixel)  for i in range(len(errors_by_pixel)) ])
        return loss
    return threshold_by_pixel_loss_function


def create_elbo_function(input_size=(3,256,256), ß=1):
    mse_function_locale = create_mse_function(input_size)
    def elbo_function(x_hat, x, y, mu, logvar):
        MSE = mse_function_locale(x_hat, x)
        KLD = divergence_function(mu,logvar)
        loss =  MSE + ß * KLD
        return loss
    return elbo_function

def create_elbo_edit_function(input_size=(3,256,256),batch_size=32, ß=1):
    mse_function_locale = create_mse_function(input_size)
    def elbo_function(x_hat, x, y, mu, logvar):
        MSE = mse_function_locale(x_hat, x)
        KLD = divergence_function(mu,logvar)
        labels = y
        distance_classes = distance_classes_loss_function(mu,labels,batch_size)
        loss =  MSE + ß * KLD + distance_classes

        print(f" loss: {loss}  mse: {MSE}  kld: {ß * KLD}  distance_classes: {distance_classes}")

        return loss
    return elbo_function


def create_mse_function(input_size=(3,256,256)):
    size = np.asarray(input_size).prod()
    def mse_function(x_hat, x):
        MSE = F.mse_loss(
            x_hat.view(-1, size), x.view(-1, size), reduction='mean'
        )
        return MSE 
    return mse_function


def create_loss_function(parameters, type="elbo", input_size=(3,256,256), ß=1):
    print(f"type loss function: {type}")
    if type=="elbo":
        return create_elbo_function(input_size=(3,256,256), ß=1)
    elif type=="elbo_edit":
        return create_elbo_edit_function(input_size,parameters["batch_size"], ß)
    elif type=="mse":
        return create_mse_function()
    elif type=="bce":
        return bce_function

    raise ValueError("doesn't know the type of loss function passed")


def bce_function(x_hat, x):
    BCE = nn.BCELoss(x_hat, x)
    return BCE 


def divergence_function(mu,logvar):
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    return KLD


