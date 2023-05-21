import numpy as np 
import pandas as pd 
import random 
import os 
from tqdm import tqdm 
from torchvision.transforms import InterpolationMode
from src.utilities.utility_images import scale_img
import torch

from src.loss_functions import *
from src.utilities.utility_images import *

def add_noise(inputs,noise_factor=0.3):
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy

def scale_img(x,size):
     transform = transforms.Compose([transforms.Resize(size,InterpolationMode.BICUBIC),transforms.Resize(256,InterpolationMode.BICUBIC)])
     blurred_img = transform(x)
     return blurred_img

class Trainer_CAE():
    def __init__(self,strategy,input_size, cae, device,latent_dim):
        self.strategy = strategy
        self.vae = cae
        self.lr = self.strategy.lr
        self.device = device   
        self.b1,self.b2 = self.strategy.b1,self.strategy.b2
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.loss_function = torch.nn.MSELoss()
        self.latent_dim = latent_dim

    def train_epoch(self,dataloader):
        self.vae.train()
        train_loss,reg_loss = 0.0,0.0
        batch_index = 0
        l_zs,lista_indices = [],[]
        norm_grad,norm_weights = 0,0
        for batch in tqdm(dataloader):
            self.batch_index = batch_index
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0])

            indices = batch[2]
            x = batch[0]

            lista_indices.extend(indices.detach().cpu().numpy())

            x = x.to(self.device)
            x_hat,z,_,_ = self.vae(x)
            loss = self.loss_function(x_hat, x)
            train_loss+=loss.item()
  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_index += 1
            last_batch = batch
            z = z.detach().cpu().numpy()
            l_zs.extend(z)

        for i in range(len(x)):
            x_i, x_hat_i = x[i].detach().cpu().numpy(), x_hat[i].detach().cpu().numpy()

            x_i = convert2img(x_i)
            x_hat_i = convert2img(x_hat_i)            
            summary = np.hstack([x_i,x_hat_i]) 

            summary= summary.astype(np.uint8)
            summary = Image.fromarray(summary)
            index_training = self.strategy.index_training

        l_zs = np.asarray(l_zs)
        lista_indices = np.asarray(lista_indices)
        metrics_epoch = {"loss":train_loss/ len(dataloader)}
        other_data_epoch = { } 

        return metrics_epoch,other_data_epoch   

    def test_epoch(self,dataloader):
        return self.evaluate_data(dataloader)

    def evaluate_data(self, dataloader,test_loss_function=None):
        device = self.device
        xs,l_zs,losses,labels,lista_indices = [],[],[],[],[]
        lista_x_hats = []
        self.vae.eval()
        batch_index = 0
        for batch in tqdm(dataloader):
            self.batch_index = batch_index
            indices = batch[2]
            lista_indices.extend(indices.detach().cpu().numpy())
            x = batch[0].to(device)
            y = batch[1].to(device)
            batch_x_hat, z, _, _ = self.vae(x)
            loss = self.loss_function(batch_x_hat, x).item()

            batch_x_hat = batch_x_hat.detach().cpu().numpy()  
            y = y.detach().cpu().numpy()  
            y = list(y)

            xs.extend(x)
            for x_hat in batch_x_hat:
                x_hat = convert2img(x_hat,self.strategy.channels)
                lista_x_hats.append(x_hat)
            z = z.detach().cpu().numpy()
              
            l_zs.extend(z)
            losses.append(loss)
            labels.extend(y)
        
        loss = np.asarray(losses).mean()
        losses = np.asarray(losses)
        lista_indices = np.asarray(lista_indices)
        labels = np.asarray(labels)

        # metrics reported here are not used 
        metrics_epoch = {"loss":loss,"TP":10,"TN":10,"FN":0,"FP":0,"accuracy":0.9, 
                        "f1":0.9, "precision":0.9, "recall":0.9 }
        other_data_epoch = {} 

        return metrics_epoch,other_data_epoch

