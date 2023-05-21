import numpy as np 
from tqdm import tqdm 

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.loss_functions import *
from src.utilities.utility_images import convert2img, scale_img


def create_elbo_function(input_size=(3,256,256), ß=1):
    mse_function_locale = create_mse_function(input_size)
    def elbo_function(x_hat, x, y, mu, logvar):
        MSE = mse_function_locale(x_hat, x)
        KLD = divergence_function(mu,logvar)
        loss =  MSE + ß * KLD
        return loss
    return elbo_function


def create_mse_function(input_size=(3,256,256)):
    size = np.asarray(input_size).prod()
    def mse_function(x_hat, x):
        MSE = F.mse_loss(
            x_hat.view(-1, size), x.view(-1, size), reduction='sum')
        return MSE 
    return mse_function

class Trainer_VAE():
    def __init__(self, strategy,vae, batch_size, latent_dim, device):
        self.strategy = strategy
        self.vae = vae
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.lr = self.strategy.lr
        self.b1,self.b2 = self.strategy.b1,self.strategy.b2
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.device = device   
        self.elbo_loss = create_elbo_function((strategy.channels,strategy.img_size,strategy.img_size))
        self.mse_loss = create_mse_function((strategy.channels,strategy.img_size,strategy.img_size))

    def train_epoch(self,dataloader):
        self.vae.train()
        train_loss,reg_loss = 0.0,0.0
        batch_index = 0
        l_zs,lista_indices = [],[]
        # save mu and log var during training
        l_mus,l_logvar = [],[]
        norm_grad,norm_weights = 0,0
        l_y = []

        for batch in tqdm(dataloader):
            self.batch_index = batch_index
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0])
            indices = batch[2]
            x = batch[0]
            y = batch[1]
            lista_indices.extend(indices.detach().cpu().numpy())
            x = x.to(self.device)
            x_hat, z, mu, logvar = self.vae(x)
            loss = self.elbo_loss(x_hat, x, y, mu, logvar) / x.size(0)
            train_loss+=loss.item()
  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_index += 1
            last_batch = batch
            z = z.detach().cpu().numpy()
            l_zs.extend(z)
            l_mus.extend(mu.detach().cpu().numpy())
            l_y.extend(y.detach().cpu().numpy())
            l_logvar.extend(logvar.detach().cpu().numpy())
            for _, param in self.vae.named_parameters():
                    norm_grad += torch.abs(param.grad.data.cpu()).sum()
                    norm_weights += param.data.cpu().pow(2).sum() 

        for i in range(len(x)):
            x_i, x_hat_i = x[i].detach().cpu().numpy(), x_hat[i].detach().cpu().numpy()

            x_i = convert2img(x_i)
            x_hat_i = convert2img(x_hat_i)            
            summary = np.hstack([x_i,x_hat_i]) 

            summary= summary.astype(np.uint8)
            summary = Image.fromarray(summary)
            index_training = self.strategy.index_training

        l_zs = np.asarray(l_zs)
        l_mus = np.asarray(l_mus)
        l_logvar = np.asarray(l_logvar)
        l_y = np.asarray(l_y)
        lista_indices = np.asarray(lista_indices)
        metrics_epoch = {"loss":train_loss/ len(dataloader)}
        other_data_epoch = {} 
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
            loss = self.mse_loss(batch_x_hat, x).item()/x.size(0)

            batch_x_hat = batch_x_hat.detach().cpu().numpy()  
            y = y.detach().cpu().numpy()  
            y = list(y)

            xs.extend(x)
            for x_hat in batch_x_hat:
                x_hat = convert2img(x_hat)
                lista_x_hats.append(x_hat)
            z = z.detach().cpu().numpy()
             
            l_zs.extend(z)
            losses.append(loss)
            labels.extend(y)
        
        loss = np.asarray(losses).sum()/len(dataloader)
        losses = np.asarray(losses)
        lista_indices = np.asarray(lista_indices)
        labels = np.asarray(labels)

        metrics_epoch = {"loss":loss,"TP":10,"TN":10,"FN":0,"FP":0,"accuracy":0.9, 
                        "f1":0.9, "precision":0.9, "recall":0.9 }
        other_data_epoch = {}
        
        return metrics_epoch,other_data_epoch 

