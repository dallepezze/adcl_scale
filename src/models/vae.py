import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def create_vae_model(strategy,img_shape, parameters):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vae = VAE(img_shape,parameters['latent_dim'])
    vae = vae.to(device)
    return vae, device

'''DownBlock is the class to build the convolutional blocks of the encoder'''
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=False,dropoutvalue=0):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        )
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropoutvalue!=0:
            layers.append(nn.Dropout(dropoutvalue))
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv_block(x)
        return x

'''UpBlock is the class to build the convolutional blocks of the decoder'''
class UpBlock(nn.Module): 
    def __init__(self,in_channels, out_channels,kernel_size, stride, padding, batchnorm=False,dropoutvalue=0):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if batchnorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        if dropoutvalue!=0:
            layers.append(nn.Dropout(dropoutvalue))
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self,x):
       x = self.conv_block(x)
       return x 

class Encoder(nn.Module):
    def __init__(self, latent_dim,n_channels=3):
        super().__init__()
        self.net = nn.Sequential(
        DownBlock(n_channels,16,4,2,1,batchnorm=False,dropoutvalue=False)   
        ,DownBlock(16,32,4,2,1,batchnorm=True,dropoutvalue=False)         
        ,DownBlock(32,64,4,2,1,batchnorm=True,dropoutvalue=False)         
        ,DownBlock(64,128,4,2,1,batchnorm=True,dropoutvalue=False)         
        ,DownBlock(128,128,4,2,1,batchnorm=True,dropoutvalue=False)       
        ,DownBlock(128,128,4,2,1,batchnorm=False,dropoutvalue=False)       
        ,nn.Flatten()
        )        
        
        self.linear1 = nn.Linear(128*4*4,latent_dim)                         
        self.linear2 = nn.Linear(128*4*4,latent_dim)



    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std) 
        return eps * std + mu

    def forward(self,x):
        x = self.net(x)
        mu = self.linear1(x)
        logvar = self.linear2(x)
        z = self.reparameterise(mu,logvar)
        return z,mu,logvar 

class Decoder(nn.Module):
      def __init__(self,img_shape, latent_dim,n_channels=3):
          super().__init__()
          self.img_shape = img_shape
          self.linear = nn.Sequential(nn.Linear(latent_dim,128*4*4),nn.LeakyReLU(0.2,inplace=True))
          self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
          self.net = nn.Sequential(
          UpBlock(128,128,4,2,1,batchnorm=False,dropoutvalue=False),
          UpBlock(128,128,4,2,1,batchnorm=True,dropoutvalue=False),
          UpBlock(128,64,4,2,1,batchnorm=True,dropoutvalue=False),
          UpBlock(64,32,4,2,1,batchnorm=True,dropoutvalue=False),
          UpBlock(32,16,4,2,1,batchnorm=True,dropoutvalue=False),
          nn.ConvTranspose2d(16,n_channels,4,2,padding=1),
          nn.Tanh())

      def forward(self,z):
          z = self.linear(z)
          z = self.unflatten(z)
          img = self.net(z)
          return img.view(img.shape[0], *self.img_shape)

def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


class VAE(nn.Module):
    def __init__(self,img_shape, latent_dim,n_channels=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(img_shape, latent_dim)

    def forward(self,x):
        z,mu,logvar = self.encoder(x)
        x_hat =  self.decoder(z)
        return x_hat,z,mu,logvar    




