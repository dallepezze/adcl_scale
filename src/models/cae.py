#from turtle import forward
import torch.nn as nn
import torch

def create_cae_model(strategy, img_shape, parameters):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cae = CAE(img_shape,parameters['latent_dim'])
    cae = cae.to(device)
    return cae, device  


########## architecture 1 with 6 conv layers with size 256 x 256 #################

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True,dropoutvalue=0):
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

class UpBlock(nn.Module): 
    def __init__(self,in_channels, out_channels,kernel_size, stride, padding, batchnorm=True,dropoutvalue=0):
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
    def __init__(self, latent_dim,n_features = 16):
        super().__init__()
        self.down1 = DownBlock(3,n_features,4,2,1,batchnorm=False,dropoutvalue=False) # 128x128
        self.down2 = DownBlock(n_features,n_features*2,4,2,1,batchnorm=True,dropoutvalue=False) # 64x64
        self.down3 = DownBlock(n_features*2,n_features*4,4,2,1,batchnorm=True,dropoutvalue=False) #32x32
        self.down4 = DownBlock(n_features*4,n_features*8,4,2,1,batchnorm=True,dropoutvalue=False) #16x16
        self.down5 = DownBlock(n_features*8,n_features*16,4,2,1,batchnorm=True,dropoutvalue=False) #8x8
        self.down6 = DownBlock(n_features*16,latent_dim,4,2,1,batchnorm=False,dropoutvalue=False) #4x4x512 

    def forward(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        z = self.down6(x)
        return z

class Decoder(nn.Module):
    def __init__(self,img_shape, latent_dim,n_features=16):
        super().__init__()
        self.img_shape = img_shape
        self.up1 = UpBlock(latent_dim,n_features*16,4,2,1,batchnorm=False,dropoutvalue=False)
        self.up2 = UpBlock(n_features*16,n_features*8,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up3 = UpBlock(n_features*8,n_features*4,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up4 = UpBlock(n_features*4,n_features*2,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up5 = UpBlock(n_features*2,n_features,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up6 = nn.Sequential(nn.ConvTranspose2d(n_features,3,4,2,padding=1),
                   nn.Tanh())

    def forward(self,z):
        img = self.up1(z)
        img = self.up2(img)
        img = self.up3(img)
        img = self.up4(img)
        img = self.up5(img)
        img = self.up6(img)
        return img.view(img.shape[0], *self.img_shape)


class CAE(nn.Module):
    def __init__(self,img_shape, latent_dim):
        super(CAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(img_shape, latent_dim)

    def forward(self,x):
        z  = self.encoder(x)
        x_hat =  self.decoder(z)
        return x_hat,z,None,None
