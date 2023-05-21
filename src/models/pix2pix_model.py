import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from torch.autograd import Variable
import sys
import os


from pytorch_pix2pix.options.train_options import TrainOptions
from pytorch_pix2pix.models import create_model


def create_pix2pix_model(device_id):
    sys.argv = ['test_new_bb_super_resolution_train.py', "--checkpoints_dir","../pytorch_pix2pix/checkpoints" ,'--dataroot', '../data/mvtec_hd', \
            '--name', 'mvtec_pix2pix_inpaint', '--model', 'pix2pix', '--direction', 'AtoB',\
            '--netD', 'pixel',  '--load_size', '256', '--crop_size', '256',
            "--gpu_ids", f"{device_id}"]
    opt = TrainOptions().parse()  # get test options

    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.


    encoder = Encoder()
    decoder = Decoder(opt)
    pix2pix_vae = Pix2Pix_Model(encoder,decoder)
    return pix2pix_vae,opt,encoder,decoder


class Pix2Pix_Model(nn.Module):
    def __init__(self,encoder,decoder):
        super(Pix2Pix_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,x):
        z,mu,logvar = self.encoder(x)
        x_hat =  self.decoder(z)
        return x_hat,z,mu,logvar


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        mu = x
        z=x
        logvar = x
        return z,mu,logvar


class Decoder(nn.Module):
    def __init__(self,opt):
        super(Decoder, self).__init__()

        model = create_model(opt)      
        model.setup(opt)  

        self.pix2pix_model = model

    def forward(self, x):
        self.pix2pix_model.real_A=x
        self.pix2pix_model.forward()
        return self.pix2pix_model.fake_B

