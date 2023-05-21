import torch
import numpy as np  

from src.utilities.utility_pix2pix import *
from src.utilities.utility_pix2pix import  produce_input_scaling_model


def produce_reconstructed_output_given_model_and_batch(strategy, data_loader, complete_dataset, batch,batch_index,index_training,test_task_index):
    device = strategy.device
    args = strategy.parameters
    data = batch[0]
    with torch.no_grad():
        data = data.to(device)
        img_size = data.size(-1)

        if args["architecture"]!='pix2pix':
            model = strategy.trainer.vae
            model.eval()
            outputs,z,_,_ = model(data)
            for i in range(data.shape[0]):
                index_sample_batch = i
                logging_images(strategy.trainer,batch, index_sample_batch, data[i:i+1], data[i:i+1], outputs[i:i+1], data_loader, complete_dataset, strategy.index_training, batch_index )
        elif 'pix2pix' in args["architecture"]:
            if args["trainer"]=="pix2pix_inpaint":
                outputs_batch = []
                model = strategy.trainer.vae.decoder.pix2pix_model
                for index_sample_batch in range(batch[0].shape[0]):
                    real_B,fake_B = produce_reconstruction_inpaint(strategy.trainer,model, complete_dataset, batch, index_sample_batch)
                    logging_images(strategy.trainer,batch, index_sample_batch, real_B, real_B, fake_B, data_loader, complete_dataset, strategy.index_training, batch_index )
                    outputs_batch.append(fake_B[0])
                outputs = torch.stack(outputs_batch)  
            if args["trainer"]=="pix2pix_scaling_model":
                    outputs_batch = []
                    model = strategy.trainer.vae.decoder.pix2pix_model
                    for index_sample_batch in range(batch[0].shape[0]):
                        real_A, real_B = produce_input_scaling_model(strategy.trainer,complete_dataset, batch, index_sample_batch)
                        fake_B = forward_test(model, real_A, real_B, model.device).cpu()
                        logging_images(strategy.trainer,batch, index_sample_batch, real_B, real_B, fake_B, data_loader, complete_dataset, strategy.index_training, batch_index )
                        outputs_batch.append(fake_B[0])
                    outputs = torch.stack(outputs_batch) 

    return outputs



def from_sample_to_batch(sample, img_size):
    batch_old0,batch_old1,batch_old2,batch_old3,batch_old4 = [],[],[],[],[]

    batch_old0.append(sample[0].reshape(1,3,img_size,img_size))
    batch_old1.append(sample[1])
    batch_old2.append(sample[2])
    batch_old3.append(sample[3])
    batch_old4.append(sample[4])

    batch_old0 = torch.cat(batch_old0)
    batch_old1 = torch.from_numpy(np.asarray(batch_old1))
    batch_old2 = np.array(batch_old2) 
    batch_old3 = np.array(batch_old3) 
    batch_old4 = np.array(batch_old4) 

    batch = [ batch_old0,batch_old1,batch_old2,batch_old3,batch_old4 ]
    return batch

def produce_output_given_model_and_sample(strategy,sample):
    x = sample[0]
    x = torch.unsqueeze(x,dim=0)
   
    if 'pix2pix' in strategy.parameters["architecture"] and strategy.parameters["trainer"]=="pix2pix_scaling_model":
        outputs_batch = []
        model = strategy.trainer.vae.decoder.pix2pix_model
        #torch.unsqueeze
        batch = from_sample_to_batch(sample, strategy.img_size)
        for index_sample_batch in range(batch[0].shape[0]):
            real_A, real_B = produce_input_scaling_model(strategy.trainer, strategy.complete_train_dataset, batch, index_sample_batch=0)
            fake_B = forward_test(model, real_A, real_B, model.device).cpu()
            # logging_images(strategy.trainer,batch, index_sample_batch, real_A, real_B, fake_B, [1,2,3,4,5], strategy.complete_train_dataset, strategy.index_training, 0 )
            outputs_batch.append(fake_B[0])
        output = [torch.stack(outputs_batch), None, None, None ]
    else:
        x = x.to(strategy.device)
        output  =  strategy.trainer.vae(x)

    return output 


def produce_output_given_model_from_noise(strategy,sample):
    x = sample[0]
    x = torch.unsqueeze(x,dim=0)
    x = x.to(strategy.device)
    if strategy.architecture=='cae':
        noise = torch.randn((1,strategy.latent_dim,4,4))
        x_hat  =  strategy.trainer.vae.decoder(noise.to(strategy.device))
    elif strategy.architecture=='vae':
        noise = torch.randn((1,strategy.latent_dim))
        x_hat  =  strategy.trainer.vae.decoder(noise.to(strategy.device))     
    return x_hat,noise