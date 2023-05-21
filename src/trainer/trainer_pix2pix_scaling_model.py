import numpy as np

import torch

from tqdm import tqdm

from src.utilities.utility_ad import *
from src.utilities.utility_images import *
from src.datasets import *
from src.utilities.utility_ad import *
from src.utilities.utility_pix2pix import create_summary, forward_test, forward_train, produce_scaled_A_from_B


class Trainer_Pix2Pix_ScalingModel():
    def __init__(self,strategy,vae,opt):
        super(Trainer_Pix2Pix_ScalingModel, self).__init__()
        self.strategy = strategy
        self.opt = opt
        self.vae = vae
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.mse_loss = torch.nn.MSELoss()

    # train_epoch_batch_size_one
    def train_epoch(self, dataloader):
        lista_zs,lista_indices,lista_x_hats,lista_mus,losses,lista_labels = [],[],[],[],[],[]
        metrics_epoch = {"loss":[], "loss_G":[], "loss_G_GAN":[], "loss_G_L1":[], "loss_D":[], "loss_D_fake":[], "loss_D_real":[] }

        # self.vae.encoder.eval() 
        # self.vae.decoder.eval()
        index_training = self.strategy.index_training
        epoch_index = self.strategy.current_epoch
        model = self.vae.decoder.pix2pix_model
        mse_loss_func = self.mse_loss
    

        pgbar = tqdm(dataloader)
        descr=f"TrainEpoch {self.strategy.current_epoch}"
        pgbar.set_description(descr)
        batch_index = 0
        for batch in pgbar:
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0])
            #x, y, idx, anomaly_info, filepath
            class_ids, indices = batch[1], batch[2]
            lista_labels.extend(class_ids)
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            # Iterate per sample
            for index_sample_batch,index_dataset in enumerate(indices):#itero una per volta
                class_id = class_ids[index_sample_batch]
                class_name = labels_datasets["mvtec"][class_id]

                #Prepare real_A
                h,w = 32,32
                real_B = batch[0][index_sample_batch]
                real_A,real_A_32 = produce_scaled_A_from_B(real_B.clone(),h,w)

                real_B = batch[0][index_sample_batch:index_sample_batch+1]
                real_A = torch.stack([real_A])
                real_A_32 = np.stack([real_A_32])

                fake_B = forward_train(model, real_A, real_B, model.device).cpu()

                summary = create_summary([real_A[0],real_B[0],fake_B[0]])

                # Log images
                if batch_index==(len(dataloader)-1):                         
                    self.strategy.run[f"Task_Results/T{index_training}"][f"visual_train"].log(summary) 

                # Log metrics
                mse_loss = mse_loss_func(model.fake_B[0].cpu(), model.real_B[0].cpu()).item()
                loss_G_GAN,loss_G_L1,loss_G,loss_D_fake,loss_D_real,loss_D = model.loss_G_GAN.item(), model.loss_G_L1.item(), model.loss_G.item(), model.loss_D_fake.item(), model.loss_D_real.item(), model.loss_D.item()
                for metric_name, metric_value in zip(["loss","loss_G_GAN","loss_G_L1","loss_G","loss_D_fake","loss_D_real","loss_D"], [mse_loss,loss_G_GAN,loss_G_L1,loss_G,loss_D_fake,loss_D_real,loss_D]):
                    metrics_epoch[metric_name].append(metric_value)

            batch_index += 1

        lista_indices = np.asarray(lista_indices)
        lista_labels = np.asarray(lista_labels)
        for key in metrics_epoch:
            metrics_epoch[key] = np.asarray(metrics_epoch[key]).mean()
        # metrics_epoch["loss_G"] = metrics_epoch["loss_G_GAN"]+metrics_epoch["loss_G_L1"]
        # metrics_epoch["loss_D"] = metrics_epoch["loss_D_fake"]+metrics_epoch["loss_D_real"]

        other_data_epoch = {"indices":lista_indices  }         # x_hats,labels, zs, mus, logvars, losses
        return metrics_epoch,other_data_epoch


    # train_epoch_batch_size_one
    def test_epoch(self, dataloader):
        lista_zs,lista_indices,lista_x_hats,lista_mus,losses,lista_labels = [],[],[],[],[],[]
        metrics_epoch = {"loss":[],"TP":10,"TN":10,"FN":0,"FP":0,"accuracy":0.9, "f1":0.9, "precision":0.9, "recall":0.9 }

        # self.vae.encoder.eval() 
        # self.vae.decoder.eval()
        index_training = self.strategy.index_training
        epoch_index = self.strategy.current_epoch
        model = self.vae.decoder.pix2pix_model
        mse_loss_func = self.mse_loss

        pgbar = tqdm(dataloader)
        descr=f"TestEpoch {self.strategy.current_epoch}"
        pgbar.set_description(descr)
        batch_index = 0
        for batch in pgbar:
            #x, y, idx, anomaly_info, filepath
            class_ids, indices = batch[1], batch[2]
            lista_labels.extend(class_ids)
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            # Iterate per sample
            for index_sample_batch,index_dataset in enumerate(indices):#itero una per volta
                class_id = class_ids[index_sample_batch]
                class_name = labels_datasets["mvtec"][class_id]

                #Prepare real_A
                h,w = 32,32
                real_B = batch[0][index_sample_batch]
                real_A,real_A_32 = produce_scaled_A_from_B(real_B.clone(),h,w)

                real_B = batch[0][index_sample_batch:index_sample_batch+1]
                real_A = torch.stack([real_A])
                real_A_32 = np.stack([real_A_32])

                fake_B = forward_test(model, real_A, real_B, model.device).cpu()

                summary = create_summary([real_A[0],real_B[0],fake_B[0]])

                # Log images
                if batch_index==(len(dataloader)-1) and index_sample_batch<=3: 
                    self.strategy.run[f"Task_Results/T{index_training}"][f"visual_test"].log(summary) 

                # Log metrics
                mse_loss = mse_loss_func(model.fake_B[0].cpu(), model.real_B[0].cpu()).item()
                metrics_epoch["loss"].append(mse_loss)

            batch_index += 1

        lista_indices = np.asarray(lista_indices)
        lista_labels = np.asarray(lista_labels)
        metrics_epoch["loss"] = np.asarray(metrics_epoch["loss"]).mean()

        other_data_epoch = {"indices":lista_indices  }         
        return metrics_epoch,other_data_epoch


    def forward(self,imgs_batch):
        encoded_imgs,_,_ = self.encoder(imgs_batch)
        decoded_imgs = self.decoder(encoded_imgs)
        rec_loss = self.mse_loss(decoded_imgs, imgs_batch)
        return encoded_imgs,decoded_imgs,rec_loss


    def evaluate_data(self,dataloader,test_loss_function=None):
        anomaly_detection_task = self.strategy.parameters.get("anomaly_detection_task", False)
        if anomaly_detection_task:
            test_task_index = self.strategy.current_test_task_index
            class_name = self.strategy.labels_map[test_task_index]
            self.mode = "reconstruct_test"
            diz = reconstruct_epoch_with_evaluation_ad(self.strategy, self.strategy.parameters, dataloader, self.strategy.complete_test_dataset,class_name,self.strategy.index_training,test_task_index,self.strategy.run,self.strategy.path_logs)
            metrics_epoch = diz
            other_data_epoch = {}
            return metrics_epoch,other_data_epoch
        else:
            test_task_index = self.strategy.current_test_task_index
            class_name = self.strategy.labels_map[test_task_index]
            self.mode = "reconstruct_test"
            diz = reconstruct_epoch(self.strategy, self.strategy.parameters, dataloader, self.strategy.complete_test_dataset,class_name,self.strategy.index_training,test_task_index,self.strategy.run,self.strategy.path_logs)
            metrics_epoch = diz
            other_data_epoch = {}
            return metrics_epoch,other_data_epoch        


