import os
import numpy as np

import torch

from tqdm import tqdm

from src.utilities.utility_ad import *
from src.utilities.utility_images import *
from src.datasets import *
from src.utilities.utility_ad import *
from src.utilities.utility_pix2pix import create_images_masked, create_summary, forward_k_value,forward_test,forward_train


class Trainer_Pix2Pix_InPaint():
    def __init__(self,strategy,vae,opt):
        super(Trainer_Pix2Pix_InPaint, self).__init__()
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
            class_ids, indices = batch[1], batch[2]
            lista_labels.extend(class_ids)
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            for index_sample_batch,index_dataset in enumerate(indices):#itero una per volta
                class_id = class_ids[index_sample_batch]
                class_name = labels_datasets["mvtec"][class_id]

                # prepare real_A
                k_value = random.sample( self.strategy.k_value, 1)
                inputs,Ms = create_images_masked(k_value, batch[0][index_sample_batch])

                # Iterate on masked images 
                outputs = []
                for index_inputs in range(len(inputs)):
                    real_A = inputs[index_inputs:index_inputs+1]
                    real_B = batch[0][index_sample_batch:index_sample_batch+1]
                    fake_B = forward_train(model, real_A, real_B, model.device).cpu()
                    outputs.append(fake_B[0])
                    summary = create_summary([real_A[0],real_B[0],fake_B[0]])

                    # Log metrics
                    mse_loss = mse_loss_func(model.fake_B[0].cpu(), model.real_B[0].cpu()).item()
                    loss_G_GAN,loss_G_L1,loss_G,loss_D_fake,loss_D_real,loss_D = model.loss_G_GAN.item(), model.loss_G_L1.item(), model.loss_G.item(), model.loss_D_fake.item(), model.loss_D_real.item(), model.loss_D.item()
                    for metric_name, metric_value in zip(["loss","loss_G_GAN","loss_G_L1","loss_G","loss_D_fake","loss_D_real","loss_D"], [mse_loss,loss_G_GAN,loss_G_L1,loss_G,loss_D_fake,loss_D_real,loss_D]):
                        metrics_epoch[metric_name].append(metric_value)

                output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False)), outputs, Ms))      
                # Log image
                if batch_index==(len(dataloader)-1):
                    summary = create_summary([real_B[0],output])
                    self.strategy.run[f"Task_Results/T{index_training}"]["visual_train"].log(summary) 

            batch_index += 1

        lista_indices = np.asarray(lista_indices)
        lista_labels = np.asarray(lista_labels)
        for key in metrics_epoch:
            metrics_epoch[key] = np.asarray(metrics_epoch[key]).mean()

        other_data_epoch = {"indices":lista_indices  }         
        return metrics_epoch,other_data_epoch


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
                k_value = random.sample( self.strategy.k_value, 1)
                inputs,Ms = create_images_masked(k_value, batch[0][index_sample_batch])

                # Iterate on masked images 
                outputs = []
                for index_inputs in range(len(inputs)):
                    real_A = inputs[index_inputs:index_inputs+1]
                    real_B = batch[0][index_sample_batch:index_sample_batch+1]
                    fake_B = forward_test(model, real_A, real_B, model.device).cpu()
                    outputs.append(fake_B[0])
                    summary = create_summary([real_A[0],real_B[0],fake_B[0]])

                    # Log images
                    # if batch_index==(len(dataloader)-1): #batch_index==(len(dataloader)-1) and index_inputs==(len(inputs)-1):                        
                    #     self.strategy.run[f"Task_Results/T{index_training}"][f"visual_test_sample_k={k_value}"].log(summary) 

                    # Log metrics
                    mse_loss = mse_loss_func(model.fake_B[0].cpu(), model.real_B[0].cpu()).item()
                    metrics_epoch["loss"].append(mse_loss)

                output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False)), outputs, Ms))      
                # Log image
                summary = create_summary([real_B[0],output])
                dir_path = os.path.join( self.strategy.path_logs, f"T{index_training}", "visual_test" )
                os.makedirs(dir_path, exist_ok=True)
                filepath =  os.path.join(dir_path, f"{index_dataset}.png" )
                
                if batch_index==(len(dataloader)-1) and index_sample_batch<=3:
                    summary.save(filepath)
                    self.strategy.run[f"Task_Results/T{index_training}"][f"visual_test"].log(summary) 

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
        anomaly_detection_task_with_metrics = self.strategy.parameters.get("anomaly_detection_task_with_metrics", False)
        model = self.vae.decoder.pix2pix_model
        test_task_index = self.strategy.current_test_task_index
        class_name = self.strategy.labels_map[test_task_index]
        mse_loss_func = self.mse_loss

        if anomaly_detection_task and anomaly_detection_task_with_metrics:
            diz = evaluate_data_pix2pix_inpaint(self.strategy.parameters, self.strategy, model, mse_loss_func, dataloader, self.strategy.complete_test_dataset, class_name, self.strategy.index_training, test_task_index, self.strategy.run, self.strategy.path_logs)
            metrics_epoch = diz
            other_data_epoch = {}
            return metrics_epoch,other_data_epoch
        else:
            return self.test_epoch(dataloader)




def evaluate_data_pix2pix_inpaint(args, strategy, model, mse_loss_func, test_loader,dataset,class_name,index_training,test_task_index,run,path_logs):
    lista_zs,lista_indices,lista_x_hats,lista_mus,losses,lista_labels = [],[],[],[],[],[]
    scores,test_imgs,gt_list,gt_mask_list,recon_imgs = [],[],[],[],[]
    msgms_score = MSGMS_Score()
    mses, gms_losses, ssim_losses = [],[],[]

    index_training = strategy.index_training

    pgbar = tqdm(test_loader)
    descr=f"Evaluate Data Pix2Pix_Inpaint"
    pgbar.set_description(descr)
    batch_index = 0
    for batch in pgbar:
        #x, y, idx, anomaly_info, filepath
        anomaly_info, class_ids, indices = batch[3], batch[1], batch[2]
        lista_labels.extend(class_ids)
        lista_indices.extend(batch[2].detach().cpu().numpy()) 

        masks = []
        for index_sample_batch,index_dataset in enumerate(indices):
            mask_path = strategy.complete_test_dataset.mask[index_dataset]
            mask = strategy.complete_test_dataset.get_mask(mask_path,anomaly_info[index_sample_batch])
            masks.append(mask)
        mask = torch.stack(masks)
        gt_list.extend(anomaly_info.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())

        # Iterate per sample
        for index_sample_batch,index_dataset in enumerate(indices):#itero una per volta
            class_id = class_ids[index_sample_batch]
            class_name = labels_datasets["mvtec"][class_id]

            output_by_k = []
            total_score = 0
            k_values = strategy.parameters["k_value"]
            for k in k_values:
                image_tensor = batch[0][index_sample_batch]
                output, score, mse_loss = forward_k_value(model, msgms_score, mse_loss_func, image_tensor, [k])

                summary = create_summary([image_tensor,output])
                dir_path = os.path.join( strategy.path_logs, f"T{index_training}", strategy.mode )
                os.makedirs(dir_path, exist_ok=True)
                filepath =  os.path.join(dir_path, f"{index_dataset}_k={k}.png" )
                summary.save(filepath)

                output_by_k.append(output)
                mses.append(mse_loss)
                img_size = batch[0].size(-1)
                N = img_size // k
                total_score+=score/(N**2)


            total_score = total_score.squeeze().numpy()
            for i in range(total_score.shape[0]):
                total_score[i] = gaussian_filter(total_score[i], sigma=7)
            # resulting output (1, 256, 256)
            total_score = np.stack([total_score]) 
            scores.extend(total_score)
            gms_losses.extend([1,1])
            ssim_losses.append([1,1])

        batch_index += 1
        test_imgs.extend(batch[0].cpu().numpy())
        recon_imgs.extend(output.cpu().numpy())

    scores = standardize_scores(scores)

    diz_metriche = test_results(args,scores,mses,gms_losses,ssim_losses, gt_list,gt_mask_list, recon_imgs, test_task_index, run, class_name,index_training,path_logs,test_imgs)
    diz_metriche["loss"] = diz_metriche["mse"]
    return diz_metriche 

        


