from base64 import b16decode
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
import random # this module will be used to select random samples from a collection
import os # this module will be used just to create directories in the local filesystem
from tqdm import tqdm # this module is useful to plot progress bars
from torchvision.transforms import InterpolationMode

import torch
from torch import Tensor
from typing import Callable, List, Tuple, Union, Iterable

from src.loss_functions import *
from src.utilities.utility_images import *

from src.utilities.utility_ad import standardize_scores, test_anomaly_maps, test_epoch_anomaly_maps

from src.utilities.utility_pix2pix import create_summary,create_summary_by_numpy, produce_visual_debug

import wandb


class FastflowLoss(nn.Module):
    """FastFlow Loss."""

    def forward(self, hidden_variables: List[Tensor], jacobians: List[Tensor]) -> Tensor:
        """Calculate the Fastflow loss.

        Args:
            hidden_variables (List[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (List[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: Fastflow loss computed based on the hidden variables and the log of the Jacobians.
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
        for (hidden_variable, jacobian) in zip(hidden_variables, jacobians):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss

class Trainer_fastflow():
    def __init__(self,strategy, fastflow):
        self.strategy = strategy
        self.vae = fastflow
        self.lr = self.strategy.lr
        self.b1 = self.strategy.b1
        self.b2 = self.strategy.b2
        self.weight_decay = self.strategy.weight_decay
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay= self.weight_decay)
        self.loss = FastflowLoss()
    
    def train_epoch(self,dataloader):
        self.vae.training = True
        l_fastflow_loss = 0.0
        dataSize = len(dataloader.dataset)
        lista_indices = []

        batch_index = 0
        for batch in tqdm(dataloader):
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0])
            x = batch[0]
            batch_size = x.size(0)
            indices = batch[2]
            lista_indices.extend(indices.detach().cpu().numpy())
            x = x.to(self.device)
            if self.strategy.img_size==384:
                x = F.interpolate(x,size=(384,384),mode="bicubic")
            hidden_variables, jacobians = self.vae(x)
            loss = self.loss(hidden_variables, jacobians)

            self.optimizer.zero_grad()
            loss.backward()
            l_fastflow_loss += loss.item() * batch_size
            self.optimizer.step()

            # only for debugging
            if  batch_index==(len(dataloader)-1):
                for i in range(1):
                    original_img = convert2img(x[i])
                    summary = create_summary_by_numpy([original_img])

            batch_index += 1

        l_fastflow_loss /= dataSize
        lista_indices = np.asarray(lista_indices)

        metrics_epoch = {"loss":l_fastflow_loss}
        other_data_epoch = {"indices":lista_indices}
        
        return metrics_epoch,other_data_epoch   

    def test_epoch(self,dataloader):
        dataset = self.strategy.complete_test_dataset
        self.vae.training = False
        lista_indices = []
        losses, l_anomaly_maps, lista_labels = [], [], []
        test_imgs, gt_list, gt_mask_list = [], [], []
        batch_index = 0
        max_score_value = 0
        min_score_value = -10000000
        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            data = data.to(self.device)
            with torch.no_grad():
                if self.strategy.img_size==384:
                    data = F.interpolate(data,size=(384,384),mode="bicubic")
                anomaly_maps = self.vae(data)
                if self.strategy.img_size==384:
                    anomaly_maps = F.interpolate(anomaly_maps,size=(256,256),mode="bicubic")
                max_score_value = max( anomaly_maps.max(), max_score_value)
                min_score_value = min( anomaly_maps.min(), min_score_value)
                
            anomaly_maps = anomaly_maps.squeeze(axis=1).cpu().numpy()  
            l_anomaly_maps.extend(anomaly_maps)

            lista_labels.extend(class_ids)
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path,anomaly_info[i])
                masks.append(mask)
            mask = torch.stack(masks)
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(anomaly_info.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy())

            batch_index +=1
        l_anomaly_maps = standardize_scores(l_anomaly_maps)
        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps,gt_mask_list, gt_list, self.strategy.index_training, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]
        
        
        mode = self.strategy.trainer.mode if hasattr(self.strategy.trainer, 'mode') else "reconstruct"
        
        metrics_epoch = diz_metriche
        other_data_epoch = {}
        return metrics_epoch, other_data_epoch

    def evaluate_data(self, dataloader,test_loss_function=None):  
        dataset = self.strategy.complete_test_dataset
        test_task_index = self.strategy.current_test_task_index
        index_training = self.strategy.index_training
        self.vae.training = False
        lista_indices = []
        losses, l_anomaly_maps, lista_labels = [], [], []
        test_imgs, gt_list, gt_mask_list = [], [], []
        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            data = data.to(self.device)
            with torch.no_grad():
                if self.strategy.img_size==384:
                    data = F.interpolate(data,size=(384,384),mode="bicubic")
                anomaly_maps = self.vae(data)
                if self.strategy.img_size==384:
                    anomaly_maps = F.interpolate(anomaly_maps,size=(256,256),mode="bicubic")

            anomaly_maps = anomaly_maps.squeeze(axis=1).cpu().numpy()  
            l_anomaly_maps.extend(anomaly_maps)

            lista_labels.extend(class_ids)
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path,anomaly_info[i])
                masks.append(mask)
            mask = torch.stack(masks)
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(anomaly_info.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy())
        l_anomaly_maps = standardize_scores(l_anomaly_maps)
        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps,gt_mask_list, gt_list, self.strategy.index_training, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]
        
        if self.strategy.produce_visual_debug:
            mode = self.strategy.mode 
            produce_visual_debug(self.strategy.parameters,mode, lista_indices, lista_labels, l_anomaly_maps, losses, gt_list,gt_mask_list, test_imgs, test_task_index, self.strategy.run, self.strategy.labels_map[test_task_index], index_training,self.strategy.path_logs,test_imgs, threshold)        
            
        metrics_epoch = diz_metriche
        other_data_epoch = {}
        return metrics_epoch, other_data_epoch