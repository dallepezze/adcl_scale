from torchvision import transforms as transforms
import torch
import pickle
import numpy as np
from PIL import Image
import os
import sys
import time

ROOT = ".."
sys.path.append(ROOT)

from src.datasets import *
from src.utilities.utility_images import *
from src.utilities.utility_logging import *



def calculate_all_metrics_given_indices(lista_metriche_per_sample,thresholds,indices,keys):
    lista_metrics_for_each_threshold_given_samples = calculate_metrics_for_each_threshold_given_samples(lista_metriche_per_sample,thresholds,indices,keys)

    num_samples = len(indices)
    best_index = np.argmax([ lista_metrics_for_each_threshold_given_samples[index_th]["f1"]  for index_th in range(len(thresholds)) ])
    metrics = {"best_index":best_index}
    for key in keys:
        lista_values = [ lista_metrics_for_each_threshold_given_samples[index_th][key]  for index_th in range(len(thresholds)) ]
        best_value = lista_values[best_index] 
        metrics[key+"s"] = lista_values         
        metrics["best_"+key] = best_value       
        metrics[key] = best_value               

    roc_auc_score = calculate_roc_auc_score(metrics["fprs"],metrics["tprs"]) 
    metrics["roc_auc_score"] = roc_auc_score

    pr_auc_score = calculate_roc_auc_score(metrics["recs"], metrics["precs"])
    average_precision = pr_auc_score
    metrics["average_precision"] = average_precision

    return metrics

    # a dictionary for each threshold
    # each dictionary contains:
    #      - TP,TN,FP,FN
    #      - th,f1,rec,prec,fpr,tpr,acc,balanced_acc 
    #      - avg_f1,avg_rec,...
def calculate_metrics_for_each_threshold_given_samples(lista_metriche_per_sample,thresholds,indices,keys):
    metrics_for_each_threshold_given_samples = []
    num_samples = len(indices)
    for index_th,th in enumerate(thresholds):
        metrics_th = {}
        for key in keys: 
            metrics_th[key] = 0
            for index_sample in indices:
                metrics_th[key] += lista_metriche_per_sample[index_sample][index_th][key]

        for key in keys:
            metrics_th["avg_"+key] = metrics_th[key]/num_samples

        metrics_th["th"] = th
        TP,FP,TN,FN = metrics_th["TP"], metrics_th["FP"], metrics_th["TN"], metrics_th["FN"]
        diz_metrics_aggregated = calculate_metrics_aggregated(TP,FP,TN,FN)
        metrics_th.update(diz_metrics_aggregated)

        metrics_for_each_threshold_given_samples.append(metrics_th)

    return metrics_for_each_threshold_given_samples
        
def calculate_metrics_aggregated(TP,FP,TN,FN):
    f1 = calculate_f1(TP,FP,TN,FN)
    fpr = calculate_fpr(TP,FP,TN,FN)
    tpr = calculate_tpr(TP,FP,TN,FN)
    prec = calculate_precision(TP,FP,TN,FN)
    rec = calculate_recall(TP,FP,TN,FN)
    acc = calculate_acc(TP,FP,TN,FN)
    balanced_acc = calculate_balanced_acc(TP,FP,TN,FN)
    IoU = calculate_IoU(TP,FP,TN,FN)
    
    return {"f1":f1,"tpr":tpr,"fpr":fpr,"prec":prec,"rec":rec,"acc":acc,"balanced_acc":balanced_acc,"IoU":IoU }

def calculate_metrics_basic(y_probs,y_true,th):
    TP = ((y_true)*(y_probs>=th)).sum()  
    FP = ((1-y_true)*(y_probs>=th)).sum()
    TN = ((1-y_true)*(1-(y_probs>=th))).sum()
    FN = ((y_true)*(1-(y_probs>=th))).sum()
    
    return {"TP":TP,"FP":FP,"TN":TN,"FN":FN,"th":th }

def calculate_metrics_from_sample(y_probs,y_true,th):
    diz = calculate_metrics_basic(y_probs,y_true,th)
    diz2 = calculate_metrics_aggregated(diz["TP"], diz["FP"], diz["TN"] , diz["FN"])
    diz.update(diz2)
    return diz

def calculate_roc_auc_score(fprs,tprs):
    roc_auc_score = 0
    for k in range(len(fprs)-1):
        roc_auc_score += (fprs[k+1]-fprs[k])*tprs[k]
    return roc_auc_score

def calculate_f1(TP,FP,TN,FN):
    denom = 2*TP + FP + FN
    if denom==0:
        f1 = 0
    else:
        f1= (2*TP)/denom
    return f1

def calculate_tpr(TP,FP,TN,FN):
    denom = FN+TP
    if denom==0:
        tpr = 0
    else:
        tpr = (TP)/denom
    return tpr

def calculate_fpr(TP,FP,TN,FN):
    denom = FP+TN
    if denom==0:
        fpr = 0
    else:
        fpr = (FP)/denom
    return fpr

def calculate_precision(TP,FP,TN,FN):
    denom = TP+FP
    if denom==0:
        prec = 0
    else:
        prec = (TP)/denom
    return prec

def calculate_recall(TP,FP,TN,FN):
    return calculate_tpr(TP,FP,TN,FN)

def calculate_acc(TP,FP,TN,FN):
    denom = TP+FP+TN+FN
    if denom==0:
        acc = 0 #lanciare un warning
    else:
        acc = (TP+TN)/denom
    return acc

def calculate_balanced_acc(TP,FP,TN,FN):
    tpr = calculate_tpr(TP,FP,TN,FN)
    fpr = calculate_fpr(TP,FP,TN,FN)
    return (tpr+fpr)/2

def calculate_IoU(TP,FP,TN,FN):
    denom = TP+FP+FN
    if denom==0:
        IoU = 0
    else:
        IoU = TP/denom
    return IoU


def load_pickle(dir_path,filename):
    filepath = os.path.join(dir_path, filename)
    f = open(filepath, "rb")
    obj = pickle.load(f)
    f.close()
    return obj

def save_pickle(dir_path,filename,obj):
    filepath = os.path.join(dir_path, filename)
    f = open(filepath, "wb")
    obj = pickle.dump(obj,f)
    f.close()
    return obj

