import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd

def apply_threshold(losses, threshold):
    labels = np.zeros(len(losses))
    labels[losses>=threshold]=1
    y_pred = labels
    return y_pred
    

def produce_y_true(lunghezza_dataset,already_seen):
    if already_seen:
        y_true = np.zeros(lunghezza_dataset)
    else:
        y_true = np.ones(lunghezza_dataset)
    return y_true

def calc_precision(tp, fp):
    if tp > 0:
        return tp / (tp + fp)
    return 0

def calc_recall(tp, fn):
    if tp > 0:
        return tp / (tp + fn)
    return 0

def calc_f1score(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)
          
    
def calculate_all_metrics(y_true, y_pred):
    diz = {}
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = TN, FP, FN, TP
    diz["TP"] = TP
    diz["TN"] = TN
    diz["FP"] = FP    
    diz["FN"] = FN
    diz.update( produce_metrics(tn, fp, fn, tp) )

    return diz

def produce_metrics(tn, fp, fn, tp):
    diz = {}
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    diz["accuracy"] = accuracy
    
    precision = calc_precision(tp, fp)
    diz["precision"] = precision
    
    recall = calc_recall(tp, fn)
    diz["recall"] = recall
    
    f1 = calc_f1score(precision, recall)
    diz["f1"] = f1

    return diz
    
        
