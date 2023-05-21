import os
from inpaint.ssim_loss import ssim
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from src.utilities.utility_images import *
from src.datasets import *
from src.utilities.utility_models import produce_reconstructed_output_given_model_and_batch

import sklearn
from sklearn.metrics import precision_recall_curve
import sklearn.metrics
from src.utilities.utility_logging import save_pickle 
from src.inpaint.gms_loss import MSGMS_Score

def test_epoch_anomaly_maps(anomaly_maps,gt_mask_list,gt_list, test_task_index, run, class_name,index_training,path_logs):
    # asarray
    gt_mask = np.asarray(gt_mask_list)
    gt_list = np.asarray(gt_list)
    scores = anomaly_maps
    
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    fpr_img, tpr_img, _ = sklearn.metrics.roc_curve(gt_list, img_scores)
    img_roc_auc = sklearn.metrics.roc_auc_score(gt_list, img_scores)
    # calculate pixel-level ROC AUC score
    fpr, tpr, _ = sklearn.metrics.roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = sklearn.metrics.roc_auc_score(gt_mask.flatten(), scores.flatten())

    # plot PR curve
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    pr_auc_score = sklearn.metrics.auc(recall, precision)
    best_f1 = np.max(f1)
    print(f"best f1: {best_f1}")

    diz = {"per_pixel_rocauc":per_pixel_rocauc, "img_roc_auc":img_roc_auc, "threshold":threshold, "pr_auc_score":pr_auc_score, "f1":best_f1, "threshold":threshold   }

    return diz

def test_anomaly_maps(anomaly_maps,gt_mask_list,gt_list, test_task_index, run, class_name,index_training,path_logs):
    # asarray
    gt_mask = np.asarray(gt_mask_list)
    gt_list = np.asarray(gt_list)
    scores = anomaly_maps
    
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    fpr_img, tpr_img, _ = sklearn.metrics.roc_curve(gt_list, img_scores)
    img_roc_auc = sklearn.metrics.roc_auc_score(gt_list, img_scores)
    # calculate pixel-level ROC AUC score
    fpr, tpr, _ = sklearn.metrics.roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = sklearn.metrics.roc_auc_score(gt_mask.flatten(), scores.flatten())
    # plot ROC curve
    dir_path = os.path.join( path_logs, f"T{index_training}", "pictures_roc")
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, str(test_task_index) + '_roc_curve.png')
    plot_roc_curve(fpr_img,tpr_img,fpr,tpr,img_roc_auc,per_pixel_rocauc,class_name,filepath,index_training,test_task_index)

    # plot PR curve
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    pr_auc_score = sklearn.metrics.auc(recall, precision)
    print(f"best threshold: {threshold}")
    best_f1 = np.max(f1)
    print(f"best f1: {best_f1}")

    print("pictures")
    dir_path = os.path.join( path_logs, f"T{index_training}", "pictures_f1s")
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, str(test_task_index) + f'_f1s.png')
    plot_values_thresholds(f1[:-1], thresholds, "f1s", filepath)

    dir_path = os.path.join( path_logs, f"T{index_training}", "pictures_pr_curve")
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, str(test_task_index) + '_pr_curve.png')
    plot_precision_recall_curve(recall, precision, filepath)   

    diz = {"per_pixel_rocauc":per_pixel_rocauc, "img_roc_auc":img_roc_auc, "threshold":threshold, "pr_auc_score":pr_auc_score, "f1":best_f1, "threshold":threshold   }

    return diz


def reconstruct_epoch(strategy,args, data_loader, complete_dataset,class_name,index_training,test_task_index,run,path_logs):
    metrics_epoch = {"loss":10,"TP":10,"TN":10,"FN":0,"FP":0,"accuracy":0.9, "f1":0.9, "precision":0.9, "recall":0.9 }

    pgbar = tqdm(data_loader)
    descr=f"Reconstruct Epoch"
    pgbar.set_description(descr)

    scores, gms_losses, ssim_losses, mses, original_imgs, gt_list, gt_mask_list, recon_imgs   = [],[],[],[],[],[],[],[]
    msgms_score = MSGMS_Score()
    lista_indices, lista_labels = [], []
    batch_index=0
    for batch in pgbar:
        data,class_ids,indices,anomaly_info= batch[0],batch[1],batch[2],batch[3]
        bs = data.shape[0]
        produce_reconstructed_output_given_model_and_batch(strategy, data_loader, complete_dataset, batch,batch_index, index_training,test_task_index)
        batch_index+=1

    return metrics_epoch
        

def reconstruct_epoch_with_evaluation_ad(strategy,args, data_loader, complete_dataset,class_name,index_training,test_task_index,run,path_logs):
    device = strategy.device
    scores, gms_losses, ssim_losses, mses, original_imgs, gt_list, gt_mask_list, recon_imgs   = [],[],[],[],[],[],[],[]
    mse_img_list = []
    msgms_score = MSGMS_Score()
    lista_indices, lista_labels, lista_anomaly_info, lista_filepaths = [], [], [], []
    batch_index=0
    for batch in tqdm(data_loader):
        data,class_ids,indices,anomaly_info= batch[0],batch[1],batch[2],batch[3]
        bs = data.shape[0]
        masks = []
        lista_indices.extend(batch[2].detach().cpu().numpy()) 
        lista_labels.extend(class_ids)
        lista_anomaly_info.extend(anomaly_info)
        lista_filepaths.extend(batch[4])
        for i,idx in enumerate(indices):
              mask_path = complete_dataset.mask[idx]
              mask = complete_dataset.get_mask(mask_path,anomaly_info[i])
              masks.append(mask)
        mask = torch.stack(masks)
        original_imgs.extend(data.cpu().numpy())
        gt_list.extend(anomaly_info.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())

        outputs = produce_reconstructed_output_given_model_and_batch(strategy, data_loader, complete_dataset, batch,batch_index, index_training,test_task_index)
        data = data.to(device)
        outputs = outputs.to(device)  

        score = msgms_score(data, outputs)

        for i in range(outputs.shape[0]):
            mse = torch.pow(outputs[i]-data[i],2).mean()
            mse_img = torch.pow(outputs[i]-data[i],2).sum(axis=0).cpu().numpy()
            #print(f"mse_img: {mse_img.shape}")
            mse_img_list.append(mse_img)
            mses.append(mse.cpu().numpy())

        gms_loss = msgms_score(data, outputs)
        ssim_loss = ssim(data, outputs)

        score = score.squeeze(axis=1).cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=7)
        scores.extend(score)
        recon_imgs.extend(outputs.cpu().numpy())
        gms_losses.extend([1,1])
        ssim_losses.append([1,1])
        batch_index+=1

    scores = standardize_scores(scores)

    diz_metriche = test_results(args,scores,mses,gms_losses,ssim_losses, gt_list,gt_mask_list, recon_imgs, test_task_index, run, class_name,index_training,path_logs,original_imgs)
    diz_metriche["loss"] = diz_metriche["mse"]
    threshold = diz_metriche["threshold"]

    if strategy.produce_visual_debug:
        print("produce_visual_debug")
        mode = strategy.mode
        from src.utilities.utility_pix2pix import create_summary, forward_k_value, produce_visual_debug
        produce_visual_debug(args,mode, lista_indices, lista_labels, scores, mses, gt_list,gt_mask_list, recon_imgs, test_task_index, run, class_name,index_training,path_logs,original_imgs, threshold)

    if strategy.save_embedding:
        print("produce_embedding")
        mode = strategy.mode
        dataset_path_test = strategy.parameters.get("dataset_path_test","")
        type_folder_test = strategy.parameters.get("type_folder_test","")
        memory_reconstruct_dataset_path_test = strategy.parameters.get("memory_reconstruct_dataset_path_test","")
        type_memory_reconstruct_test = strategy.parameters.get("type_memory_reconstruct_test","")
        if dataset_path_test!="" and type_folder_test!="":
            #memorized_data/original_dataset 
            path_originale = os.path.join(dataset_path_test, type_folder_test, f"T9", f"{test_task_index}" )
            if memory_reconstruct_dataset_path_test!="" and type_memory_reconstruct_test!="":
                path_dst = os.path.join(memory_reconstruct_dataset_path_test,  type_memory_reconstruct_test, f"T{index_training}", f"{test_task_index}")
            else:
                path_dst = os.path.join(path_logs,  "evaluate_data", f"T{index_training}", f"{test_task_index}")
            print(f"path_originale: {path_originale}")
            print(f"path_dst: {path_dst}")

            allowed_indices = np.where(strategy.complete_test_dataset.anomaly_info==0)[0]
            allowed_indices = list(allowed_indices)
            from datetime import datetime
            now = datetime.now() # current date and time
            date_time = now.strftime("%d_%m_%Y__%H:%M:%S")
            filename_allowed_indices = f"allowed_indices_T{index_training}_{test_task_index}.pickle"
            filepath_allowed_indices = os.path.join(path_logs,filename_allowed_indices)
            save_pickle(allowed_indices, filepath_allowed_indices)

            output_file = os.path.join(strategy.path_logs, f"output_fid.txt")
            device_id = strategy.device_id
            path_python_env = "python"
            path_script = "/ContinualLearning/pytorch-frechet-inception-distance/fid.py  "
            command = f"{path_python_env} {path_script}  --path1 {path_originale} --path2 {path_dst}  --batch-size 8 --deviceid 0 --fai {filepath_allowed_indices} > {output_file}   2>&1 "
            print(command)
            os.system(command)
            with open(output_file) as f:
                fid_value = f.readlines()[-1]
                fid_value = float(fid_value)
                diz_metriche["FID"] = fid_value
                print(f"FID: {fid_value}")

    return diz_metriche

def load_feature_extractor(backbone_name, device):
    import timm
    from timm.models.cait import Cait
    from timm.models.vision_transformer import VisionTransformer

    if backbone_name in ["cait_m48_448", "deit_base_distilled_patch16_384"]:
        feature_extractor = timm.create_model(backbone_name, pretrained=True)
    elif backbone_name in ["resnet18", "wide_resnet50_2"]:
        feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    feature_extractor = feature_extractor.to(device)

    return feature_extractor


def produce_embedding(strategy,args,mode,lista_indices, lista_labels,lista_anomaly_info,lista_filepaths, recon_imgs,test_task_index, run, class_name,index_training,path_logs,original_imgs):
    feature_extractor = load_feature_extractor("wide_resnet50_2", strategy.device)
    feature_extractor.eval()

    lista_features_original = []
    lista_features_reconstruct = []

    for i in tqdm(list(range(len(lista_indices)))):
        class_id = lista_labels[i]
        sample_id = i
        dir_path = os.path.join( path_logs, f"{mode}", f"T{index_training}",  f"{class_id}" )
        os.makedirs(dir_path,exist_ok=True)

        recon_img, original_img, class_id, index_dataset, anomaly_info, filepath = recon_imgs[i], original_imgs[i], class_id, i, lista_anomaly_info[i], lista_filepaths[i]
        input_tensor = torch.tensor(recon_img)
        input_tensor = torch.stack([input_tensor])
        
        features = get_cnn_features(feature_extractor, input_tensor, strategy.device)
        features = [f.cpu() for f in features]
        filepath = os.path.join(dir_path, f"{index_dataset}_embedding.pickle")
        save_pickle(features,filepath) 
        lista_features_reconstruct.append(torch.flatten(features[2]))        

        input_tensor = torch.tensor(original_img)
        input_tensor = torch.stack([input_tensor])
        
        features = get_cnn_features(feature_extractor, input_tensor, strategy.device)
        features = [f.cpu() for f in features]
        filepath = os.path.join(dir_path, f"{index_dataset}_original_embedding.pickle")
        save_pickle(features,filepath) 
        lista_features_original.append(torch.flatten(features[2]))

    lista_features_original = torch.stack(lista_features_original)
    lista_features_reconstruct = torch.stack(lista_features_reconstruct)

    mse_embedding = torch.sum( torch.pow(lista_features_original-lista_features_reconstruct,2) )

    cov_features_original = torch.cov(lista_features_original)
    cov_features_reconstruct = torch.cov(lista_features_reconstruct)
    comp = cov_features_original+cov_features_reconstruct-2*torch.sqrt(torch.matmul(cov_features_reconstruct,cov_features_original))
    trace = torch.trace(comp)

    FID = mse_embedding + trace
    return mse_embedding, trace, FID


def get_cnn_features(feature_extractor, input_tensor, device):
    with torch.no_grad():
        features = feature_extractor(input_tensor.to(device))                 
        features = [feature for feature in features ] 
        return features

def standardize_scores(scores):
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
    return scores


def plot_precision_recall_curve(recall, precision, filepath):
    plt.plot(recall, precision, marker='.', label='')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(filepath, dpi=200)
    plt.close("all")

def plot_values_thresholds(values, thresholds, value_name, filepath):
    plt.plot(thresholds, values, marker='.', label='')
    plt.xlabel('threshold')
    plt.ylabel(f'{value_name}')
    plt.savefig(filepath, dpi=200)
    plt.close("all")

def plot_roc_curve(fpr_img,tpr_img,fpr,tpr,img_roc_auc,per_pixel_rocauc,class_name,filepath,index_training,test_task_index):
    label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc)
    plt.plot(fpr_img, tpr_img, label = label)
    # run[f"Task_Results/T{index_training}/evaluation_ad/img_roc_auc"].log(img_roc_auc)
    
    label='%s pixel_ROCAUC: %.3f' % (class_name, per_pixel_rocauc)
    # run[f"Task_Results/T{index_training}/evaluation_ad/pix_rocauc"].log(per_pixel_rocauc)
    plt.plot(fpr, tpr, label=label)

    plt.legend(loc="lower right")    
    plt.savefig(filepath, dpi=200)
    plt.close("all")


def test_results(args, scores, mses, gms_losses, ssim_losses, gt_list,gt_mask_list, recon_imgs, test_task_index, run, class_name,index_training,path_logs,test_imgs):
    gt_mask = np.asarray(gt_mask_list)
    gt_list = np.asarray(gt_list)
    
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    fpr_img, tpr_img, _ = sklearn.metrics.roc_curve(gt_list, img_scores)
    img_roc_auc = sklearn.metrics.roc_auc_score(gt_list, img_scores)
    # calculate pixel-level ROC AUC score
    fpr, tpr, _ = sklearn.metrics.roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = sklearn.metrics.roc_auc_score(gt_mask.flatten(), scores.flatten())
    # plot ROC curve
    dir_path = os.path.join( path_logs, f"T{index_training}", "pictures_roc")
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, str(test_task_index) + '_roc_curve.png')
    plot_roc_curve(fpr_img,tpr_img,fpr,tpr,img_roc_auc,per_pixel_rocauc,class_name,filepath,index_training,test_task_index)

    # plot PR curve
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    pr_auc_score = sklearn.metrics.auc(recall, precision)
    print(f"best threshold: {threshold}")
    print(f"best f1: {np.max(f1)}")

    print("pictures")
    dir_path = os.path.join( path_logs, f"T{index_training}", "pictures_f1s")
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, str(test_task_index) + f'_f1s.png')
    plot_values_thresholds(f1[:-1], thresholds, "f1s", filepath)

    dir_path = os.path.join( path_logs, f"T{index_training}", "pictures_pr_curve")
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, str(test_task_index) + '_pr_curve.png')
    plot_precision_recall_curve(recall, precision, filepath)   

    #calculate IoU and f1
    print("calculate_additional_metrics")
    best_values, metrics_values, thresholds_2 = calculate_additional_metrics(scores, gt_mask, run, index_training )
    # print(f"best f1 {np.max(f1)}")
    bv = best_values["f1"]
    # print(f"best f1 - 2 {bv}")

    for metric_name in metrics_values:
        dir_path = os.path.join( path_logs, f"T{index_training}", "pictures_thresholds")
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, str(test_task_index) + f'_{metric_name}.png')
        plot_values_thresholds(metrics_values[metric_name], thresholds_2, metric_name, filepath)

    mse = np.asarray(mses).mean()   
    gms_loss = np.asarray(gms_losses ).mean()
    ssim_loss = np.asarray(ssim_losses ).mean()
    comb_loss = mse+gms_loss+ssim_loss

    diz = {"mse":mse, "gms_loss":gms_loss, "ssim_loss":ssim_loss, "comb_loss":comb_loss, "per_pixel_rocauc":per_pixel_rocauc, "img_roc_auc":img_roc_auc, "threshold":threshold, "pr_auc_score":pr_auc_score   }
    for key in best_values:
        diz[key] = best_values[key]

    return diz



import src.utilities.utility_metrics as utility_metrics
def calculate_additional_metrics(scores, gt_mask, run, index_training ):
    scores_flatten = scores.flatten()
    y_probs = scores_flatten
    y_true = gt_mask.flatten()
    # mean = scores_flatten.mean()
    # std = scores_flatten.std()
    minimum = np.min(scores_flatten)
    maximum = np.max(scores_flatten)
    thresholds = np.linspace(start=minimum, stop=maximum, num=50)
    metrics_values = {}
    metrics_name = ["f1","tpr","fpr","prec","rec","acc","balanced_acc","IoU" ]
    for th in thresholds:
        diz = utility_metrics.calculate_metrics_basic(y_probs,y_true,th)
        TP, FP, TN, FN = diz["TP"],diz["FP"],diz["TN"],diz["FN"]
        diz_metrics_aggregated = utility_metrics.calculate_metrics_aggregated(TP,FP,TN,FN)
        for key in metrics_name :
            metrics_values.setdefault(key, [])
            metrics_values[key].append(diz_metrics_aggregated[key])


    best_values = {}
    for metric_name in ["IoU","f1"]:
        best_value = np.max(metrics_values[metric_name])
        best_values[metric_name] = best_value

    return best_values,metrics_values,thresholds