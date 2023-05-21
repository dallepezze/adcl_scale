import pickle
from pathlib import Path
from neptune.new.types import File
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  
import shutil
import os
import sys
import torch
from sklearn.decomposition import PCA
from filelock import FileLock
from src.utilities.utility_images import convert2img


def make_dirs(dir_path):
    lock = FileLock("high_ground.txt.lock")
    with lock:
        isExist = os.path.exists(dir_path)
        if isExist is False:
            os.makedirs(dir_path,exist_ok=True)


# save all info of sample in a dict(exclude the image itself)
def from_sample_to_dict(sample):
    diz = {}
    diz["y"] = sample[1]
    class_id = diz["y"]
    diz["idx"] = sample[2]
    diz["anomaly_info"] = sample[3]
    diz["filepath"] = sample[4]
    return diz

def load_pickle(filepath):  
    f = open(filepath, "rb")
    obj = pickle.load(f)
    f.close()
    return obj


def save_pickle(obj, filepath):  
    f = open(filepath, "wb")
    pickle.dump(obj,f)
    f.close()

def save_img(img, filepath):
    if isinstance(img, torch.Tensor):
        img = convert2img(img.cpu())
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filepath) 

def save_model(strategy, architecture, path_logs, index_training):
        model = strategy.trainer.vae
        save_dir = os.path.join(path_logs, "models")
        print(f"Save model: {save_dir}")
        os.makedirs(save_dir,exist_ok=True)
        if "pix2pix" not in architecture:
            filepath = os.path.join(save_dir, f"T{index_training}_model.pt")
            torch.save(model.state_dict(), filepath)
        elif "pix2pix" in architecture:
            save_pix2pix_model(model,save_dir,index_training)
                

def save_pix2pix_model(model,save_dir,index_training,suffix=""):
    filepath = os.path.join(save_dir, f"T{index_training}_model{suffix}.pt")
    torch.save(model.state_dict(), filepath)

    filepath = os.path.join(save_dir, f"T{index_training}_netG{suffix}.pt") 
    torch.save(model.decoder.pix2pix_model.netG.state_dict(), filepath)

    filepath = os.path.join(save_dir, f"T{index_training}_netD{suffix}.pt") 
    torch.save(model.decoder.pix2pix_model.netD.state_dict(), filepath)
            

def load_model(strategy, architecture, path_logs, index_training):
        load_dir = os.path.join(path_logs, "models")
        model = strategy.trainer.vae
        print(f"load model {load_dir} T{index_training}_model.pt")

        if "pix2pix" not in architecture:
            filepath = os.path.join(load_dir, f"T{index_training}_model.pt")
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict)
        elif "pix2pix" in architecture:
            load_pix2pix_model(model,load_dir,index_training,suffix="")

            
def load_pix2pix_model(model,load_dir,index_training,suffix=""):
    filepath = os.path.join(load_dir, f"T{index_training}_model{suffix}.pt")
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)

    # Load netG
    filepath = os.path.join(load_dir, f"T{index_training}_netG{suffix}.pt") 
    state_dict = torch.load(filepath)
    model.decoder.pix2pix_model.netG.load_state_dict(state_dict)

    # Load netD
    filepath = os.path.join(load_dir, f"T{index_training}_netD{suffix}.pt") 
    state_dict = torch.load(filepath)
    model.decoder.pix2pix_model.netD.load_state_dict(state_dict)

def remove_folder(dir_path):
    if not os.path.exists(dir_path):
        print(f"{dir_path} doesn't exist !")
    else:
        shutil.rmtree(dir_path)

def reset_training_task(strategy, run, path_logs, losses, index_training):
    task_id_str = f"T{index_training}"
    run["Summary/reset_training_task"].log(task_id_str)

    fig = plt.figure(figsize=(10,5))
    plt.plot( list(range(len(losses))), losses)
    run["Summary/reset_training_task_losses_img"].log(fig)

    losses_roll = np.roll(losses, 5)
    fig = plt.figure(figsize=(10,5))
    plt.plot( list(range(len(losses_roll))), losses_roll)
    run["Summary/reset_training_task_losses_roll_img"].log(fig)

    for diz in [strategy.metrics_train,strategy.other_data_train,strategy.metrics_test,strategy.other_data_test]:
        if index_training in diz:
            diz.pop(index_training)

    dir_path = os.path.join(path_logs, task_id_str)
    remove_folder(dir_path)

    if index_training==0:
        current_task = index_training
    else:
        current_task = index_training-1
    print("Reload model")
    load_model(strategy.trainer.vae, strategy.parameters["architecture"], path_logs, current_task)


def create_paths(paths):
    for path in paths:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)


def find_index_task_given_class_id(class_id_search,task_order):
    for index_task,classes_ids in enumerate(task_order):
        if class_id_search in np.asarray(classes_ids):
            return index_task
    return -1


def log_image(key,specific_name=True,filepath=None,run=None):
    filename = os.path.basename(filepath)
    if specific_name:
        key_img=key+"/"+filename
    else:
        key_img = key
    img = Image.open(filepath)
    run[key_img].log(File.as_image(img))
    img.close()  

#specific_name: if True then image is saved with key+filename otherwise only with key
def log_images(key,specific_name=True,filepaths=None,run=None):
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        if specific_name:
            key_img=key+"/"+filename
        else:
            key_img = key
        img = Image.open(filepath)
        run[key_img].log(File.as_image(img))
        img.close()

def return_values_from_parameters(parameters):
    # anomaly_task_id = parameters["anomaly_task_id"]
    num_tasks = parameters["num_tasks"]
    latent_dim = parameters["latent_dim"]
    num_epochs = parameters["num_epochs"]
    batch_size = parameters["batch_size"]
    lr = parameters["lr"]
    beta = parameters["beta"]

    criterion_type = parameters["criterion_type"]
    activation = parameters["activation"]
    device = parameters["device"]
    dataset_name = parameters["dataset_name"]
    transformation = parameters["transformation"]
    task_order = parameters["task_order"]
    sample_strategy = parameters["sample_strategy"]

    return num_tasks,latent_dim,num_epochs,batch_size,lr,task_order,sample_strategy,beta,dataset_name


from argparse import Namespace
def from_parameters_to_opt(parameters):
    crp_size = parameters["crp_size"] if "crp_size" in parameters else parameters["img_size"]
    opt = Namespace(batch_size=parameters["batch_size"], channels=parameters["channels"],use_all_classes=True,only_normal=False,only_anomalies=False, class_name='toothbrush', clip_value=0.01, 
    crp_size=crp_size, data_path=parameters["data_path"], img_size=parameters["img_size"],brightness=0.1, latent_dim=parameters["latent_dim"], lr=parameters["lr"], model_name='waae', n_critic=5, n_epochs=parameters["num_epochs"],gray=parameters["gray"],apply_rotation=parameters["apply_rotation"])
    opt.img_shape = (opt.channels, opt.img_size, opt.img_size)
    return opt


