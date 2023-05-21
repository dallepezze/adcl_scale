import numpy as np
import os 
import pickle

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset,Subset
from torchvision import transforms
#from torchvision.io import read_video, write_jpeg

from PIL import Image
import PIL

from src.utilities import utility_logging


normalazition_parameters_mvtec = {"mean":(0.485, 0.456, 0.406), "std":(0.229, 0.224, 0.225)}
# normalazition_parameters_mvtec = {"mean":(0.5, 0.5, 0.5), "std":(0.5, 0.5, 0.5)}


'''transformation of the images'''
def create_transform_img(scale_size):
    transform = []
    transform.append(transforms.Resize((scale_size,scale_size), Image.BICUBIC))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(**normalazition_parameters_mvtec))
    return transforms.Compose(transform)    


def filter_dataset(dataset, task_id, batch_size):
    if np.asarray(task_id).ndim==0:
        class_idx = np.where((dataset.targets==task_id))[0]
    else:
        class_idx = np.where( np.isin(dataset.targets,task_id) )[0]
    
    class_subset = Subset(dataset, class_idx)
    class_loader = DataLoader(class_subset, shuffle=True, batch_size=batch_size)

    return class_subset,class_loader


class ContinualLearningBenchmark:
    def __init__(self,complete_train_dataset, complete_test_dataset, num_tasks, task_order):
        self.complete_train_dataset = complete_train_dataset
        self.complete_test_dataset = complete_test_dataset
        self.num_tasks = num_tasks
        self.task_order = task_order

        if num_tasks!=len(task_order):
            print("Attenzione ! Numero di tasks!=task_order length")

    def produce_task_stream(self):
        lista_datasets_train = []
        lista_datasets_test = []
        for task_id in self.task_order:
            dataset_train_task,_ = filter_dataset(self.complete_train_dataset, task_id, 1)
            dataset_test_task,_ = filter_dataset(self.complete_test_dataset, task_id, 1)
            lista_datasets_train.append(dataset_train_task)
            lista_datasets_test.append(dataset_test_task)

        train_stream = lista_datasets_train
        test_stream  = lista_datasets_test
        task_stream = train_stream,test_stream

        return task_stream


normalize_transforms = {  "mvtec": transforms.Normalize(**normalazition_parameters_mvtec)   }
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule',  'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
labels_datasets = { "mvtec":MVTEC_CLASS_NAMES  }
    

def create_transform_x(opt,crp_size):
    transform = []
    transform.append(transforms.Resize((crp_size, crp_size), interpolation=2))
    if opt.gray == True:
      gray_transform = transforms.Grayscale(num_output_channels=3)
      transform.append(gray_transform)    
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(**normalazition_parameters_mvtec))

    return transforms.Compose(transform)

def create_transform_x_with_rotation(opt,crp_size,rotation_degree,fill=0):
    transform = []
    transform.append(transforms.Resize((crp_size, crp_size), interpolation=2))
    random_rotation = torchvision.transforms.RandomAffine(degrees=rotation_degree,resample=PIL.Image.BICUBIC,fill=fill)
    transform.append( random_rotation )
    if opt.gray == True:
      gray_transform = transforms.Grayscale(num_output_channels=3)
      transform.append(gray_transform)
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(**normalazition_parameters_mvtec))
    return transforms.Compose(transform)


def load_dataset(parameters,type_dataset,download=True,normalize=True):
    print(f"Type of Dataset: {type_dataset}")

    dataset_name = type_dataset.lower()
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list+= [normalize_transforms[type_dataset]]
    train_transform = transforms.Compose(transform_list)    

    filepath = f"{parameters['data_path']}{dataset_name}"
    print(f"filepath dataset: {filepath}")

    if type_dataset=="mvtec":
        from argparse import Namespace
        opt = utility_logging.from_parameters_to_opt(parameters)  
        apply_rotation=parameters.get("apply_rotation",False)
        opt.apply_rotation=apply_rotation
        dataset_train = MVTecDataset(opt, is_train=True)
        opt.apply_rotation=False
        dataset_test = MVTecDataset(opt,is_train=False) 
    else:
        raise ValueError(f"{type_dataset} dataset is not present !")

    return dataset_train, dataset_test


'''
Import mvtec dataset

├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   ├── broken_small
│   │   └── contamination
│   ├── test
│   │   ├── broken_large
│   │   ├── broken_small
│   │   ├── contamination
│   │   └── good
│   └── train
│       └── good
...

'''
        

class MVTecDataset(Dataset):
    '''
    Import MVTecDataset with all the classes containing objects  
    '''
    def __init__(self, opt,is_train=True):
        assert opt.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(opt.class_name, MVTEC_CLASS_NAMES)
        self.opt = opt
        self.dataset_path = opt.data_path
        self.class_name = opt.class_name
        self.is_train = is_train
        self.cropsize = opt.crp_size
        self.use_all_classes = opt.use_all_classes
        self.only_normal = opt.only_normal
        self.only_anomalies = opt.only_anomalies
        self.apply_rotation = opt.apply_rotation

        if self.only_normal is True and self.only_anomalies is True:
            raise ValueError("only_normal=True and only_anomalies=True")

        if self.use_all_classes:
            self.x, self.y, self.anomaly_info, self.mask, self.filepaths = self.load_dataset_folders()
        else:
            self.x, self.y, self.anomaly_info, self.mask, self.filepaths = self.load_dataset_folder()
        # mask
        self.transform_mask = transforms.Compose([transforms.ToTensor()])

        rgb_dict = { "white":(255,255,255),"blue":(80,116,151),"brown":(164,115,93),"black":(0,0,0),"gray":(190,190,190)  }
        colors = ["white","blue","gray","black","black","black","gray","black","brown","white"]
        rgb_colors = [ rgb_dict[color_class] for color_class in colors ]
        self.rgb_colors = rgb_colors

        new_x,new_y,new_anomaly_info,new_mask,new_filepaths = [],[],[],[],[]
        
        self.y = np.asarray(self.y)
        self.targets = np.asarray(self.y)
        self.anomaly_info = np.asarray(self.anomaly_info)
        self.loaded_from_memory = False


    def __getitem__(self, idx):
        x, y,anomaly_info, mask,filepath = self.x[idx], self.y[idx],self.anomaly_info[idx], self.mask[idx], self.filepaths[idx]
        try:
            x = Image.open(x)
        except:
            print(f"I am not able to load image at path: {x}")

        crp_size = self.cropsize
        if self.apply_rotation:
            rotation_degree = self.rotation_degrees[idx]
            fill = self.rgb_colors[y]
            transform_img = create_transform_x_with_rotation(self.opt,crp_size,rotation_degree,fill)     
        else:
            transform_img = create_transform_x(self.opt,crp_size)
            

        class_name = MVTEC_CLASS_NAMES[y]
        if class_name in ['zipper', 'screw', 'grid'] and self.loaded_from_memory is False:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        x = transform_img(x)

        if anomaly_info == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
            
        return x, y, idx, anomaly_info, filepath
        
    def get_wrapper(self,idx):
        x, y, idx, anomaly_info, filepath  = self.__getitem__(idx)
        mask = self.mask[idx]
        diz = {"x":x, "y":y, "idx":idx, "anomaly_info":anomaly_info,"mask":mask, 
              "filepath":filepath, "real_A":x}
        from argparse import Namespace
        ns = Namespace(**diz)
        return ns

    def get_mask(self,mask_path,anomaly_info):
        self.transform_mask = transforms.Compose(
            [transforms.Resize(self.cropsize, Image.NEAREST),
             transforms.ToTensor()])
        if anomaly_info == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask_path)
            mask = self.transform_mask(mask)
        return mask 


    def __len__(self):
        return len(self.x)

    def load_dataset_folders(self):
        lista_x,lista_y,lista_anomaly_info,lista_mask,lista_filepaths = [],[],[],[],[]
        for class_name in MVTEC_CLASS_NAMES:
            self.class_name = class_name
            x, y, anomaly_info,mask,filepaths = self.load_dataset_folder()
            lista_x.extend(x)
            lista_y.extend(y)
            lista_anomaly_info.extend(anomaly_info)
            lista_mask.extend(mask) 
            lista_filepaths.extend(filepaths)
        return lista_x,lista_y,lista_anomaly_info,lista_mask,lista_filepaths


    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, anomaly_info, y, mask,filepaths = [], [], [], [],[]

        index = np.where( np.asarray(MVTEC_CLASS_NAMES)==self.class_name)[0][0]
        index_class = index

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])

            # load gt labels
            if img_type == 'good':
                if self.only_anomalies is False:
                    anomaly_info.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                    y.extend([index_class] * len(img_fpath_list))
                    x.extend(img_fpath_list)
                    filepaths.extend(img_fpath_list)
            else:
                if self.only_normal is False:
                    anomaly_info.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)
                    y.extend([index_class] * len(img_fpath_list))
                    x.extend(img_fpath_list)
                    filepaths.extend(img_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(anomaly_info), list(mask), list(filepaths) 


class MemoryDataset(Dataset):
    def __init__(self, filepaths, strategy): #, dataset_current_task
    
        self.strategy = strategy

        self.filepaths = filepaths

        filepaths_dict = [ filepath_dict for filepath_dict,filepath_img in filepaths]
        indices_original,filepaths_original,class_ids = [],[],[]
        for filepath_dict in filepaths_dict:
            f = open(filepath_dict, "rb")
            diz = pickle.load(f)
            f.close()

            y, idx, anomaly_info, filepath = diz["y"],diz["idx"],diz["anomaly_info"],diz["filepath"]
            indices_original.append(idx)
            filepaths_original.append(filepath)
            class_ids.append(y)
        self.indices_original = np.asarray(indices_original)
        self.filepaths_original = np.asarray(filepaths_original)
        self.class_ids = np.asarray(class_ids)


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath_dict,filepath_img = self.filepaths[idx]

        f = open(filepath_dict, "rb")
        diz = pickle.load(f)
        f.close()

        y, idx, anomaly_info, filepath = diz["y"],diz["idx"],diz["anomaly_info"],diz["filepath"]
        class_id = y

        img = Image.open(filepath_img)

        transform_img = create_transform_img(img.size[0])
        x = transform_img(img)

        return x, np.asarray(y), np.asarray(idx), np.asarray(anomaly_info), filepath

        
