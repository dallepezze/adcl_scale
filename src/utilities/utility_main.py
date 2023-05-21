import torch
from torch.utils.data import DataLoader,ConcatDataset
import numpy as np  
import json
import sys
import neptune.new as neptune

ROOT = ".."
sys.path.append(ROOT)


from src.strategy_ad import *
from src.models.vae import *
from src.datasets import *
from src.utilities import utility_logging
from src.datasets import *
from src.strategy_ad import *
from src.models.vae import *
from src.trainer.trainer_vae import *


def create_new_labels_map(original_labels_map, task_order, num_tasks):
    labels_map_new = {}
    for index_task in range(num_tasks):
        task_id = task_order[index_task]
        if np.asarray(task_id).ndim==0:
            label = labels_map[task_id]
            labels_map_new[index_task] = label
        else: #np.asarray(classes_ids).ndim>0:
            classes_ids = task_id
            label = [ labels_map[class_id] for class_id in classes_ids  ]
            label = ",".join(label)        
            labels_map_new[index_task] = label       
        
    labels_map = labels_map_new
    return labels_map
    
def set_default_ad_parameters(parameters):
    num_tasks = parameters["num_tasks"]
    anomaly_detection_task = parameters.setdefault("anomaly_detection_task", False)
    anomaly_detection_task_with_metrics = parameters.setdefault("anomaly_detection_task_with_metrics", True)
    test_only_seen_tasks = parameters.setdefault("test_only_seen_tasks",False)
    return  anomaly_detection_task,anomaly_detection_task_with_metrics,test_only_seen_tasks

def give_ad_parameters(parameters,index_training):
    num_tasks = parameters["num_tasks"]
    anomaly_detection_task = parameters.get("anomaly_detection_task")
    anomaly_detection_task_with_metrics = parameters.get("anomaly_detection_task_with_metrics")
    test_only_seen_tasks = parameters.get("test_only_seen_tasks")
    test_all_dataset_together = parameters.get("test_all_dataset_together")

    if test_only_seen_tasks and test_all_dataset_together==False:
        num_tasks_to_examine = index_training+1 
    elif test_all_dataset_together:
        num_tasks_to_examine = 1
    elif parameters["sample_strategy"]=="multi_task":
        num_tasks_to_examine = 10
    else:
         num_tasks_to_examine = num_tasks

    return  anomaly_detection_task,anomaly_detection_task_with_metrics,test_only_seen_tasks,num_tasks_to_examine

def give_memory_parameters(parameters):
    use_memory = parameters.get("use_memory")
    memory_dataset_path_train = parameters.get("memory_dataset_path_train")
    memory_dataset_path_test = parameters.get("memory_dataset_path_test")
    type_memory_train = parameters.get("type_memory_train")
    type_memory_test = parameters.get("type_memory_test")
    memory_model_path = parameters.get("memory_model_path")
    new_memory = parameters.get("new_memory")
    sample_strategy = parameters.get("sample_strategy")

    return use_memory,memory_dataset_path_train,memory_dataset_path_test,type_memory_train,type_memory_test,memory_model_path,new_memory,sample_strategy

def set_default_memory_parameters(parameters):
    parameters.setdefault("use_memory", False)
    parameters.setdefault("type_memory_train","memorized")
    parameters.setdefault("type_memory_test","memorized")
    parameters.setdefault("memory_dataset_path_train", "")
    parameters.setdefault("memory_dataset_path_test", "")
    parameters.setdefault("memory_model_path", "")

    parameters.setdefault("use_memory_reconstruct", False)
    parameters.setdefault("type_memory_reconstruct_train","")
    parameters.setdefault("type_memory_reconstruct_test","")
    parameters.setdefault("memory_reconstruct_dataset_path_train", "")
    parameters.setdefault("memory_reconstruct_dataset_path_test", "")

    parameters.setdefault("new_memory", True)
    parameters.setdefault("sample_strategy", "naive")

    use_memory,memory_dataset_path_train,memory_dataset_path_test,type_memory_train,type_memory_test,memory_model_path,new_memory,sample_strategy  = give_memory_parameters(parameters)

    if new_memory and use_memory==True:
        raise ValueError("use_memory is True while new_memory is also True")

    if new_memory:
        if memory_dataset_path_train!="" or memory_dataset_path_test!="":
            raise ValueError("memory_dataset_path parameter is not empty")

    if use_memory:
        if sample_strategy!="":
            print(f"Attention: Sample strategy {sample_strategy} has not effect")

def load_memory_main(strategy,memory_dataset_path,type_memory):
    self = strategy
    index_training = strategy.index_training
    task_order = strategy.task_order

    if self.parameters["use_memory"] and memory_dataset_path!="":
        print(f"Load memory train: {memory_dataset_path} of type: {type_memory}")
        current_task = index_training+1
        memory = load_memory(self,memory_dataset_path,type_memory, task_order, current_task)
        self.memory = memory

        dataset_current_task = memory.tasks_memory[index_training]
        self.dataset_current_task = dataset_current_task
        current_train_data_loader = DataLoader(dataset_current_task, shuffle=True, batch_size=self.batch_size) 
    else:
        print(f"Create new memory train with sample_strategy: {self.sample_strategy}")
        memory = create_memory(self, self.task_order, self.path_logs, self.mem_size, self.index_training, self.sample_strategy)
        self.memory = memory

def load_memory_reconstruct_main(strategy,type, memory_reconstruct_dataset_path, type_memory_reconstruct):
    index_training = strategy.index_training
    task_order = strategy.task_order

    if strategy.parameters["use_memory_reconstruct"] and  memory_reconstruct_dataset_path!="":   
        print(f"Loading reconstruct memory: {memory_reconstruct_dataset_path}/{type_memory_reconstruct}")

        if strategy.parameters.get("test_only_seen_tasks") is False:
            load_all_tasks = True
        else:
            load_all_tasks = False

        memory_reconstruct = load_memory(strategy, memory_reconstruct_dataset_path, type_memory_reconstruct, strategy.task_order, strategy.index_training+1, load_all_tasks)

        if type=="train":
            strategy.memory_reconstruct_train = memory_reconstruct
        elif type=="test":
            strategy.memory_reconstruct_test = memory_reconstruct
        else:
            raise ValueError(f"type {type} unknown")


def memory_update_main(strategy):
    self = strategy
    # MEMORY UPDATE
    if self.parameters["new_memory"]:
        for task_id_old in range(0,self.index_training+1):
            self.memory.task_id_old = task_id_old
            if self.index_training==task_id_old:
                self.memory.memory_update(self.task_train_dataset, self.index_training)
            else:
                self.memory.memory_update(self.memory.tasks_memory[task_id_old], self.index_training)

def load_model_main(strategy):
    if strategy.num_epochs>0 and strategy.use_model:
        raise ValueError(f"use_model:True but num_epochs>0")
    if strategy.use_model and strategy.save_model:
        raise ValueError(f"use_model:True but save_model is also True")
    self = strategy
    index_training = self.index_training
    memory_model_path = self.parameters.get("memory_model_path","")
    if self.parameters["use_model"] and memory_model_path=="":
        raise ValueError("use_model:True but memory_model_path is empty")
    if self.parameters["use_model"] and memory_model_path!="":    
        utility_logging.load_model(self, self.parameters["architecture"], memory_model_path, index_training) 
    

def save_model_main(strategy):
    self = strategy
    save_model_param = self.save_model
    index_training = self.index_training
    #SAVE MODEL 
    if save_model_param:
        print("Save model")
        utility_logging.save_model(self, self.parameters["architecture"], strategy.path_logs, index_training)

def init_strategy_variables(strategy, complete_train_dataset,complete_test_dataset,train_stream,test_stream,original_complete_train_dataset, original_complete_test_dataset, original_train_stream, original_test_stream,labels_map,run,path_logs):
    strategy.complete_train_dataset = complete_train_dataset
    strategy.complete_test_dataset = complete_test_dataset
    strategy.train_stream = train_stream
    strategy.test_stream = test_stream
    strategy.labels_map = labels_map


    strategy.original_complete_train_dataset = original_complete_train_dataset
    strategy.original_complete_test_dataset = original_complete_test_dataset
    strategy.original_train_stream = original_train_stream
    strategy.original_test_stream = original_test_stream

    from collections import defaultdict
    strategy.metrics_train = defaultdict(lambda: {})    #task_id:{epoch0:{train_loss,val_loss}}
    strategy.metrics_test = defaultdict(lambda:{}) #task_id:{task0:{loss,f1,...}}
    strategy.other_data_train = defaultdict(lambda:{}) #task_id:{epoch0:{xs,x_hats,zs,mus...}}
    strategy.other_data_test = defaultdict(lambda:{}) #task_id:{task0:{xs,x_hats,zs,mus...}}

    strategy.run = run
    strategy.labels_map = labels_map
    strategy.path_logs = path_logs


def init_execute(credentials_path, default_path,parameters_path,create_run=True):
    f = open(f"{credentials_path}", "rb")
    credentials = json.load(f)  
    f.close()

    f = open(f"{default_path}", "rb")
    parameters_default = json.load(f)
    parameter_common = parameters_default.copy()
    f.close()

    f = open(f"{parameters_path}", "rb")
    parameters = json.load(f)
    parameters_specific = parameters.copy()
    f.close()

    f = open(f"../configurations/models/{parameters['architecture']}.json", "rb")
    parameters_architecture = json.load(f)
    parameters_architecture_copy = parameters_architecture.copy()
    f.close()

    new_parameters = {}
    new_parameters.update(parameters_default)
    new_parameters.update(parameters_architecture)
    new_parameters.update(parameters_specific)
    parameters = new_parameters

    if "tags" not in parameters:
        parameters["tags"] = [parameters["architecture"],parameters["sample_strategy"]]
        if "trainer" in parameters and parameters["trainer"]!="":
            parameters["tags"]=parameters["tags"]+[parameters["trainer"]]

    print(f"parameters: {parameters}")

    if "device_id" in parameters:
        device_id = parameters["device_id"]
    else:
        device_id = 0
    torch.cuda.set_device(device_id)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{device_id}' if use_cuda else 'cpu')
    print(device)
    parameters["device"] = device

    #default values
    set_default_memory_parameters(parameters)
    set_default_ad_parameters(parameters)

    if create_run:
        if "run_name" not in parameters:
            run = neptune.init(project=credentials["project_name"],
                api_token=credentials["api_token"],
                tags = parameters["tags"])

            run["config/hyperparameters"] = parameters
            run["config/hyperparameters_common"] = parameter_common
            run["config/hyperparameters_specific"] = parameters_specific
            run["config/hyperparameters_architecture"] = parameters_architecture_copy
        else:
            run_name = parameters["run_name"]
            run = neptune.init(project=credentials["project_name"],
                api_token=credentials["api_token"],
                tags = parameters["tags"],
                run=run_name )
    else:
        run=None

    return run,parameters,device


def load_and_split_dataset(parameters,dataset_name,num_tasks,task_order):
    # Load Dataset
    complete_train_dataset, complete_test_dataset = load_dataset(parameters,type_dataset=dataset_name,download=True)
    print(f"complete_train_dataset: {len(complete_train_dataset)}")
    print(f"complete_test_dataset: {len(complete_test_dataset)}")
    # Split Dataset in train_stream and test_stream
    cl_benchmark = ContinualLearningBenchmark(complete_train_dataset, complete_test_dataset, num_tasks, task_order)
    train_stream,test_stream = cl_benchmark.produce_task_stream()
    return complete_train_dataset, complete_test_dataset,train_stream,test_stream


def create_new_labels_map(labels_map,task_order,num_tasks):
    labels_map_new = {}
    for index_task in range(num_tasks):
        task_id = task_order[index_task]
        if np.asarray(task_id).ndim==0:
            label = labels_map[task_id]
            labels_map_new[index_task] = label
        else: 
            classes_ids = task_id
            label = [ labels_map[class_id] for class_id in classes_ids  ]
            label = ",".join(label)        
            labels_map_new[index_task] = label       
        
    labels_map = labels_map_new
    return labels_map

def manage_dataset(strategy, parameters,complete_train_dataset,complete_test_dataset,train_stream,test_stream):
    sample_strategy = strategy.parameters.get("sample_strategy")
    dataset_path_train = strategy.parameters.get("dataset_path_train")
    dataset_path_test = strategy.parameters.get("dataset_path_test")
    type_folder_train = strategy.parameters.get("type_folder_train")
    type_folder_test = strategy.parameters.get("type_folder_test")
    
    task_order = strategy.task_order
    num_tasks = strategy.num_tasks


    if dataset_path_train!="" and type_folder_train!="":
        print(f"Loading folder train_dataset: {dataset_path_train}/{type_folder_train}")
        train_stream,complete_train_dataset = load_dataset_from_memory(strategy,dataset_path_train,type_folder_train,task_order,num_tasks)
    if dataset_path_test!="" and type_folder_test!="":
        print(f"Loading folder test_dataset: {dataset_path_test}/{type_folder_test}")
        test_stream,complete_test_dataset_new = load_dataset_from_memory(strategy,dataset_path_test,type_folder_test,task_order,num_tasks=10)
        print(f"test_stream length: {len(test_stream)}")
        for test_dataset in test_stream:
            for i in range(len(test_dataset)):
                index_dataset = test_dataset[i][2]
                filepath_img = test_dataset.filepaths[i][1]
                # new_img = complete_test_dataset_new[i][0]
                complete_test_dataset.x[index_dataset] = filepath_img
        complete_test_dataset.loaded_from_memory=True
        print("Loading folder test_dataset finished")

    if sample_strategy=="MultiTask" or sample_strategy=="multi_task":
        print(f"Sample Strategy used: {sample_strategy}")
        train_stream = [ ConcatDataset(train_stream) ]
        strategy.num_tasks = 1
        strategy.parameters["num_tasks"] = 1
        
    if sample_strategy=="cumulative" or sample_strategy=="Cumulative":
        print(f"Sample Strategy used: {sample_strategy}")
        new_train_stream = [ ConcatDataset(train_stream[0:i+1]) for i in range(num_tasks) ]
        train_stream = new_train_stream
        for i in range(len(train_stream)):
            print(f"{i}: {len(train_stream[i])}")

    use_train_as_test = parameters.get("use_train_as_test")
    if use_train_as_test:
        test_stream = train_stream
        complete_test_dataset = complete_train_dataset

    test_all_dataset_together = parameters.get("test_all_dataset_together")
    test_only_seen_tasks = parameters.get("test_only_seen_tasks")
    if test_all_dataset_together:
        if test_only_seen_tasks:
            print(f"num_tasks: {num_tasks}")
            for i in range(num_tasks):
                print(f"len(test_stream[{i}]): {len(test_stream[i])}")
            test_stream = [ ConcatDataset(test_stream[0:i+1]) for i in range(num_tasks) ]
            print()
            for i in range(num_tasks):
                print(f"len(test_stream{i}]): {len(test_stream[i])}")
        else: 
            test_stream = [ complete_test_dataset for i in range(num_tasks) ]

    return complete_train_dataset,complete_test_dataset,train_stream,test_stream

    
def memory_update_main(strategy):
    self = strategy
    new_memory = strategy.new_memory
    if new_memory:
        for task_id_old in range(0,self.index_training+1):
            self.memory.task_id_old = task_id_old
            if self.index_training==task_id_old:
                self.memory.memory_update(self.task_train_dataset, self.index_training)
            else:
                self.memory.memory_update(self.memory.tasks_memory[task_id_old], self.index_training)



