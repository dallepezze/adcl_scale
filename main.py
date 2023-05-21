import argparse
import os
import sys
import shutil
from datetime import datetime


ROOT = ".."
sys.path.append(ROOT)
sys.path.append(ROOT+"/pytorch_pix2pix")

from src.models.pix2pix_model import *
from src.strategy_ad import *
from src.trainer.trainer_cae import *
from src.datasets import *
from src.utilities.utility_main import *
from src.utilities import utility_logging
from src.utilities.utility_models import *


seed = 0
torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Parser to take filepaths")
parser.add_argument("--parameters_path", type=str, nargs="?", action = 'store', help="parameters path", default="test_pix2pix.json" )
parser.add_argument("--credentials_path", type=str, nargs="?", action = 'store', help="credentials path", default="davi.json")
parser.add_argument("--default_path", type=str, nargs="?", action = 'store', help="default parameters path", default="common_param.json")

args = parser.parse_args()
path = '../configurations'
parameters_path = os.path.join(path,args.parameters_path)
print(f"parameters_path: {parameters_path}")
credentials_path = os.path.join(path,args.credentials_path)
print(f"credentials_path: {credentials_path}")
default_path = os.path.join(path,args.default_path)
print(f"default_path: {default_path}")

# Define run,parameters,path_logs
run, parameters, device = init_execute(credentials_path, default_path, parameters_path)
project_name = run._project_name
experiment_name = run._short_id

now = datetime.now() # current date and time
date_time = now.strftime("%d_%m_%Y__%H:%M:%S")
path_logs = os.path.join(f"/logs/{project_name}/{experiment_name}_{date_time}")
print(f"path_logs: {path_logs}")
utility_logging.create_paths([path_logs])


filename = os.path.basename(parameters_path)
dst = os.path.join(path_logs,filename)
shutil.copyfile(parameters_path, dst)

# Load Dataset
channels,dataset_name,num_tasks,task_order = parameters["channels"],parameters["dataset_name"],parameters["num_tasks"],parameters["task_order"]
complete_train_dataset, complete_test_dataset,train_stream,test_stream = load_and_split_dataset(parameters,dataset_name,num_tasks,task_order)

labels_map = create_new_labels_map(labels_datasets[dataset_name], task_order, num_tasks)
print(f"labels_map: {labels_map}")

# Create Strategy
input_size = complete_train_dataset[0][0].shape
print(f"input_size: {input_size}")

original_stdout = sys.stdout # Save a reference to the original standard output
filepath = os.path.join(path_logs, 'model_info.txt')
with open( filepath, 'w') as f:
    sys.stdout = f
    strategy = create_strategy(parameters,run,labels_map,device,path_logs,input_size)
sys.stdout = original_stdout 

import copy
original_complete_train_dataset, original_complete_test_dataset, original_train_stream, original_test_stream = copy.deepcopy(complete_train_dataset),copy.deepcopy(complete_test_dataset),copy.deepcopy(train_stream),copy.deepcopy(test_stream)
complete_train_dataset,complete_test_dataset,train_stream,test_stream = manage_dataset(strategy, parameters,complete_train_dataset,complete_test_dataset,train_stream,test_stream)
num_tasks = strategy.num_tasks

init_strategy_variables(strategy, complete_train_dataset,complete_test_dataset,train_stream,test_stream,original_complete_train_dataset, original_complete_test_dataset, original_train_stream, original_test_stream,labels_map,run,path_logs)

sample_strategy = strategy.parameters.get("sample_strategy")
test_only_seen_tasks = strategy.parameters.get("stest_only_seen_tasks")
if sample_strategy=="multi_task" and test_only_seen_tasks:
    raise ValueError("test_only_seen_tasks is True but you are in multi_task mode")

for index_training in range(0,num_tasks):
    train_dataset = train_stream[index_training]
    test_dataset = test_stream[index_training]

    strategy.index_training = index_training
    strategy.train_task_id = task_order[index_training]
    strategy.task_label = labels_map[index_training]
    task_label = strategy.task_label

    print(f"\nStart Training Task T{index_training} ({ task_label })")
    
    current_train_dataset,current_test_dataset = strategy.init_variables_dataset(train_dataset,test_dataset )

    # LOAD Memory 
    # assign memory to strategy and load it from memory(use_memory) or create a new one(new_memory)    use_memory,memory_dataset_path_train,memory_dataset_path_test,type_memory_train,type_memory_test,memory_model_path,new_memory,sample_strategy = give_memory_parameters(strategy.parameters)
    load_memory_main(strategy, strategy.parameters["memory_dataset_path_train"], strategy.parameters["type_memory_train"])

    #Load MemoryReconstruct
    load_memory_reconstruct_main(strategy, "train", strategy.parameters["memory_reconstruct_dataset_path_train"], strategy.parameters["type_memory_reconstruct_train"])
    load_memory_reconstruct_main(strategy, "test" , strategy.parameters["memory_reconstruct_dataset_path_test"], strategy.parameters["type_memory_reconstruct_test"])

    if strategy.sample_strategy=="single_model":
        print("Reset Trainer")
        reset_trainer(strategy)

    # LOAD MODEL (if memory_model_path!="")
    load_model_main(strategy)

    # SAVE MODEL
    if index_training==0:
        save_model_main(strategy)

    # TRAINING
    print(f"\nTraining Task T{index_training}")
    batch_size = strategy.batch_size 
    num_epochs = strategy.num_epochs 
    strategy.training_task(current_train_dataset,current_test_dataset,num_epochs,batch_size)

    memory_update_main(strategy)

    # SAVE MODEL
    save_model_main(strategy)
    
    # EVALUATION
    print("\nEvaluation:")
    strategy.evaluate_test_stream(test_stream, batch_size=8)

    plt.close("all")

run["Finished"].log(True)




