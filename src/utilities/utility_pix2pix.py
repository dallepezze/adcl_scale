import os
import numpy as np
import torch
from filelock import FileLock
from PIL import Image

from src.utilities.utility_images import *
from src.utilities.utility_ad import MSGMS_Score
from src.datasets import *
from src.utilities.utility_logging import from_sample_to_dict,save_pickle
from src.inpaint.gen_mask import gen_mask


def create_images_masked(k_value, image):
    img_size = image.shape[1]
    Ms_generator = gen_mask(k_value, 3, img_size )
    Ms = next(Ms_generator)
    inputs = [ (image * (torch.tensor(mask, requires_grad=False))) for mask in Ms]
    inputs = torch.stack(inputs)
    return inputs,Ms

def create_summary(lista_tensor_images):
    lista_numpy_images = [ convert2img(img_tensor.cpu()) for img_tensor in lista_tensor_images ]
    summary = np.hstack(lista_numpy_images)
    summary= summary.astype(np.uint8)
    summary = Image.fromarray(summary)
    return summary

def create_summary_by_numpy(lista_numpy_images):
    summary = np.hstack(lista_numpy_images)
    summary= summary.astype(np.uint8)
    summary = Image.fromarray(summary)
    return summary

def forward_train(model, real_A, real_B, device):
    model.real_A = real_A.to(model.device)
    model.real_B = real_B.to(model.device)
    model.optimize_parameters()
    fake_B = model.fake_B
    return fake_B

def forward_test(model, real_A, real_B, device):
    model.real_A = real_A.to(model.device)
    if real_B!=None:
        model.real_B = real_B.to(model.device)
    model.test()
    fake_B = model.fake_B
    return fake_B

def logging_images(self,batch, index_sample_batch,real_A, real_B, fake_B, dataloader, complete_dataset, index_training, batch_index):
    # Log images
    mode = self.strategy.mode
    self.mode = mode
    batch_size = self.strategy.parameters["batch_size"]
    class_ids, indices = batch[1], batch[2]
    class_id = class_ids[index_sample_batch]
    index_dataset = indices[index_sample_batch]
    sample_id = index_dataset
    sample = complete_dataset[index_dataset]
    diz = from_sample_to_dict(sample)

    dir_path = os.path.join( self.strategy.path_logs, f"{self.mode}", f"T{index_training}",  f"{class_id}" )
    lock = FileLock("high_ground.txt.lock")
    with lock:
        isExist = os.path.exists(dir_path)
        if isExist is False:
            os.makedirs(dir_path,exist_ok=True)

    if self.mode=="test" or self.mode=="train":
        summary = create_summary([real_A[0],real_B[0],fake_B[0]])   

        # save all images in local
        filepath =  os.path.join(dir_path, f"summary_{sample_id}.png" )
        summary.save(filepath)

        img = convert2img(fake_B[0])
        img = Image.fromarray(img) 
        filepath_img = os.path.join(dir_path,f"{sample_id}.png")
        img.save(filepath_img) 

    if "reconstruct" in self.mode or self.mode=="evaluate_data":
        summary = create_summary([real_B[0],fake_B[0]])  
        filepath =  os.path.join(dir_path, f"summary_{sample_id}.png" )
        summary.save(filepath)

        if batch_index==(len(dataloader)-1) and index_sample_batch==0:  
            self.strategy.run[f"Task_Results/T{index_training}"][f"visual_{mode}"].log(summary) 

        filepath_diz = os.path.join(dir_path,f"{sample_id}.pickle")
        save_pickle(diz,filepath_diz)  

        img = convert2img(fake_B[0])
        img = Image.fromarray(img) 
        filepath_img = os.path.join(dir_path,f"{sample_id}.png")
        img.save(filepath_img) 


def produce_reconstruction_inpaint(trainer,model, complete_dataset, batch, index_sample_batch):
    strategy = trainer.strategy
    msgms_score = MSGMS_Score()
    mse_loss_func = trainer.mse_loss
    index_training = strategy.index_training
    index_dataset = batch[2][index_sample_batch]

    output_by_k = []
    total_score = 0
    k_values = strategy.parameters["k_value"]
    for k in k_values:
        image_tensor = batch[0][index_sample_batch]
        output, score, mse_loss = forward_k_value(model, msgms_score, mse_loss_func, image_tensor, [k])

        summary = create_summary([image_tensor,output])
        dir_path = os.path.join( strategy.path_logs, f"T{index_training}", "visual_val" )
        os.makedirs(dir_path, exist_ok=True)
        filepath =  os.path.join(dir_path, f"{index_dataset}_k={k}.png" )
        summary.save(filepath)

        # print(f"output.shape: {output.shape}")
        output_by_k.append(output)
        img_size = batch[0].size(-1)
        N = img_size // k
        total_score+=score/(N**2)

def forward_k_value(model, msgms_score,mse_loss_func, image_tensor, k_value):
    inputs,Ms = create_images_masked(k_value, image_tensor)

    # Itero sulle images masked
    outputs = []
    for index_inputs in range(len(inputs)):
        real_A = inputs[index_inputs:index_inputs+1]
        real_B = torch.stack([image_tensor])
        fake_B = forward_test(model, real_A, real_B, model.device).cpu()
        outputs.append(fake_B[0])
        summary = create_summary([real_A[0],real_B[0],fake_B[0]])        
    output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False)), outputs, Ms))  

    mse_loss = mse_loss_func(image_tensor,output).item()
    image_tensor_2 = torch.stack([image_tensor]).to(model.device)
    output_2 = torch.stack([output]).to(model.device)
    score = msgms_score(image_tensor_2, output_2 ).cpu()
    return output, score, mse_loss  


def produce_reconstruction_scaling_model(trainer,model, complete_dataset, batch, index_sample_batch):
    h,w = 32,32
    real_B = batch[0][index_sample_batch]
    real_A,real_A_32 = produce_scaled_A_from_B(real_B.clone(),h,w)

    real_B = batch[0][index_sample_batch:index_sample_batch+1]
    real_A = torch.stack([real_A])
    real_A_32 = np.stack([real_A_32])

    fake_B = forward_test(model, real_A, real_B, model.device).cpu()
    return real_B,fake_B


def  produce_input_scaling_model(trainer, complete_dataset, batch, index_sample_batch):
    #Preparo real_A
    h,w = 32,32
    real_B = batch[0][index_sample_batch]
    real_A,real_A_32 = produce_scaled_A_from_B(real_B.clone(),h,w)

    real_B = batch[0][index_sample_batch:index_sample_batch+1]
    real_A = torch.stack([real_A])
    real_A_32 = np.stack([real_A_32])

    return real_A, real_B

def reconstruct_img(model,real_B,filepath):
    w = 32
    h = 32

    img_size = real_B.shape[1]

    covered = np.zeros((img_size,img_size))
    covered_counters = np.zeros((img_size,img_size,3))
    # new_img = convert2img(real_B.clone())
    new_img = np.zeros((img_size,img_size,3))
    new_img = np.asarray(new_img)
    covered_perc = 0

    num_it = 0
    while(covered_perc<0.99):
        real_A,boxes = produce_black_boxes(real_B.clone(),h,w)
        images_B = [ real_B ]
        images_A = [ real_A ]
        filepaths_A = [ filepath ]
        data = { "A":torch.stack(images_A), "B":torch.stack(images_B), 'A_paths':filepaths_A, 'B_paths':filepaths_A  }
        model.set_input(data)  # unpack data from data loader

        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        _, fake_images_B, _ = visuals["real_A"],visuals["fake_B"],visuals["real_B"]
        fake_B = fake_images_B[0]

        fake_B = convert2img(fake_B)

        for box in boxes:
            x,y,h,w = box
            new_img[y:y+h,x:x+w] += fake_B[y:y+h,x:x+w]
            covered[y:y+h,x:x+w] = 1
            covered_counters[y:y+h,x:x+w] += 1

        covered_perc = np.sum(covered)/(img_size*img_size)
        num_it+=1

    real_B = convert2img(real_B)

    cond = np.where(covered==0)
    indices = list(zip(cond[0],cond[1]))
    for index in indices:
        y,x = index
        new_img[index] = real_B[index]
        covered_counters[y,x] += 1

    new_img = new_img/covered_counters
    rec_img = new_img

    return rec_img

# black boxes
def produce_black_boxes(img,h,w):
    img = img.detach().cpu().numpy()
    img_size = img.shape[1]
    img = convert2img(img)
    img = np.asarray(img)
    boxes = []
    for k in range(20):
        y,x = np.random.randint(0,256-w,(2,))
        img[y:y+h,x:x+w] = 0 
        boxes.append((x,y,h,w))
    transform_tensor = create_transform_img(256)
    img = Image.fromarray(img) 
    img = transform_tensor(img)
    return img,boxes

# given a tensor image img
# return scaled version as numpy 32x32
# and rescaled image as torch tensor 256x256
# 256x256 -> 32x32(real_A_32 - numpy) -> 256x256(real_A - tensor)
def produce_scaled_A_from_B(img,h,w):
    img = img.detach().cpu().numpy()
    img_size = img.shape[1]
    img = convert2img(img)

    img = resize_numpy_img(img,32)
    img_32 = img.copy()
    img = resize_numpy_img(img,256)

    transform_tensor = create_transform_img(256)
    img = Image.fromarray(img) 
    img = transform_tensor(img)

    return img, img_32