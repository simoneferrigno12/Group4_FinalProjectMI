# USAGE
# python3 train.py

# import the necessary packages
from training.dataset import TaskDataLoader
from training.MultiTaskLoss import MultiTaskLoss
from training.model import UNet
from training import config
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np

# load the image and mask filepaths in a sorted manner
root_dir = 'MI_FinalProject/dataset/train'
csv_file = 'MI_FinalProject/dataset/crossValidationCsv/foldA_train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the train datasets
trainData = TaskDataLoader(csv_file=csv_file, root_dir=root_dir)
    
print(trainData)

numTrainSamples = int(len(trainData))

print(f"[INFO] found {len(trainData)} examples in the training set...")

# create the training data loaders
trainLoader = DataLoader(trainData, shuffle=True,	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,	num_workers=1)


# initialize our UNet model
unet = UNet(config.NUM_CHANNELS, config.NUM_CLASSES).to(config.DEVICE)

# initialize loss function and optimizer
#lossFunc = TverskyLoss(alpha=0.3, beta=0.7)
lossFunc=MultiTaskLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainData) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss_1": [], "train_loss_2": [], "train_loss_3": []}
#H = {"train_loss_1": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
      
    totalTrainLoss1 = 0
    totalTrainLoss2 = 0
    totalTrainLoss3 = 0
    
  	# loop over the training set
    for (i, (x, mask, label, intensity)) in enumerate(trainLoader):
 		    # send the input to the device
        x, mask, label, intensity = x.to(config.DEVICE), mask.to(config.DEVICE), label.to(config.DEVICE), intensity.to(config.DEVICE)
        
        pred = unet(x)
        
        loss = lossFunc(pred, mask, label, intensity)
        
        totalTrainLoss1 = totalTrainLoss1 + loss[0].item()
        totalTrainLoss2 = totalTrainLoss2 + loss[1].item()
        totalTrainLoss3 = totalTrainLoss3 + loss[2].item()
        
        #gradNorm
        weighted_task_loss = torch.mul(unet.weights, loss)
        if i == 0:
            initial_task_loss = loss.data.cpu().numpy()
        total_loss = torch.sum(weighted_task_loss)
        opt.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=True)
        unet.weights.grad.data = unet.weights.grad.data * 0.0
        
        W = unet.get_last_shared_layer()
        norms = []
        for i in range(len(loss)):
            gygw = torch.autograd.grad(loss[i], W.parameters(), retain_graph=True)
            
            norms.append(torch.norm(torch.mul(unet.weights[i], gygw[0])))
        norms = torch.stack(norms)
        
        
        loss_ratio = loss.data.cpu().numpy() / initial_task_loss
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        
        mean_norm = np.mean(norms.data.cpu().numpy())
        
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.06), requires_grad=False).cuda()
        
        GradLoss = torch.nn.L1Loss(reduction = 'sum')
        grad_norm_loss = 0
        for loss_index in range(0, len(loss)):
            grad_norm_loss = torch.add(grad_norm_loss, GradLoss(norms[loss_index], constant_term[loss_index]))
            
        unet.weights.grad = torch.autograd.grad(grad_norm_loss, unet.weights)[0]
        
        opt.step()
        
        normalize_coeff = 3 / torch.sum(unet.weights.data, dim = 0)
        unet.weights.data = unet.weights.data * normalize_coeff
        
        
    avgTrainLoss1 = totalTrainLoss1 / trainSteps
    avgTrainLoss2 = totalTrainLoss2 / trainSteps
    avgTrainLoss3 = totalTrainLoss3 / trainSteps
    
    # update our training history
    H["train_loss_1"].append(avgTrainLoss1)
    H["train_loss_2"].append(avgTrainLoss2)
    H["train_loss_3"].append(avgTrainLoss3)
          
  	# print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss 1: {:.6f}, Train loss 2: {:.6f}, Train loss 3: {:.6f}".format(avgTrainLoss1, avgTrainLoss2, avgTrainLoss3))
    #print("Train loss 1: {:.6f}".format(avgTrainLoss1))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

# plot the training loss

plt.style.use("ggplot")
plt.figure(0)
plt.plot(H["train_loss_1"], label="train_loss_dice")
plt.title("Dice Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig('MI_FinalProject/output/plots/modelA_Dice')

plt.figure(1)
plt.plot(H["train_loss_2"], label="train_loss_crossEntropy")
plt.title("Cross Entropy Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig('MI_FinalProject/output/plots/modelA_CrossEntropy')

plt.figure(2)
plt.plot(H["train_loss_3"], label="train_loss_binaryCrossEntropy")
plt.title("Binary Cross Entropy Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig('MI_FinalProject/output/plots/modelA_BinaryCrossEntropy')


# serialize the model to disk
torch.save(unet, 'MI_FinalProject/output/models/model_A_Dice')