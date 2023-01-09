# import the necessary packages
from torch.utils.data import Dataset
import cv2
import pandas as pd
import torch
import os
from skimage import io
import imutils
import numpy as np

class TaskDataLoader(Dataset):
    def __init__(self, csv_file, root_dir):
        # store the image filepaths, the mask associated
        self.frame = pd.read_csv(csv_file, names=["Image", "Mask", "Label", "Intensity"])
        self.root_dir = root_dir
        
    def __len__(self):
    		# return the number of total samples contained in the dataset
    		return len(self.frame)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get the image and the associated mask, and return them
        # grab the image path from the current index
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = io.imread(img_name)
        
        image = imutils.resize(image, width=384, height=384)
        
        image2 = np.zeros((3,384,384))
        image2[0,:,:] = image
        image2[1,:,:] = image
        image2[2,:,:] = image
        image = image2
        img2=[]
        
        
        mask_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        mask = io.imread(mask_name)
        mask[mask == 255.0] = 1.0
        mask = imutils.resize(mask, width=384, height=384)
        #qui
        mask2 = np.zeros((1,384,384))
        mask2[0,:,:] = mask
        mask = mask2
        mask2 = []
        '''
        mask2 = np.zeros((3,224,224))
        mask2[0,:,:] = mask
        mask2[1,:,:] = mask
        mask2[2,:,:] = mask
        mask = mask2
        '''
        
        label = self.frame.iloc[idx, 2]
        label = label.strip()
        
        class_to_num = {"homogeneous" : 0,
                   "speckled" : 1,
                   "nucleolar" : 2,
                   "centromere" : 3,
                   "golgi" : 4,
                   "numem" : 5,
                   "mitsp" : 6}
        
        #label = one_hot[label]
        label = class_to_num[label]
        
        
        intensity = self.frame.iloc[idx, 3]
        intensity = intensity.strip()
        if intensity == "positive":
            intensity = 1
        else:
            intensity = 0
        
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)      
        label = torch.as_tensor(label, dtype=torch.int16)
        intensity = torch.as_tensor(intensity, dtype=torch.int16)
                    
		    # return a tuple of the image and its label
        return (image, mask, label, intensity)