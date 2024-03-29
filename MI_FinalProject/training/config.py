# import the necessary packages
import torch
import os

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 1
#NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 30
BATCH_SIZE = 8

# define the input image dimensions
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384

# define threshold to filter weak predictions
THRESHOLD = 0.5
