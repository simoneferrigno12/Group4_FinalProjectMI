# USAGE
# python3 predict.py

# import the necessary packages
from training import config
import numpy as np
import torch
from training.dataset import TaskDataLoader
import torchmetrics.functional as f
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import confusion_matrix


LETTER = 'E'
MODEL_PATH = 'MI_FinalProject/output/models/model_A_bce_alpha'
#MODEL_PATH = config.MODEL_PATH_A_DICE

root_dir = 'MI_FinalProject/dataset/train'
csv_file = 'MI_FinalProject/dataset/crossValidationCsv/foldA_val.csv'

if __name__ == '__main__':
    print("[INFO] loading up test image paths...")
    testData = TaskDataLoader(csv_file=csv_file, root_dir=root_dir)
    testLoader = DataLoader(testData, shuffle=False, batch_size=1, pin_memory=config.PIN_MEMORY, num_workers=2)

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")
    unet = torch.load(MODEL_PATH).to(config.DEVICE)

    dice_scores = []
    y_label = []
    y_intensity = []
    preds_label = []
    preds_intensity = []

    unet.eval()
    # turn off gradient tracking
    with torch.no_grad():
        for (x, y0, y1, y2) in testLoader:
            # send the input to the device
            x = x.to(config.DEVICE)
            y0 = y0.to(config.DEVICE)

            # make the predictions, calculate dice score and evaluate the classification for the label and the intensity
            preds = unet(x)

            mask = torch.sigmoid(preds[0])
            predMask = np.where(mask.cpu() > config.THRESHOLD, 1, 0)
            y0 = y0.type(torch.uint8)
            predMask = torch.Tensor(predMask).type(torch.uint8).to(config.DEVICE)
            #print(predMask.size())
            #print(y1.size())
            value = f.dice(predMask, y0).item()
            dice_scores += [value, ]
            y_label += [y1.cpu().item(), ]
            preds_label += [preds[1].argmax(1).to(config.DEVICE).cpu().item(), ]
            y_intensity += [y2.cpu().item(), ]
            preds_intensity += [
                torch.where(torch.sigmoid(preds[2].squeeze()) > torch.Tensor([config.THRESHOLD]).to(config.DEVICE), 1,
                            0)[0].squeeze().cpu().item(), ]
    print('Mask accuracy ', np.array(dice_scores).mean())
    d = {'dice_scores': dice_scores}
    df_tmp = pd.DataFrame(data=d)
    name = 'dice_scores' + '.csv'
    df_tmp.to_csv(MODEL_PATH + name, header=True, index=False)

    print('Label accuracy',
          len(np.array(torch.where((torch.Tensor(preds_label) == torch.Tensor(y_label)))[0])) / len(testLoader))
    print(confusion_matrix(preds_label, y_label))

    print('Intensity accuracy',
          len(np.array(torch.where((torch.Tensor(preds_intensity) == torch.Tensor(y_intensity)))[0])) / len(testLoader))
    print(confusion_matrix(preds_intensity, y_intensity))
