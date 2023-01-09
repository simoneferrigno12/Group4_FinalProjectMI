import torch
import torch.nn as nn
#from torchmetrics import Dice
from training import config

class MultiTaskLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MultiTaskLoss, self).__init__()
        
        
    def _dice_loss(self, inputs, targets, eps=1e-7):
        targets = targets.type(torch.int64)
        true_1_hot = torch.eye(2, device = "cuda")[targets.to("cuda").squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        
        inputs= torch.sigmoid(inputs)
        ninputs= 1-inputs
        probas = torch.cat([inputs, ninputs], dim=1)
        true_1_hot = true_1_hot.type(inputs.type())
        dims= (0,) + tuple(range(2, targets.ndimension()))
        
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection/(cardinality + eps)).mean()
        loss = (1-dice_loss)
        return loss 
    
    def forward(self, preds, mask, label, intensity):
        mask = mask.int()
        #preds[1] = preds[1].type(torch.float32)
        #preds[2] = preds[2].type(torch.float32)
        #preds[0].to("cuda")
        #preds[1].to(config.DEVICE)
        #preds[2].to(config.DEVICE)
        #preds[0] = preds[0].type(torch.int64)
        #diceLoss = Dice().to(config.DEVICE)
        crossEntropy = nn.CrossEntropyLoss()
        binaryCrossEntropy = nn.BCEWithLogitsLoss()
        label = label.long()
        intensity = intensity.unsqueeze(1)
        intensity = intensity.float()
        #loss0 = diceLoss(preds[0], mask)
        loss0 = self._dice_loss(preds[0], mask)
        loss1 = crossEntropy(preds[1], label)
        loss2 = binaryCrossEntropy(preds[2], intensity)
            
        return torch.stack([loss0, loss1, loss2])