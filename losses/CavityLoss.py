import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from skimage import measure
from scipy import ndimage
import MBD

class Dilation(nn.Module):
    def __init__(self,):
        super().__init__()
        self.dilate = nn.Conv3d(1, 1, kernel_size=5, padding=2, bias=False)
        self.dilate.weight.data.fill_(1)
        
    def forward(self, x):
        return self.dilate(x)


def cavity_loss(input, target):
    EPM_s = input.squeeze()
    label = target.squeeze()
    # EPM_s = torch.sigmoid(EPM)
    EPM_s_clone = EPM_s.clone()
    EPM_s_clone = (EPM_s_clone*255).detach().cpu().numpy().astype(np.uint8)
    EPM_s_clone  = (EPM_s_clone > 0.5 *255).astype(np.uint8)
    label_clone = label.clone()
    # label_clone = (label_clone).detach().cpu().numpy().astype(np.uint8)
    # kernel = np.ones((5,5,5),np.uint8)
    # label_dl = cv2.dilate(label_clone,kernel,iterations = 1)
    dilate = Dilation()
    dilate.eval()
    label_dl = dilate(label_clone.unsqueeze(1)).squeeze().detach().cpu().numpy().astype(np.uint8)
    EPM_s_clone = EPM_s_clone*label_dl
    diff = np.array(label_clone, dtype = float) - np.array(EPM_s_clone, dtype = float) 
    diff = (diff > 0).astype(np.uint8)
    img = (diff + EPM_s_clone) < 0.5
    img = img.astype(np.uint8)
    # run connected components on the predicted mask; consider only 1-connectivity
    seg = measure.label(img, background=0, connectivity=1)
    critical = MBD.topology_error_3D(np.array(seg, dtype = np.int32), diff)
    critical = critical.astype(np.uint8)
    critical = torch.from_numpy(np.array([critical])).cuda()
    EPM_contour = (EPM_s.squeeze() * critical.squeeze()).unsqueeze(axis=0)
    label_contour = (label * critical.squeeze()).unsqueeze(axis=0)
    loss = nn.BCELoss(reduction='mean')(EPM_contour, label_contour)
    return loss

class CavityLoss(nn.Module):
    # TODO: Check about partition_weights, see original code
    # what i didn't understand is that for dice loss, partition_weights gets
    # multiplied inside the forward and also in the factory_loss function
    # I think that this is wrong, and removed it from the forward
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return cavity_loss(pred, gt)

if __name__=="__main__":
    pred = torch.zeros((2, 1,80,80,80))
    pred[0,0,40:60, 40:60,40:60]=1
    pred[1,0,40:60, 40:60,40:60]=1
    gt = torch.zeros((2, 1,80,80,80))
    gt[0,0,40:50, 40:50,40:50]=1
    gt[1,0,40:60, 40:60,40:60]=1
    Closs = CavityLoss()
    cavity_loss = Closs(pred, gt)
    print(cavity_loss)
