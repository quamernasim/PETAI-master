import torch
from torch import Tensor
import numpy as np
import sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance

def ce_unflat(inp,tar):
    inputs = torch.sigmoid(inp)
    a = inputs.type(torch.float32)
    y = tar.type(torch.float32)
    a = a.clamp(min=1e-06, max=1-1e-06)
    l = -(y*torch.log(a)+(1-y)*torch.log(1-a))
    return l.mean(2).mean(2)

class CE(nn.Module):
    def __init__(self, w = 100, it = 0):
        super(CE, self).__init__()
        self.w = w
        self.it = it
    def forward(self, inputs, targets):
        l = ce_unflat(inputs,targets)

        ls = l[:,0].mean()
        ar = l[:,1].mean()

        if self.it ==0:
            return self.w*ar
        else:
            return self.w*(ls+self.dw*ar)

def precision(inp,tar, thresh = 0.5):
    '''tp/tp+fp'''
    inp = inp[:,0,...]
    tar = tar[:,0,...]
    pc = (torch.sigmoid(inp) > thresh).float()
    tc = tar.float()
    tp = (pc*tc).sum(-1).sum(-1)
    fp = (pc-tc).clamp(min = 0).sum(-1).sum(-1)
    
    return tp/(tp+fp+1e-09)

def recall(inp,tar, thresh = 0.5):
    '''tp/tp+fn'''
    inp = inp[:,0,...]
    tar = tar[:,0,...]
    pc = (torch.sigmoid(inp) > thresh).float()
    tc = tar.float()
    tp = (pc*tc).sum(-1).sum(-1)
    fn = (tc-pc).clamp(min = 0).sum(-1).sum(-1)
    return tp/(tp+fn+1e-09)

def dice(inp,tar):
    inp = inp[:,0,...]
    tar = tar[:,0,...]
    pc = (torch.sigmoid(inp)).float()
    tc = tar.float()
    pc_sum = pc.sum(-1).sum(-1)
    tc_sum = tc.sum(-1).sum(-1)
    numer = 2*(pc*tc).sum(-1).sum(-1)
    denom = pc_sum+tc_sum
    return (numer/denom + 1e-09)
        
        
def iou(inp,tar):
    """iou for only the first mask, the apical lesions"""
    inp = inp[:,0,...]
    tar = tar[:,0,...]
    pc = (torch.sigmoid(inp)).float()
    tc = tar.float()
    pc_sum = pc.sum(-1).sum(-1)
    tc_sum = tc.sum(-1).sum(-1)
    
    intersection = (pc*tc).sum(-1).sum(-1)
    union = pc_sum+tc_sum-intersection
    return intersection/(union+1e-09)

class FocalLoss(nn.Module):
    def __init__(self, alpha = 500, gamma = 0.5, fw = Tensor([1.,0.1,0.1]), device = torch.device('cuda:0')):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = device
        self.fw = fw.to(device)       
    def forward(self, inputs, targets):
        logpt = -ce_unflat(inputs,targets)
        pt = logpt.exp()
        loss = -1 * (1-pt)**self.gamma * self.alpha * logpt
        loss = loss*self.fw

        return loss.mean()
    
def McDiceLoss(input, target, weight=None, ignore_index=255):
    smooth = 1e-5
    loss = 0.
    #print(input.size()) (32,6,99,99)
    #print(target.size())
    # have to use contiguous since they may from a torch.view op
    for c in range(6): #n_classes = 6
        iflat = input[:, c,:,: ].contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        if weight != None:
            w = weight[c]
            loss += w*(1 - ((2. * intersection + smooth) /
                             (iflat.sum() + tflat.sum() + smooth)))
        else:
            loss += (1 - ((2. * intersection + smooth) /
                             (iflat.sum() + tflat.sum() + smooth)))
            
    return loss

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
#     assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def distmap(t):
    t = t.numpy()
    outs = np.zeros_like(t)
    if len(t.shape) == 4:
        bs = t.shape[0]
        for b in range(bs):
            outs[b] = one_hot2dist(t[b])
    if len(t.shape) == 3:
        return Tensor(one_hot2dist(t))
    return Tensor(outs)

class SurfaceLoss(nn.Module):
    def __init__(self, w = 1., sw = 0.1, device = torch.device('cuda:0')):
        super(SurfaceLoss, self).__init__()
        self.w = w
        self.sw = sw
        self.device = device
    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert not one_hot(dist_maps)
        dist_maps = distmap(target.cpu())
        inputs = torch.sigmoid(inputs).cpu()
        pc = inputs.float()
        dc = dist_maps.float()

        multipled = pc*dc
        loss = multipled.mean(2).mean(2).to(self.device)

        ls = loss[:,0].mean()
        ar = loss[:,1].mean()

        loss = ls+self.sw*ar
        return self.w*loss

def surface_metric(inp,tar):
    inputs = inp[:,0,...]
    target = tar[:,0,...]
    dist_maps = distmap(target.cpu())
    inputs = torch.sigmoid(inputs).cpu()
    pc = inputs.type(torch.float32)
    dc = dist_maps.type(torch.float32)

    multipled = pc*dc
    return multipled.mean()

