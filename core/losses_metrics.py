import torch
from torch import Tensor
import numpy as np
import sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from torch.autograd import Variable

def dice_loss(true, logits, eps=1e-7, ignore_index = 255):
    """Computes the SÃ¸rensenâ€“Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the SÃ¸rensenâ€“Dice loss.
    """
    true = true * (true != ignore_index).long()
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def jaccard_loss(true, logits, eps=1e-7, ignore_index = 255):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    true = true * (true != ignore_index).long()
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


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

#class FocalLoss(nn.Module):
#    def __init__(self, alpha = 500, gamma = 0.5, fw = Tensor([1.,0.1,0.1]), device = torch.device('cuda:0')):
#        super(FocalLoss,self).__init__()
#        self.gamma = gamma
#        self.alpha = alpha
#        self.device = device
#        self.fw = fw.to(device)       
#    def forward(self, inputs, targets):
#        logpt = -ce_unflat(inputs,targets)
#        pt = logpt.exp()
#        loss = -1 * (1-pt)**self.gamma * self.alpha * logpt
#        loss = loss*self.fw
#
#        return loss.mean()
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
    
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
#-------------------------------------------------------------LovaszSoftmax Loss Function---------------------------------------------------------------
# copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax/lovasz_loss.py

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='sum'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets, ignore_index = 255):
        targets = targets * (targets != ignore_index).long()
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses