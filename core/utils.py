import numpy as np
import torch 
import torchvision.utils as vutils


def np_to_tb(array):
    # if 2D :
    if array.ndim == 2:
        # HW => CHW
        array = np.expand_dims(array,axis=0)
        # CHW => NCHW
        array = np.expand_dims(array,axis=0)
    elif array.ndim == 3:
        # HWC => CHW
        array = array.transpose(2, 0, 1)
        # CHW => NCHW
        array = np.expand_dims(array,axis=0)
    
    array = torch.from_numpy(array)
    array = vutils.make_grid(array, normalize=True, scale_each=True)
    return array
class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)

