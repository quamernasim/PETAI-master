import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from core.models import ResNet9
from core.models import SeismicNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
MODELS
'''


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        
        source_batch_size = source.shape[0]
        target_batch_size = target.shape[0]
        
        source = source.reshape((source_batch_size, -1))
        target = target.reshape((target_batch_size, -1))
        
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss



def covariance(data):

    ns = data.shape[0]

    first_term = torch.matmul(data.t(), data)

    ones = torch.ones((ns, 1)).to(device)
    temp = torch.matmul(ones.t(), data)
    temp = torch.matmul(temp.t(), temp)
    second_term = temp/ns

    covariance_matrix = torch.div((first_term - second_term), (ns - 1))

    return covariance_matrix

def CORAL(source, target):
    
    source_batch_size = source.shape[0]
    target_batch_size = target.shape[0]

    source = source.reshape((source_batch_size, -1))
    target = target.reshape((target_batch_size, -1))

    d = source.shape[1]

    covariance_source = covariance(source * 10)
    covariance_target = covariance(target)

    difference = covariance_source - covariance_target
    squared_forb_norm = torch.trace(torch.matmul(torch.conj(difference).t(), difference))

    loss = squared_forb_norm * (1/(4*d*d))
    return loss

class DeepCORAL(nn.Module):
    def __init__(self, num_classes, backbone):
        super(DeepCORAL, self).__init__()
        
        self.backbone = backbone

        if self.backbone == 'DeepLab_ResNet9':
            self.sharedNet = getattr(ResNet9, 'resnet9')(num_classes=num_classes)
            self.conv_last = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        elif self.backbone == 'SeismicNet':
            self.sharedEncoder = getattr(SeismicNet, 'seismicnet_encoder')()
            self.decoder = getattr(SeismicNet, 'seismicnet_decoder')(n_classes=num_classes)
        else:
            print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")

    def forward(self, source, target):
        
        if self.backbone in ["DeepLab_ResNet9", "deeplab_resnet9", "deeplab", "resnet9", "ResNet9", "DeepLab"]:
            source_feature_map, size = self.sharedNet(source)
            source_output = self.conv_last(source_feature_map)
            source_output = nn.Upsample(size, mode='bilinear', align_corners=True)(source_output)
    
            target_feature_map, _ = self.sharedNet(target)
            
            return (source_output, source_feature_map, target_feature_map)
            
        elif self.backbone in ["SeismicNet", "seismicnet"]:
            #out_middle_source, x, x1, x2, x3, x4, out3s, out4s, out5s = self.sharedEncoder(source)
            out_middle_source, x, x1, x2, x3, x4 = self.sharedEncoder(source)
            source_output = self.decoder(out_middle_source, x, x1, x2, x3, x4 )
            
            #out_middle_target, _, _, _, _, _, out3t, out4t, out5t = self.sharedEncoder(target)
            out_middle_target, _, _, _, _, _ = self.sharedEncoder(target)
            
            #return (source_output, out_middle_source, out3s, out4s, out5s, out_middle_target, out3t, out4t, out5t)
            return (source_output, out_middle_source, out_middle_target)

        else:
            print("Unknown Backbone, available options:[DeepLab_ResNet9, SeismicNet]")