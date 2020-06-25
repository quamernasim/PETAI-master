import torch
import torch.nn as nn
from core.models import SeismicNet_Classifier_diff_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
MODELS
'''

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

    d = source.shape[1]

    covariance_source = covariance(source)
    covariance_target = covariance(target)

    difference = covariance_source - covariance_target
    squared_forb_norm = torch.trace(torch.matmul(torch.conj(difference).t(), difference))

    loss = squared_forb_norm * (1/(4*d*d)) * 1e4
    return loss

class DeepCORAL(nn.Module):
    def __init__(self, num_classes):
        super(DeepCORAL, self).__init__()

        self.shared = getattr(SeismicNet_Classifier_diff_layers, 'seismicnet')(n_classes = num_classes)

    def forward(self, source, target):
            
        fc_last_source, fc3_source, fc2_source, fc1_source = self.shared(source)
        fc_last_target, fc3_target, fc2_target, fc1_target = self.shared(target)
        
        return (fc_last_source, fc3_source, fc2_source, fc1_source , fc_last_target, fc3_target, fc2_target, fc1_target)