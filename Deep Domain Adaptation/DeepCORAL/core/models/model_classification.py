import torch
import torch.nn as nn
from core.models import SeismicNet_Classifier

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

        self.shared = getattr(SeismicNet_Classifier, 'seismicnet')(n_classes = num_classes)

    def forward(self, source, target):
            
        out1 = self.shared(source)
        out2 = self.shared(target)
        
        return (out1, out2)