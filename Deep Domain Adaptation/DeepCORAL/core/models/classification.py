import torch
import torch.nn as nn
from core.models import SeismicNet_Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
MODELS
'''

class SeismicNet_New(nn.Module):
    def __init__(self, num_classes):
        super(SeismicNet_New, self).__init__()

        self.shared = getattr(SeismicNet_Classifier, 'seismicnet')(n_classes = num_classes)

    def forward(self, source):
            
        out1 = self.shared(source)
        
        return out1