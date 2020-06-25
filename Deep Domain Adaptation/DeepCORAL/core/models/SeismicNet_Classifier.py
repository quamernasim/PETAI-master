"""
    This is our proposed Seismic Net for semantic segmentation of facies from seismic images. This Network has
    residual connections in both Encoder and Decoder. It has also long residual skip connections to retain the
    spatial locations. Primary investigations shows promising results. Need to play with the architecture and
    hyper-parameters to obtain optimal results.
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2),
        )
        self.act = nn.PReLU()

    def forward(self, x):
        residual = self.block(x)
        x = self.downsample(x)
        return self.act(x + residual)


class SeismicNet(nn.Module):
    def __init__(self, n_classes, in_channels=1):
        super(SeismicNet, self).__init__()

        self.start = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.encode1 = ResidualBlock(64, 64)
        self.encode2 = ResidualBlock(64, 128)
        self.encode3 = ResidualBlock(128, 256)
        self.encode4 = ResidualBlock(256, 512)
        self.encode5 = ResidualBlock(512, 1024)

        self.middle = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.PReLU()
        )
		    
        self.fc1 = nn.Linear(1024 * 2 * 2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc_last = nn.Linear(512, n_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.start(x)

        # Encoder
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)

        out_middle = self.middle(x5)

        out = out_middle.view(out_middle.size(0), out_middle.size(1) * out_middle.size(2) * out_middle.size(3))


        fc1 = self.fc1(out)
        fc1 = self.act(fc1)
        fc2 = self.fc2(fc1)
        fc2 = self.act(fc2)
        fc3 = self.fc3(fc2)
        fc3 = self.act(fc3)
        fc_last = self.fc_last(fc3)

        return fc_last

def seismicnet(n_classes):
    
    model = SeismicNet(n_classes = n_classes)
    
    return model