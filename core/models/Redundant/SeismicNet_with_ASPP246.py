"""
    This is our proposed Seismic Net for semantic segmentation of facies from seismic images. This Network has
    residual connections in both Encoder and Decoder. It has also long residual skip connections to retain the
    spatial locations. Primary investigations shows promising results. Need to play with the architecture and
    hyper-parameters to obtain optimal results.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, mult=1/3):
        super(ASPP, self).__init__()
        
        self.aspp1 = nn.Sequential(
                                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1,
                                              stride = 1, padding = 0, dilation = 1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU()
                                  )
        self.aspp2 = nn.Sequential(
                                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,
                                              stride = 1, padding = int(6*mult), dilation = int(6*mult)),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU()
                                  )
        self.aspp3 = nn.Sequential(
                                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,
                                              stride = 1, padding = int(12*mult), dilation = int(12*mult)),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU()
                                  )
        self.aspp4 = nn.Sequential(
                                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,
                                              stride = 1, padding = int(18*mult), dilation = int(18*mult)),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU()
                                  )
        self.global_avg_pool = nn.Sequential(
                                              nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1,
                                                        stride = 1, padding = 0, dilation = 1),
                                              nn.BatchNorm2d(out_channels),
                                              nn.PReLU()
                                            )
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels = out_channels * 5, out_channels = in_channels, kernel_size = 1, 
                                              stride = 1, padding = 0, dilation = 1),
                                    nn.BatchNorm2d(in_channels),
                                    nn.PReLU()
                                  )

    def forward(self, x):
        
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(input = x5, size = (x.shape[2], x.shape[3]), mode = 'bilinear', align_corners = True)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=2):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
        )
        self.act = nn.PReLU()

    def forward(self, x):
        residual = self.block(x)
        x = self.downsample(x)
        return self.act(x + residual)


class TransposeResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=2):
        super(TransposeResidualBlock, self).__init__()
        padding = kernel_size // 2

        self.block1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, output_padding=1)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding),
            nn.BatchNorm2d(out_channels)
        )

        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                           output_padding=1)
        self.act = nn.PReLU()

    def forward(self, x, output_size):
        residual = self.block2(self.block1(x, output_size=output_size))
        x = self.upsample(x, output_size=output_size)

        return self.act(x + residual)


class SeismicNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=6):
        super(SeismicNet, self).__init__()

        self.start = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.encode1 = ResidualBlock(64, 64)
        self.encode2 = ResidualBlock(64, 128)
        self.encode3 = ResidualBlock(128, 256)
        self.encode4 = ResidualBlock(256, 512, stride=1)
        self.encode5 = ResidualBlock(512, 1024, stride=1)

        self.middle = ASPP(1024, 256)

        self.dencode5 = TransposeResidualBlock(1024, 512, stride=1)
        self.dencode4 = TransposeResidualBlock(512, 256, stride=1)
        self.dencode3 = TransposeResidualBlock(256, 128)
        self.dencode2 = TransposeResidualBlock(128, 64)
        self.dencode1 = TransposeResidualBlock(64, 64)

        self.end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.start(x)

        # Encoder
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)

        out_middle = self.middle(x5)

        # Decoder
        out5 = self.dencode5(out_middle, x4.size()) + x4
        out4 = self.dencode4(out5, x3.size()) + x3
        out3 = self.dencode3(out4, x2.size()) + x2
        out2 = self.dencode2(out3, x1.size()) + x1
        out1 = self.dencode1(out2, x.size()) + x

        out = self.end(out1)

        return out
        
def SeismicNet_ASPP():
    
    model = SeismicNet()
    
    return model