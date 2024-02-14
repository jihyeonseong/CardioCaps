import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class UNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = in_features
        self.n_classes = out_features
        self.bilinear = bilinear
        self.hidden_dim = hidden_dim

        self.inc = (DoubleConv(in_features, self.hidden_dim))
        self.down1 = (Down(self.hidden_dim, self.hidden_dim*2))
        self.down2 = (Down(self.hidden_dim*2, self.hidden_dim*4))
        self.down3 = (Down(self.hidden_dim*4, self.hidden_dim*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.hidden_dim*8, self.hidden_dim*16 // factor))
        self.up1 = (Up(self.hidden_dim*16, self.hidden_dim*8 // factor, bilinear))
        self.up2 = (Up(self.hidden_dim*8, self.hidden_dim*4 // factor, bilinear))
        self.up3 = (Up(self.hidden_dim*4, self.hidden_dim*2 // factor, bilinear))
        self.up4 = (Up(self.hidden_dim*2, self.hidden_dim, bilinear))
        self.outc = (OutConv(self.hidden_dim, out_features))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(out_features*256*256, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, out_features)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = F.relu(self.out(x))
        return logits