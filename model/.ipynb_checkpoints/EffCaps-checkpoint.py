from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(input, eps=10e-21):
    n = torch.norm(input, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (input / (n + eps))


def length(input):
    return torch.sqrt(torch.sum(input**2, dim=-1) + 1e-8)


def mask(input):
    if type(input) is list:
        input, mask = input
    else:
        x = torch.sqrt(torch.sum(input**2, dim=-1))
        mask = F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).float()

    masked = input * mask.unsqueeze(-1)
    return masked.view(input.shape[0], -1)


class PrimaryCapsLayer(nn.Module):
    """Create a primary capsule layer where the properties of each capsule are extracted
    using a 2D depthwise convolution.

    Args:
        in_channels (int): depthwise convolution's number of features
        kernel_size (int): depthwise convolution's kernel dimension
        num_capsules (int): number of primary capsules
        dim_capsules (int): primary capsule dimension
        stride (int, optional): depthwise convolution's strides. Defaults to 1.
    """

    def __init__(self, in_channels, kernel_size, num_capsules, dim_capsules, stride=1):
        super(PrimaryCapsLayer, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding="valid",
        )
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def forward(self, input):
        output = self.depthwise_conv(input)
        output = output.view(output.size(0), self.num_capsules, self.dim_capsules)
        return squash(output)


class RoutingLayer(nn.Module):
    """Self-attention routing layer using a fully-connected network, to create a parent
    layer of capsules.

    Args:
        num_capsules (int): number of primary capsules
        dim_capsules (int): primary capsule dimension
    """

    def __init__(self, num_capsules, dim_capsules):
        super(RoutingLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_capsules, 211600, 8, dim_capsules))
        self.b = nn.Parameter(torch.zeros(num_capsules, 211600, 1))
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, input):
        u = torch.einsum(
            "...ji,kjiz->...kjz", input, self.W
        )  # u shape = (None, num_capsules, height*width*16, dim_capsules)
        c = torch.einsum("...ij,...kj->...i", u, u)[
            ..., None
        ]  # b shape = (None, num_capsules, height*width*16, 1) -> (None, j, i, 1)
        c = c / torch.sqrt(
            torch.Tensor([self.dim_capsules]).type(torch.cuda.FloatTensor)
        )
        c = torch.softmax(c, axis=1)
        c = c + self.b
        s = torch.sum(
            torch.mul(u, c), dim=-2
        )  # s shape = (None, num_capsules, dim_capsules)
        return squash(s)


class EfficientCapsNet(nn.Module):
    """Efficient-CapsNet architecture implementation.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_features, out_features):
        super(EfficientCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features, out_channels=32, kernel_size=5, padding="valid"
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding="valid")
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding="valid")
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding="valid")
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.primary_caps = PrimaryCapsLayer(
            in_channels=128, kernel_size=9, num_capsules=211600, dim_capsules=8
        )
        self.digit_caps = RoutingLayer(num_capsules=out_features, dim_capsules=16)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = torch.relu(self.batch_norm4(self.conv4(x)))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        probs = length(x)
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3*256*256)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = mask(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, 3*256*256)


class EfficientCapsNetWithReconstruction(nn.Module):
    def __init__(self, efficient_capsnet, reconstruction_net):
        super(EfficientCapsNetWithReconstruction, self).__init__()
        self.efficient_capsnet = efficient_capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x):
        x, probs = self.efficient_capsnet(x)
        reconstruction = self.reconstruction_net(x)
        return probs, reconstruction


class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true, size_average=True):
        # y_pred shape is [16,10], while y_true is [16]
        t = torch.zeros(y_pred.size()).long()
        if y_true.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, y_true.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets * torch.pow(
            torch.clamp(self.m_pos - y_pred, min=0.0), 2
        ) + self.lambda_ * (1 - targets) * torch.pow(
            torch.clamp(y_pred - self.m_neg, min=0.0), 2
        )
        return losses.mean() if size_average else losses.sum()
