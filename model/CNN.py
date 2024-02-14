import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_features, out_features, pool_size, hidden_dim, capsule_num):
        super().__init__()
        self.output = out_features
        self.pool_size = pool_size
        
        self.hidden_dim = hidden_dim
        self.capsule_num = capsule_num
        
        self.conv1 = nn.Conv2d(in_features, self.hidden_dim, kernel_size=(9, 9), stride=1)
        self.pool1 = nn.MaxPool2d((self.pool_size, self.pool_size),1)
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim*2, kernel_size=(9, 9), stride=(2,2))
        self.pool2 = nn.MaxPool2d((self.pool_size, self.pool_size),1)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim*2)
        
        self.conv3 = nn.Conv2d(self.hidden_dim*2, self.hidden_dim, kernel_size=(1,1), stride=1)
        self.pool3 = nn.AdaptiveMaxPool2d(1)
       
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(self.hidden_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, out_features)
        
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x