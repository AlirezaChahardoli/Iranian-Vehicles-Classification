import torch
from torch import nn
"""
custom CNN model
"""


class CustomBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride=1): 
        super(CustomBlock, self).__init__() 
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) 
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1) 
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.conv3 = nn.Conv2d(out_channels, out_channels , kernel_size=1) 
        self.bn3 = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU()
        self.identity_conv=nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=stride, padding=1)
    def forward(self, x):
        identity = x
        #print(identity.shape)
        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.conv2(x) 
        x = self.bn2(x) 
        x = self.relu(x) 
        x = self.conv3(x) 
        x = self.bn3(x) 
        #print(x.shape)
        if self.identity_conv is not None: 
            identity = self.identity_conv(identity) 
        if x.shape != identity.shape:
            identity=nn.functional.interpolate(identity,size=(x.shape[2],x.shape[3]),mode='nearest')
            
        x += identity 
        x = self.relu(x) 
        return x 
            
            
class SimpleResNet(nn.Module): 
    def __init__(self, num_classes=13): 
        super(SimpleResNet, self).__init__() 
        self.conv1 = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(16) 
        self.relu = nn.ReLU() 
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1) 
        self.block1 = CustomBlock(16, 32)
        self.block2 = CustomBlock(32,64) 
        self.block3 = CustomBlock(64,128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128 ,128)
        self.fc2=nn.Linear(128,256)
        self.drop=nn.Dropout(p=0.5)
        self.fc3=nn.Linear(256,num_classes)
        
        
    def forward(self, x): 
        x = self.conv1(x) 
        
        x = self.bn1(x) 
        x = self.relu(x) 
        x = self.maxpool(x)
        x = self.block1(x) 
        x = self.block2(x) 
        x = self.block3(x) 
        x = self.avgpool(x) 
        x = self.flatten(x) 
        x = self.fc(x)
        x=self.fc2(x)
        x=self.drop(x)
        x=self.fc3(x)
        return x
