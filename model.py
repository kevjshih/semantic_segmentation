import torch
import torch.nn as nn
from torch.autograd import Variable

class SimpleFCN(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.setup_model(model_cfg)
        

    # setup model params based on configuration

    def _add_conv(in_dim, out_dim):
        layers = [nn.BatchNorm2d(in_dim), nn.conv2d(in_dim, out_dim, 3), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)
    
    def _add_linear(in_dim, out_dim):
        layers = [nn.BatchNorm1d(in_dim), nn.Linear(in_dim, out_dim),  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)
    
    def _setup_model(self, model_cfg):
        self.c1 = self._add_conv(3, 64)
        self.c2 = self._add_conv(64, 128)
        self.c3 = self._add_conv(128, 256)
        self.c4 = self._add_conv(256, 512)
        self.c5 = self._add_conv(256, 459)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        
    def forward(self, x):
        x = self.c1(x)# Nx224x224xD
        x = self.maxpool(x)
        x = self.c2(x)#Nx112x112xD
        x = self.maxpool(x)
        x = self.c3(x)#Nx56x56xD
        x = self.maxpool(x)
        x = self.c4(x)#Nx28x28xD
        x = self.c5(x)#Nx28x28x459
        return x

    
