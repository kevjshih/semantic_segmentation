import torch
import torch.nn as nn
from torch.autograd import Variable

class SimpleFCN(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._setup_model(cfg)
        
    # setup model params based on configuration

    def _add_conv(self,in_dim, out_dim):
        layers = [nn.BatchNorm2d(in_dim), nn.Conv2d(in_dim, out_dim, 3, padding=1), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)
    
    def _add_linear(self, in_dim, out_dim):
        layers = [nn.BatchNorm1d(in_dim), nn.Linear(in_dim, out_dim),  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)
    
    def _setup_model(self, model_cfg):
        self.c1 = self._add_conv(3, 64)
        self.c2 = self._add_conv(64, 128)
        self.c3 = self._add_conv(128, 256)
        self.c4 = self._add_conv(256, 512)
        self.c5 = self._add_conv(512, 459)
        #self.maxpool = nn.MaxPool2d(2, stride=2)

        
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x) 
        x = self.c3(x) 
        x = self.c4(x) 
        x = self.c5(x) 
        return x

    
