import torch
import torch.nn as nn
import torchvision.models as model
import torch.nn.functional as functional
from net.MagnifyCell import BasicMagnifyCell
from net.Decoder import Decoder
from net.Encoder import Encoder
    

class MNet(nn.Module):
    def __init__(self, pretrain=True, IF_EN_CELL=False, IF_DE_CELL=False, IF_BN=True, **kwargs):
        super(MNet, self).__init__()
        self.encoder = Encoder(pretrain, IF_EN_CELL, IF_BN, **kwargs)
        self.decoder = Decoder(IF_DE_CELL, IF_BN, **kwargs)
        self.stage = kwargs['stage']
        if self.stage == 'shape':
            self.output = torch.nn.Sigmoid()
        else:
            self.output = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        B5_C3, B4_C3, B3_C3, B2_C2 = self.encoder(x)
        output = self.decoder(B5_C3, B4_C3, B3_C3, B2_C2)
        output = self.output(output)
        return output