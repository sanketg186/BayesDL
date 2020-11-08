import torch
import torch.nn as nn
from .model_part import *

class BayesUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,Bayes=True,dropout_prob=.5):
        super(BayesUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
         
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.aleatoric = OutConv(64, n_classes)
        if Bayes==True:
            self.dropout1 = nn.Dropout(dropout_prob)
            self.dropout2 = nn.Dropout(dropout_prob)
            self.dropout3 = nn.Dropout(dropout_prob)
            self.dropout4 = nn.Dropout(dropout_prob)
            self.dropout5 = nn.Dropout(dropout_prob)
            self.dropout6 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x3 = self.dropout1(x3)
        x4 = self.down3(x3)
        
        x4 = self.dropout2(x4)
        x5 = self.down4(x4)
        
        x5 = self.dropout3(x5)
        x = self.up1(x5, x4)
        
        x = self.dropout4(x)
        x = self.up2(x, x3)
        
        x = self.dropout5(x)
        x = self.up3(x, x2)
        
        x = self.dropout6(x)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        uncertainty = self.aleatoric(x)
        return logits,uncertainty
