import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_residual_block_1(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(out_channels),nn.ReLU())

def make_residual_block_2(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1),nn.Dropout(0.4),nn.BatchNorm2d(out_channels),nn.ReLU())

def oneXone_residual_connection(in_channels,out_channels):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=1)

class ResidualNet_18(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Sequential(nn.Conv2d(11,16,kernel_size=3,padding=1,stride=1),nn.BatchNorm2d(16),nn.ReLU())

        self.layer2_1=make_residual_block_1(16,32)
        self.layer2_2=make_residual_block_2(32,64)
        self.layer2_one=oneXone_residual_connection(16,64)

        self.layer3_1 = make_residual_block_1(64,64  )
        self.layer3_2 = make_residual_block_2(64,128  )
        self.layer3_one = oneXone_residual_connection(64,128 )

        self.layer4_1 = make_residual_block_1(128,128 )
        self.layer4_2 = make_residual_block_2(128,128 )

        self.layer5_1 = make_residual_block_1(128,128 )
        self.layer5_2 = make_residual_block_2(128,256 )
        self.layer5_one = oneXone_residual_connection(128,256 )

        self.layer6_1 = make_residual_block_1(256,256 )
        self.layer6_2 = make_residual_block_2(256,256 )

        self.layer7_1 = make_residual_block_1(256, 256)
        self.layer7_2 = make_residual_block_2(256,512)
        self.layer7_one = oneXone_residual_connection(256,512)

        self.layer8_1 = make_residual_block_1(512,512)
        self.layer8_2 = make_residual_block_2(512,512)

        self.fc1=nn.Linear(32768,100)
        self.fc2=nn.Linear(100,2)
        self.dropout=nn.Dropout(0.2)

    def foward(self,x):
        Y=self.layer1(x)

        S=self.layer2_one(Y)
        Y=self.layer2_1(Y)
        Y=self.layer2_2(Y)
        Y+=S

        S = self.layer3_one(Y)
        Y=self.layer3_1(Y)
        Y=self.layer3_2(Y)
        Y += S

        S=Y
        Y=self.layer4_1(Y)
        Y=self.layer4_2(Y)
        Y += S

        S = self.layer5_one(Y)
        Y=self.layer5_1(Y)
        Y=self.layer5_2(Y)
        Y += S

        S = Y
        Y=self.layer6_1(Y)
        Y=self.layer6_2(Y)
        Y += S

        S = self.layer7_one(Y)
        Y=self.layer7_1(Y)
        Y=self.layer7_2(Y)
        Y += S

        S=Y
        Y=self.layer8_1(Y)
        Y=self.layer8_2(Y)
        Y += S

        Y=Y.reshape(Y.size(0),-1)

        Y=nn.functional.relu(self.fc1(Y))
        Y=self.fc2(Y)
        Y=self.dropout(Y)
        Y=nn.functional.softmax(Y,dim=1)
