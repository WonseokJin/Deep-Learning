import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
                                  nn.BatchNorm2d(16),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(24), nn.ReLU())
        self.skip_connection = nn.Sequential(nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(24), nn.ReLU())
        self.fc1 = nn.Linear(2048,100)
        self.fc2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        skip_connection=x
        x=self.layer1(x)
        x=self.layer2(x)
        x=x+self.skip_connection(skip_connection)
        x=x.reshape(x.size(0),-1)
        x=nn.functional.relu(self.fc1(x))
        x=self.dropout(x)
        x=nn.functional.softmax(self.fc2(x),dim=0)
        return x