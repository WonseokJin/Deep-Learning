import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(100,256),nn.ReLU(),
                                   nn.Linear(256,512),nn.ReLU()
                                   ,nn.Linear(512,1024),nn.ReLU())
        self.layer2=nn.Linear(1024,2)
        self.dropout=nn.Dropout(0.2)
    def forward(self, x):
        x=self.layer
        x=self.dropout(x)
        x=nn.functional.softmax(self.layer2(x))
        return x
