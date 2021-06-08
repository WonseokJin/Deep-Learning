import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class StackedGRU(nn.Module):
    def __init__(self,signals):
        self.rnn=nn.GRU(input_size=signals,hidden_size=100,num_layers=3,bidirectional=True,dropout=0)
        self.fc=nn.Linear(200,signals)

    def forward(self, x):
        x=x.transpose(0,1) #(batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs,_=self.rnn(x)
        out=self.fc(nn.ReLU(outs[-1]))
        return out
