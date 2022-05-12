import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TSforecaster(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=in_channel,
                                        out_channels=16,
                                        kernel_size=1,
                                        stride=1,
                                        padding=1),
                                    nn.Conv1d(in_channels=16,
                                        out_channels=32,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
                                    )
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=in_channel,
                                        out_channels=16,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                                    nn.Conv1d(in_channels=16,
                                        out_channels=32,
                                        kernel_size=5,
                                        stride=1,
                                        padding=1)
                                    )
        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=in_channel,
                                        out_channels=16,
                                        kernel_size=5,
                                        stride=1,
                                        padding=1),
                                    nn.Conv1d(in_channels=16,
                                        out_channels=32,
                                        kernel_size=7,
                                        stride=1,
                                        padding=1)
                                    )
        
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=50,
                            num_layers=1,
                            bias=True,
                            bidirectional=False,
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.5)

        self.dense1 = nn.Linear(50, 32)
        self.dense2 = nn.Linear(32, out_channel)

    def forward(self, x):
	# Raw x shape : (B, S, F) => (B, 10, 3)
        """
        # Shape : (B, F, S) => (B, 3, 10)
        x = x.transpose(1, 2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        x = self.conv1d_1(x)
        # Shape : (B, C, S) => (B, 32, 10)
        x = self.conv1d_2(x)
        # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        # Shape : (B, S, H) // H = hidden_size => (B, 10, 50)
        _, (hidden, _) = self.lstm(x)
        # Shape : (B, H) // -1 means the last sequence => (B, 50)
        x = hidden[-1]
        
        # Shape : (B, H) => (B, 50)
        x = self.dropout(x)
        
        # Shape : (B, 32)
        x = self.fc_layer1(x)
        # Shape : (B, O) // O = output => (B, 1)
        x = self.fc_layer2(x)
        """

         # Shape : (B, F, S) => (B, 3, 10)
        x = x.transpose(1, 2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        x1 = self.conv1d_1(x)
        if self.in_channel > 2: 
            x2 = self.conv1d_3(x)    
            x = (x1 + x3) / 2
        if self.in_channel > 4: 
            x3 = self.conv1d_5(x)    
            x = (x1 + x2 + x3) / 3
        if self.in_channel < 2: x = x1
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        
        x = self.dropout(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)

        return x
