class TSforecaster(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(TSforecaster, self).__init__()
        self.in_channel = in_channel
       
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=in_channel,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)                            
                                )
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)                            
                                )

        self.lstm1 = nn.LSTM(input_size=128,
                            hidden_size=256,
                            num_layers=3,
                            bias=True,
                            bidirectional=True,
                            batch_first=True)
        # self.dense1 = nn.Linear(in_channel, 2*in_channel)
        self.lstm2 = nn.LSTM(input_size=256,
                            hidden_size=128,
                            num_layers=3,
                            bias=True,
                            bidirectional=True,
                            batch_first=True)

        self.dropout = nn.Dropout(0.5)

        self.dense2 = nn.Linear(64, 128)

        self.dense3 = nn.Linear(128, out_channel)

    def forward(self, x):
	    # Raw x shape : (B, S, F) => (B, 10, 3)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        # x = np.array(list_x) / len(list_x)
        x = x.transpose(1, 2)
        # x = x.permute(1,2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
       
        self.lstm1.flatten_parameters()
        _, (hidden, _) = self.lstm1(x)
        x = hidden[-1]
        # x = x.transpose(1, 2)
        # x = self.dense1(x)
        x = x.unsqueeze(1)
        self.lstm2.flatten_parameters()
        _, (hidden, _) = self.lstm2(x)
        x = hidden[-1]
        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x
