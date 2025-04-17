import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSequenceModel(nn.Module):

    def __init__(self, num_filters=12, kernel_size=8, dropout_rate=0.1, rnn_units=32):
        super().__init__()

        self.hidden_size = rnn_units
        self.num_layers = 1  # Of the LSTM/GRU

        # Expected input shape: (num_batches, channels_in, seq_len)
        self.conv1_l = nn.Conv1d(in_channels=5, out_channels=num_filters, kernel_size=kernel_size)

        self.dropout1_l = nn.Dropout(p=dropout_rate)
        self.maxpool_l = nn.MaxPool1d(kernel_size=2)

        self.conv1_r = nn.Conv1d(in_channels=5, out_channels=num_filters, kernel_size=kernel_size)
        self.dropout1_r = nn.Dropout(p=dropout_rate)
        self.maxpool_r = nn.MaxPool1d(kernel_size=2)

        self.concatenate = torch.cat

        # Expected input shape: (seq_len, num_batches, input_size) 
        # seq_len = 23 + 9 = 32, num_batches = 32, input_size = num_filters * len(kernel_sizes), 
        # if only one conv layer and kernel size, then len = 1
        self.bilstm = nn.GRU(input_size=num_filters, hidden_size=rnn_units, num_layers=1,
                             batch_first=True, bidirectional=True)
        # Expected out shape: (batch_size, seq_len, num_directions * hidden_size)
        self.dropout3 = nn.Dropout(p=dropout_rate)

        # self.fc1 = nn.Linear(in_features=32, out_features=16)
        self.fc1 = nn.Linear(in_features=64, out_features=32)

        self.dropout4 = nn.Dropout(p=dropout_rate)

        # self.fc2 = nn.Linear(in_features=16, out_features=1)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, inputs):
        # print("input shape left: ", inputs[0].shape) # torch.Size([32, 26, 5])
        # print("input shape right: ", inputs[1].shape) # torch.Size([32, 53, 5])

        x_l = F.relu(self.conv1_l(inputs[0].permute(0, 2, 1)))
        # print("conv1d left: ", x_l.shape) # torch.Size([32, 12, 19])

        x_l = self.dropout1_l(x_l)
        x_l = self.maxpool_l(x_l)

        x_r = F.relu(self.conv1_r(inputs[1].permute(0, 2, 1)))
        # print("conv1d right: ", x_l.shape) # torch.Size([32, 12, 9])

        x_r = self.dropout1_r(x_r)
        x_r = self.maxpool_r(x_r)

        # print("concatenate in left: ", x_l.shape) # torch.Size([32, 12, 9])
        # print("concatenate in right: ", x_r.shape) # torch.Size([32, 12, 23])

        x = self.concatenate([x_l, x_r], dim=2)
        # print("concatenate out: ", x.shape) # torch.Size([32, 12, 32])

        lstm_out, lstm_hidden = self.bilstm(x.permute(0, 2, 1))
        lstm_out = F.relu(lstm_out[:, -1, :])

        # print("lstm out: ", lstm_out.shape) # torch.Size([32, 32, 64])

        # print("lstm out: ", lstm_out.shape) # torch.Size([32, 32, 64])
        # print("lstm hidden: ", lstm_hidden.shape) # torch.Size([32, 64])

        x = self.dropout3(lstm_out)
        # print("dropout out: ", x.shape) # torch.Size([32, 64])

        x = F.relu(self.fc1(x))
        # print("fc1 out: ", x.shape) # torch.Size([32, 32])

        x = self.dropout4(x)
        # print("dropout out: ", x.shape) # torch.Size([32, 32])

        # No need to apply sigmoid here - it is built-in the loss function BCEWithLogitsLoss()
        x = self.fc2(x)
        # print("fc2 out: ", x.shape) # torch.Size([32, 1])
        return x


class CNNTransformerModel(nn.Module):

    def __init__(self, num_filters=12, kernel_size=8, dropout_rate=0.1, hidden_size=512, num_heads=8):
        super(CNNTransformerModel, self).__init__()

        self.conv1_l = nn.Conv1d(in_channels=5, out_channels=num_filters, kernel_size=kernel_size)

        self.dropout1_l = nn.Dropout(p=dropout_rate)
        self.maxpool_l = nn.MaxPool1d(kernel_size=2)

        self.conv1_r = nn.Conv1d(in_channels=5, out_channels=num_filters, kernel_size=kernel_size)
        self.dropout1_r = nn.Dropout(p=dropout_rate)
        self.maxpool_r = nn.MaxPool1d(kernel_size=2)

        # Transformer layer
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_size * 2, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=1)

        # Output classification layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x_l = F.relu(self.conv1_l(inputs[0].permute(0, 2, 1)))

        x_l = self.dropout1_l(x_l)
        x_l = self.maxpool_l(x_l)

        x_r = F.relu(self.conv1_r(inputs[1].permute(0, 2, 1)))
        # print("conv1d right: ", x_l.shape) # torch.Size([32, 12, 9])

        x_r = self.dropout1_r(x_r)
        x_r = self.maxpool_r(x_r)

        x = torch.cat((x_l, x_r), dim=2)
        # print("concatenate out: ", x.shape)

        x = x.permute(0, 2, 1)
        # print("permute out: ", x.shape)

        x = self.transformer_encoder(x)
        # print("transformer out: ", x.shape)

        x = torch.mean(x, dim=1)
        # print("mean out: ", x.shape)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
