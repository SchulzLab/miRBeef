import torch
import torch.nn as nn
import torch.nn.functional as F


class DMISO(nn.Module):

    def __init__(self, max_mirna_len=30, max_target_len=60, n_encoding=(0.25, 0.25, 0.25, 0.25), l1_lambda=0.01, mirna_dim=5, target_dim=5):
        super(DMISO, self).__init__()
        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len
        self.l1_lambda = l1_lambda
        self.n_encoding = n_encoding
        in_channels = len(n_encoding)
        self.mirna_dim = mirna_dim
        self.target_dim = target_dim

        # miRNA branch input (4, 30) -> (10, 23) -> (10, 20)
        self.conv_mirna = nn.Conv1d(in_channels=in_channels, out_channels=10, kernel_size=8, stride=1, padding=0)
        self.maxpool_mirna = nn.MaxPool1d(kernel_size=4, stride=1)

        # target branch input (4, 60) -> (10, 53) -> (10, 50)
        self.conv_target = nn.Conv1d(in_channels=in_channels, out_channels=10, kernel_size=8, stride=1, padding=0)
        self.maxpool_target = nn.MaxPool1d(kernel_size=4, stride=1)

        # concatenation -> (10, 70)
        n_features = max_mirna_len - 8 + 1 - 4 + 1 + max_target_len - 8 + 1 - 4 + 1

        # batch norm along features dimension drop out layer after merge two branches -> (70, 10)
        self.bn_merge = nn.BatchNorm1d(10, eps=1e-3, momentum=0.01)
        self.dropout_merge = nn.Dropout(0.25)

        # bidirectional LSTM layer -> (70, 20)
        self.bilstm = nn.LSTM(input_size=10, hidden_size=10, num_layers=1, batch_first=True, bidirectional=True)

        # batch norm and drop out after bilstm -> (70, 20)
        self.bn_bilstm = nn.BatchNorm1d(20, eps=1e-3, momentum=0.01)
        self.dropout_bilstm = nn.Dropout(0.5)

        # flatten the data for fully-connected layers -> (1, 1400)
        self.flatten = nn.Flatten()

        # fully-connected layers -> 100
        self.dense_fc = nn.Linear(in_features=2 * 10 * n_features, out_features=100)

        # batch norm and drop out after fully-connected layer -> 100
        self.bn_fc = nn.BatchNorm1d(100, eps=1e-3, momentum=0.01)
        self.dropout_fc = nn.Dropout(0.5)

        # logistic regression layer -> 1
        self.dense_logistic = nn.Linear(in_features=100, out_features=1)
        self.bn_logistic = nn.BatchNorm1d(1, eps=1e-3, momentum=0.01)

    def forward(self, mirna_input, target_input):
        # miRNA branch
        x_mirna = F.relu(self.conv_mirna(mirna_input.permute(0, 2, 1)))
        x_mirna = self.maxpool_mirna(x_mirna)

        # target branch
        x_target = F.relu(self.conv_target(target_input.permute(0, 2, 1)))
        x_target = self.maxpool_target(x_target)

        # concatenate
        x = torch.cat((x_mirna, x_target), dim=2)
        x = self.bn_merge(x)
        x = self.dropout_merge(x)

        # bidirectional LSTM
        x, _ = self.bilstm(x.permute(0, 2, 1))
        x = self.bn_bilstm(x.permute(0, 2, 1))
        x = self.dropout_bilstm(x)

        # flatten the data for fully-connected layers
        x = self.flatten(x)

        # fully-connected layers
        x = self.dense_fc(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout_fc(x)

        # logistic regression layer
        x = self.dense_logistic(x)
        x = self.bn_logistic(x)
        x = torch.sigmoid(x)

        return x
