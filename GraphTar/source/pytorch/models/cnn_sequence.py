import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSequenceModel(nn.Module):

    def __init__(self, incorporate_type=False, num_filters=12, kernel_size=8, dropout_rate=0.1,
                 max_mirna_len=26, max_target_len=53, n_encoding=(1, 0, 0, 0, 0),
                 mirna_dim=5, target_dim=5):
        super().__init__()
        self.incorporate_type = incorporate_type
        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len
        self.n_encoding = n_encoding
        self.mirna_dim = mirna_dim
        self.target_dim = target_dim

        # Expected input shape: (num_batches, channels_in, seq_len)
        self.conv1_l = nn.Conv1d(in_channels=mirna_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.dropout1_l = nn.Dropout(p=dropout_rate)
        self.maxpool_l = nn.MaxPool1d(kernel_size=2)

        self.conv1_r = nn.Conv1d(in_channels=target_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.dropout1_r = nn.Dropout(p=dropout_rate)
        self.maxpool_r = nn.MaxPool1d(kernel_size=2)

        self.concatenate = torch.cat

        self.dropout3 = nn.Dropout(p=dropout_rate)

        fc_in_features = self.compute_fc_in_features(max_mirna_len, max_target_len)
        if incorporate_type:
            fc_in_features += 2
        self.fc1 = nn.Linear(in_features=fc_in_features, out_features=16)  # with extended targets

        self.dropout4 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(in_features=16, out_features=1)

    @staticmethod
    def compute_fc_in_features(mirna_len, seq_len, num_filters=12, kernel_size=8, pool_kernel_size=2):
        conv_output_len_miRNA = mirna_len - kernel_size + 1
        conv_output_len_seq = seq_len - kernel_size + 1

        pooled_len_miRNA = conv_output_len_miRNA // pool_kernel_size
        pooled_len_seq = conv_output_len_seq // pool_kernel_size

        fc_in_features = num_filters * (pooled_len_miRNA + pooled_len_seq)
        return fc_in_features

    def forward(self, mirna_input, target_input, type_input=None):
        # inputs[0].shape torch.Size([32, 26, 5])
        # inputs[1].shape torch.Size([32, 53, 5])

        x_l = F.relu(self.conv1_l(mirna_input.permute(0, 2, 1)))  # torch.Size([32, 12, 19])
        x_l = self.dropout1_l(x_l)
        x_l = self.maxpool_l(x_l)  # torch.Size([32, 12, 9])

        x_r = F.relu(self.conv1_r(target_input.permute(0, 2, 1)))  # torch.Size([32, 12, 9])
        x_r = self.dropout1_r(x_r)
        x_r = self.maxpool_r(x_r)  # torch.Size([32, 12, 23])

        x = self.concatenate([x_l, x_r], dim=2)  # torch.Size([32, 12, 32])
        x = self.dropout3(x)  # torch.Size([32, 12, 32])

        # Flatten the data for fully-connected layers, reshapes from (N, 12, 32) to (N, 384)
        N, _, _ = x.size()
        x = x.view(N, -1)

        if self.incorporate_type and type_input is not None:
            x = torch.cat((x, type_input), dim=1)

        x = F.relu(self.fc1(x))  # torch.Size([32, 32])
        x = self.dropout4(x)  # torch.Size([32, 32])
        x = self.fc2(x)  # torch.Size([32, 1])

        x = torch.sigmoid(x)

        return x
