import torch
import torch.nn as nn


class MitarNet(nn.Module):
    def __init__(self, n_embeddings: int = 5, max_mirna_len: int = 26, max_target_len: int = 40):
        super().__init__()
        kernel_size = 12
        n_filters = 320
        lstm_hidden_size = 32

        self.n_encoding = (1, 0, 0, 0, 0)
        self.mirna_dim = self.target_dim = n_embeddings
        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len
        input_size = max_mirna_len + max_target_len

        self.embedding_dim = 5
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * n_embeddings, input_size * n_embeddings),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(self.embedding_dim, n_filters, kernel_size),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.birnn = nn.LSTM(
            n_filters, lstm_hidden_size, bidirectional=True, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear((input_size - (kernel_size - 1)) // 2 * 2 * lstm_hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, mirna_input, target_input):
        x = torch.cat((mirna_input, target_input), dim=1)  # (batch_size, 66, 5)
        x = self.embedding(x)  # (batch_size, 330)
        x = torch.reshape(
            x, (x.shape[0], self.embedding_dim, x.shape[1] // self.embedding_dim)
        )  # (batch_size, 5, 66)
        x = self.conv(x)  # (batch_size, 320, 27)
        x = self.birnn(x.permute(0, 2, 1))[0]  # (batch_size, 27, 64)
        return self.classifier(x)