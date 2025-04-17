import torch
import torch.nn as nn
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, dim_model, max_wavelength=5000):
        super().__init__()
        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        scale = max_wavelength / (2 * np.pi)

        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, x):
        pos = torch.arange(x.shape[1], dtype=self.sin_term.dtype, device=x.device)
        sin_in = pos.unsqueeze(-1) / self.sin_term
        cos_in = pos.unsqueeze(-1) / self.cos_term

        sin_pos = torch.sin(sin_in)
        cos_pos = torch.cos(cos_in)
        encoded = torch.cat([sin_pos, cos_pos], dim=-1)
        return encoded.unsqueeze(0) + x


class EmbeddingTransform(nn.Module):
    def __init__(self, nin, nout, dropout=0.5, nhead=1, num_layers=6):
        super(EmbeddingTransform, self).__init__()
        self.embedding = nn.Embedding(5, nin, padding_idx=0)
        self.position_embedding = PositionalEncoder(nin)

        layer = nn.TransformerEncoderLayer(
            d_model=nin, nhead=nhead, dim_feedforward=nin * 2, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.transform = nn.Linear(nin, nout)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : [batch_size, n_sequence]
        """
        mask = x == 0
        # x -> [batch_size, n_sequence, nin]
        x = self.embedding(x.long())
        # x -> [batch_size, n_sequence, nin]
        x = self.position_embedding(x)
        # x -> [batch_size, n_sequence, nin]
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.activation(self.transform(x))
        return self.dropout(x)


class ContactCNN(nn.Module):
    def __init__(self, ks=9, projection_dim=256):
        super(ContactCNN, self).__init__()

        self.layers = nn.Sequential(
            self._conv_bn_act(2 * projection_dim, projection_dim, ks),
            self._conv_bn_act(projection_dim, projection_dim // 2, ks),
            self._conv_bn_act(projection_dim // 2, projection_dim // 4, ks),
            self._conv_bn_act(projection_dim // 4, 1, ks, activation=nn.Sigmoid())
            # self._conv_bn_act(2 * projection_dim, 1, ks, activation=nn.Sigmoid()),
        )

    @staticmethod
    def _conv_bn_act(in_channels, out_channels, kernel_size, activation=nn.ReLU()):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def clip(self):
        self.layers[1][0].weight.data[:] = 0.5 * (self.layers[1][0].weight + self.layers[1][0].weight.transpose(2, 3))

    def forward(self, z0, z1):
        """
        z0 : [batch_size, mirna_seq_len, projection_dim]
        z1 : [batch_size, target_seq_len, projection_dim]
        """
        z0, z1 = z0.transpose(1, 2), z1.transpose(1, 2)
        # z_dif, z_mul : [batch_size, projection_dim, mirna_seq_len, target_seq_len]
        z_dif, z_mul = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2)), z0.unsqueeze(3) * z1.unsqueeze(2)
        # z_cat : [batch_size, 2 * projection_dim, mirna_seq_len, target_seq_len]
        z_cat = torch.cat([z_dif, z_mul], 1)
        # -> [batch_size, 1, mirna_seq_len, target_seq_len]
        return self.layers(z_cat)


class LogisticActivation(nn.Module):
    def __init__(self, x0: float = 0.5, k: int = 1):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k_eval = k

    def forward(self, x):
        k = self.k if self.training else self.k_eval
        return torch.clamp(1 / (1 + torch.exp(-k * (x - self.x0))), min=0, max=1)


class TECMiTarget(nn.Module):
    def __init__(self, input_dim=512, projection_dim=256, n_heads=1, n_layers=6, dropout=0, kernal_size=9,
                 p0=0.5, gamma_init=0, max_mirna_len=26, max_target_len=53):
        super(TECMiTarget, self).__init__()
        self.embedding_transform = EmbeddingTransform(
            nin=input_dim, nout=projection_dim, nhead=n_heads, num_layers=n_layers, dropout=dropout
        )
        self.contact = ContactCNN(kernal_size, projection_dim)
        self.activation = LogisticActivation(x0=p0, k=20)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.clip()

        # benchmark framework specific parameters
        self.max_mirna_len, self.max_target_len = max_mirna_len, max_target_len
        self.n_encoding = 4
        self.target_dim = self.mirna_dim = 1

    def clip(self):
        self.contact.clip()
        self.gamma.data.clamp_(min=0)

    def forward(self, mirna_input, target_input):
        """
        z0 : [batch_size, mirna_seq_len]
        z1 : [batch_size, target_seq_len]
        """
        # z0, z1 : [batch_size, mirna_seq_len, projection_dim], [batch_size, target_seq_len, projection_dim]
        mirna_input, target_input = self.embedding_transform(mirna_input), self.embedding_transform(target_input)
        # yhat : [batch_size, 1, mirna_seq_len, target_seq_len]
        yhat = self.contact(mirna_input, target_input)

        # mu = torch.mean(yhat, dim=(1, 2, 3), keepdim=True).repeat(1, 1, yhat.shape[2], yhat.shape[3])
        # sigma = torch.var(yhat, dim=(1, 2, 3), keepdim=True).repeat(1, 1, yhat.shape[2], yhat.shape[3])
        mu = torch.mean(yhat, dim=(1, 2, 3), keepdim=True)
        sigma = torch.var(yhat, dim=(1, 2, 3), keepdim=True)

        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q, dim=(1, 2, 3)) / (torch.sum(torch.sign(Q), dim=(1, 2, 3)) + 1)
        phat = self.activation(phat)

        return phat.unsqueeze(dim=1)
