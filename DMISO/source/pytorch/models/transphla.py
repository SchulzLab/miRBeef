import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.scaled_dot_product_attention(Q, K, V, attn_mask, self.d_k)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.layer_norm(output + residual), attn

    @staticmethod
    def scaled_dot_product_attention(Q, K, V, attn_mask, d_k):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e4)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, dec_self_attn_mask):  # dec_inputs = enc_outputs
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, tgt_len):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),

            # output layer
            nn.Linear(64, 1)
        )

    def forward(self, dec_outputs):
        """
        dec_outputs: [batch_size, tgt_len, d_model]
        """
        return self.projection(dec_outputs.view(dec_outputs.size(0), -1))  # [batch_size, tgt_len, tgt_vocab_size]


class TransPHLA(nn.Module):
    def __init__(self, n_layers=1, d_model=64, n_heads=9, d_k=64, d_v=64, d_ff=512, dropout_rate=0.1,
                 max_mirna_len=26, max_target_len=53, vocab_size=5):
        super(TransPHLA, self).__init__()
        # transPHLA specific parameters
        self.n_layers = n_layers
        self.d_model = d_model
        # benchmark framework specific parameters
        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len
        self.tgt_len = max_mirna_len + max_target_len
        self.n_encoding = 4
        self.target_dim = self.mirna_dim = 1

        # Embedding Block
        self.mirna_embedding = nn.Embedding(vocab_size, d_model)
        self.target_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate)
        # Encoder Block
        self.mirna_encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.target_encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        # Projection Layer
        self.projection = ProjectionLayer(d_model, self.tgt_len)

    def forward(self, mirna_input, target_input):
        """
        mirna_input: [batch_size, max_mirna_len]
        target_input: [batch_size, max_target_len]
        """
        # mirna
        # -> [batch_size, max_mirna_len, d_model]
        mirna_outputs = self.mirna_embedding(mirna_input)
        # -> [batch_size, max_mirna_len, d_model]
        mirna_outputs = self.positional_encoding(mirna_outputs.transpose(0, 1)).transpose(0, 1)
        # -> [batch_size, max_mirna_len, max_mirna_len]
        mirna_mask = self.get_attn_pad_mask(mirna_input, mirna_input).to(mirna_input.device)
        # target
        # -> [batch_size, max_target_len, d_model]
        target_outputs = self.target_embedding(target_input)
        # -> [batch_size, max_target_len, d_model]
        target_outputs = self.positional_encoding(target_outputs.transpose(0, 1)).transpose(0, 1)
        # -> [batch_size, max_target_len, max_target_len]
        target_mask = self.get_attn_pad_mask(target_input, target_input).to(target_input.device)
        # encoder
        mirna_attns, target_attns = [], []
        for i in range(self.n_layers):
            # -> [batch_size, max_mirna_len, d_model], [batch_size, n_heads, max_mirna_len, max_mirna_len]
            mirna_outputs, mirna_attn = self.mirna_encoder[i](mirna_outputs, mirna_mask)
            mirna_attns.append(mirna_attn)
            # -> [batch_size, max_target_len, d_model], [batch_size, n_heads, max_target_len, max_target_len]
            target_outputs, target_attn = self.target_encoder[i](target_outputs, target_mask)
            target_attns.append(target_attn)
        # decoder
        # -> [batch_size, tgt_len, d_model]
        enc_outputs = torch.cat((mirna_outputs, target_outputs), 1)
        dec_outputs = self.positional_encoding(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # -> [batch_size, tgt_len, tgt_len]
        dec_mask = torch.LongTensor(np.zeros((enc_outputs.shape[0], self.tgt_len, self.tgt_len))).bool().to(enc_outputs.device)
        dec_attns = []
        for i in range(self.n_layers):
            # -> [batch_size, tgt_len, d_model], [batch_size, n_heads, tgt_len, tgt_len]
            dec_outputs, dec_self_attn = self.decoder[i](dec_outputs, dec_mask)
            dec_attns.append(dec_self_attn)

        # flatten
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)
        # projection
        # -> [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        # TODO: mirna_attns, target_attns, dec_attns for visualization
        return torch.sigmoid(dec_logits)

    @staticmethod
    def get_attn_pad_mask(seq_q, seq_k):
        """
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        """
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
