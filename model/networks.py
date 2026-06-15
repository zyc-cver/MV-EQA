import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ViewEncoder(nn.Module):
    def __init__(self, feature_size=28, d_model=8, nhead=1, num_layers=3, dim_feedforward=512, max_seq_length=64,
                 dropout=0.1):
        super(ViewEncoder, self).__init__()

        self.linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # (S, B, F)
        src = self.linear(src)  # (S, B, d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)  # (S, B, d_model)

        output = output.permute(1, 2, 0)
        return output

class BodyEncoder(nn.Module):
    def __init__(self, feature_size=28, d_model=16, nhead=1, num_layers=3, dim_feedforward=512, max_seq_length=64,
                 dropout=0.1):
        super(BodyEncoder, self).__init__()

        self.linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src):
        # Reshape input to [seq_len, batch, features]
        src = src.permute(2, 0, 1)  # (S, B, F)
        src = self.linear(src)  # (S, B, d_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)  # (S, B, d_model)

        # Adjust the output feature dimension
        output = output.permute(1, 2, 0)
        return output

class MotionEncoder(nn.Module):
    def __init__(self, feature_size=30, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, max_seq_length=64,
                 dropout=0.1):
        super(MotionEncoder, self).__init__()

        self.linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # (S, B, F)
        src = self.linear(src)  # (S, B, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # (S, B, d_model)
        output = output.permute(1, 2, 0)

        return output

class Decoder_3x(nn.Module):
    def __init__(self, feature_size=160, d_model=30, nhead=2, num_layers=2, dim_feedforward=2048, max_seq_length=64, dropout=0.1):
        super(Decoder_3x, self).__init__()

        self.linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, tgt):

        tgt = tgt.permute(2, 0, 1)  # (S, B, F)                     ``
        tgt = self.linear(tgt)  # (S, B, d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_encoder(tgt)

        # Adjust the feature size to the output size (30)
        output = output.permute(1, 2, 0)
        return output

class TransformerNetwork_3x(nn.Module):
    def __init__(self, mot_feature_size=30, view_feature_size=28, body_feature_size=28, combined_feature_size=152, decoder_feature_size=30, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048, max_seq_length=1000, dropout=0.1):
        super(TransformerNetwork_3x, self).__init__()

        # Define MOT Encoder
        self.mot_encoder = MotionEncoder(
            feature_size=mot_feature_size,
            d_model=128,
            nhead=nhead,
            num_layers=2,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        # Define Static Encoder
        self.view_encoder = ViewEncoder(
            feature_size=view_feature_size,
            d_model=8,
            nhead=1,  # 3
            num_layers=3,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        # Define Static Encoder
        self.body_encoder = BodyEncoder(
            feature_size=body_feature_size,
            d_model=16,
            nhead=1,  # 5
            num_layers=3,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        # Define Decoder
        self.decoder = Decoder_3x(
            feature_size=combined_feature_size,  # This is the combined feature size after concatenation
            d_model=30,
            nhead=2,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

    def cross(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b1 = self.body_encoder(x1[:, :-2, :])
        v1 = self.view_encoder(x1[:, :-2, :])
        m2 = self.mot_encoder(x2)
        b2 = self.body_encoder(x2[:, :-2, :])
        v2 = self.view_encoder(x2[:, :-2, :])

        out1 = self.decoder(torch.cat([m1, b1, v1], dim=1))
        out2 = self.decoder(torch.cat([m2, b2, v2], dim=1))
        out121 = self.decoder(torch.cat([m1, b2, v1], dim=1))
        out112 = self.decoder(torch.cat([m1, b1, v2], dim=1))
        out122 = self.decoder(torch.cat([m1, b2, v2], dim=1))
        out212 = self.decoder(torch.cat([m2, b1, v2], dim=1))
        out221 = self.decoder(torch.cat([m2, b2, v1], dim=1))
        out211 = self.decoder(torch.cat([m2, b1, v1], dim=1))

        return out1, out2, out121, out112, out122, out212, out221, out211

    def cross_with_triplet(self, inputs):
        x1, x2, x121, x112, x122, x212, x221, x211 = inputs
        m1 = self.mot_encoder(x1)
        b1 = self.body_encoder(x1[:, :-2, :])
        v1 = self.view_encoder(x1[:, :-2, :])
        m2 = self.mot_encoder(x2)
        b2 = self.body_encoder(x2[:, :-2, :])
        v2 = self.view_encoder(x2[:, :-2, :])

        out1 = self.decoder(torch.cat([m1, b1, v1], dim=1))
        out2 = self.decoder(torch.cat([m2, b2, v2], dim=1))
        out121 = self.decoder(torch.cat([m1, b2, v1], dim=1))
        out112 = self.decoder(torch.cat([m1, b1, v2], dim=1))
        out122 = self.decoder(torch.cat([m1, b2, v2], dim=1))
        out212 = self.decoder(torch.cat([m2, b1, v2], dim=1))
        out221 = self.decoder(torch.cat([m2, b2, v1], dim=1))
        out211 = self.decoder(torch.cat([m2, b1, v1], dim=1))

        outputs = [out1, out2, out121, out112, out122, out212, out221, out211]

        m122 = self.mot_encoder(x122)
        m211 = self.mot_encoder(x211)
        b212 = self.body_encoder(x212[:, :-2, :])
        b121 = self.body_encoder(x121[:, :-2, :])
        v221 = self.view_encoder(x221[:, :-2, :])
        v112 = self.view_encoder(x112[:, :-2, :])

        motionvecs = [m1.reshape(m1.shape[0], -1),
                      m2.reshape(m2.shape[0], -1),
                      m122.reshape(m122.shape[0], -1),
                      m211.reshape(m211.shape[0], -1)]
        bodyvecs = [b1.reshape(b1.shape[0], -1),
                    b2.reshape(b2.shape[0], -1),
                    b212.reshape(b212.shape[0], -1),
                    b121.reshape(b121.shape[0], -1)]
        viewvecs = [v1.reshape(v1.shape[0], -1),
                    v2.reshape(v2.shape[0], -1),
                    v221.reshape(v221.shape[0], -1),
                    v112.reshape(v112.shape[0], -1)]

        return outputs, motionvecs, bodyvecs, viewvecs


    def transfer_body(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.body_encoder(x2[:, :-2, :])
        v1 = self.view_encoder(x1[:, :-2, :])

        out12 = self.decoder(torch.cat([m1, b2, v1], dim=1))

        return out12

    def transfer_view(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b1 = self.body_encoder(x1[:, :-2, :])
        v2 = self.view_encoder(x2[:, :-2, :])

        out12 = self.decoder(torch.cat([m1, b1, v2], dim=1))

        return out12

    def transfer_both(self, x1, x2):
        m1 = self.mot_encoder(x1)
        b2 = self.body_encoder(x2[:, :-2, :])
        v2 = self.view_encoder(x2[:, :-2, :])

        out12 = self.decoder(torch.cat([m1, b2, v2], dim=1))

        return out12

    def transfer_three(self, x1, x2, x3):
        m1 = self.mot_encoder(x1)
        b2 = self.body_encoder(x2[:, :-2, :])
        v3 = self.view_encoder(x3[:, :-2, :])

        out = self.decoder(torch.cat([m1, b2, v3], dim=1))

        return out

    def forward(self, mot_input, view_input, body_input, decoder_input=None):
        mot_encoded = self.mot_encoder(mot_input)
        view_encoded = self.view_encoder(view_input)
        body_encoded = self.view_encoder(body_input)

        combined_encoded = torch.cat((mot_encoded, view_encoded, body_encoded), dim=1)

        decoded_output = self.decoder(combined_encoded)

        return decoded_output