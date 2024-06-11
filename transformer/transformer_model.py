import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoderLayer, Transformer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model-1, 2) * (-math.log(10000.0) / d_model-1))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


class TFEncoderDecoder(nn.Module):
    def __init__(self, task_dim:int, traj_dim: int, embed_dim: int, nhead: int, max_len: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.2, device=None):
        super(TFEncoderDecoder, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)

        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.lin0 = nn.Linear(task_dim, embed_dim, dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)

        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def forward(self, obj_seq, traj_seq, key_mask=None):
        obj_emb = self.lin0(obj_seq) * math.sqrt(self.d_model)
        traj_emb = self.lin0(traj_seq) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        x = self.decoder(traj_emb, obj_emb, tgt_mask=self.mask, tgt_key_padding_mask=key_mask)
        x = self.lin4(x)
        return x

class TFEncoderDecoderAction(nn.Module):
    def __init__(self, task_dim:int, traj_dim: int, n_tasks: int, n_objs:int,embed_dim: int, nhead: int, max_len: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.2, device=None):
        super(TFEncoderDecoderAction, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)

        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.lin0 = nn.Linear(task_dim, embed_dim, dtype=torch.float64)
        self.lin1 = nn.Linear(embed_dim * n_objs, 64 , dtype=torch.float64)
        self.lin2 = nn.Linear(64, 16 , dtype=torch.float64)
        self.lin3 = nn.Linear(16, n_tasks , dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)

        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def forward(self, obj_seq, traj_seq, key_mask=None):
        obj_emb = self.lin0(obj_seq) * math.sqrt(self.d_model)
        traj_emb = self.lin0(traj_seq) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        y = self.lin1(F.relu(obj_emb).view(traj_emb.shape[0], -1))
        y = F.relu(y)
        y = self.lin2(y)
        y = F.relu(y)
        y = self.lin3(y)
        x = self.decoder(traj_emb, obj_emb, tgt_mask=self.mask, tgt_key_padding_mask=key_mask)
        x = self.lin4(x)
        return x, y


class TFEncoderDecoderNoMask(nn.Module):
    def __init__(self, task_dim:int, traj_dim: int, embed_dim: int, nhead: int, max_len: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.2, device=None):
        super(TFEncoderDecoderNoMask, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)

        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.lin0 = nn.Linear(task_dim, embed_dim, dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)

        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def forward(self, obj_seq, traj_seq, key_mask=None):
        obj_emb = self.lin0(obj_seq) * math.sqrt(self.d_model)
        traj_emb = self.lin0(traj_seq) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        x = self.decoder(traj_emb, obj_emb)
        x = self.lin4(x)
        return x
