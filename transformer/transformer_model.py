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

    def forward(self, source, target, padding_mask=None):
        obj_emb = self.lin0(source) * math.sqrt(self.d_model)
        traj_emb = self.lin0(target) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        x = self.decoder(traj_emb, obj_emb, tgt_mask=self.mask, tgt_key_padding_mask=padding_mask)
        x = self.lin4(x)
        return x

class TFEncoderDecoder2(nn.Module):
    def __init__(self, task_dim:int, traj_dim: int, n_tasks: int, n_objs:int,embed_dim: int, nhead: int, max_len: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.2, device=None):
        super(TFEncoderDecoder2, self).__init__()
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

    def forward(self, source, target, padding_mask=None):
        obj_emb = self.lin0(source) * math.sqrt(self.d_model)

        traj_emb = self.lin0(target) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        y = self.lin1(F.relu(obj_emb).view(traj_emb.shape[0], -1))
        y = F.relu(y)
        y = self.lin2(y)
        y = F.relu(y)
        y = self.lin3(y)
        x = self.decoder(traj_emb, obj_emb, tgt_mask=self.mask, tgt_key_padding_mask=padding_mask)
        x = self.lin4(x)
        return x, y

class ActionClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ActionClassifier, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, 64, dtype=torch.float64),
                                                nn.ReLU(),
                                                nn.Linear(64, 16, dtype=torch.float64),
                                                nn.ReLU(),
                                                nn.Linear(16, output_dim, dtype=torch.float64)]
                                               )
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TFEncoderDecoder3(nn.Module):
    def __init__(self, task_dim:int, target_dim: int, source_dim, n_tasks: int, embed_dim: int, nhead: int, max_len: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.2, device=None):
        super(TFEncoderDecoder3, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.n_objs = int(source_dim / target_dim)
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)

        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)
        self.traj_embed_layer = nn.Linear(target_dim, embed_dim, dtype=torch.float64)
        self.obj_embed_layer = nn.Linear(source_dim, embed_dim, dtype=torch.float64)
        self.action_classifier = ActionClassifier(embed_dim, n_tasks)
        self.lin4 = nn.Linear(embed_dim, task_dim, dtype=torch.float64)

        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def encode(self, source, src_mask, padding_mask):
        ### Encoder side
        src_emb = self.obj_embed_layer(source) * math.sqrt(self.d_model)
        src_emb = self.pos_embed(src_emb)
        if self.num_encoder_layers > 0:
            src_emb = self.encoder(src_emb, mask= src_mask, src_key_padding_mask=padding_mask)
        return src_emb

    def decode(self, src_emb, target, tgt_mask, tgt_padding_mask, memory_mask, memory_padding_mask ):
        tgt_emb = self.traj_embed_layer(target) * math.sqrt(self.d_model)
        tgt_emb = self.pos_embed(tgt_emb)
        x = self.decoder(tgt_emb, src_emb, tgt_mask= tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                         memory_mask=memory_mask, memory_key_padding_mask=memory_padding_mask)
        x = self.lin4(x)
        return x

    def forward(self, source, target, src_padding_mask=None, tgt_padding_mask = None, memory_padding_mask = None, predict_action = False):
        ### Encoder side
        src_emb = self.encode(source, self.mask, src_padding_mask)
        ### Decoder side
        x = self.decode(src_emb, target, self.mask, tgt_padding_mask, self.mask, memory_padding_mask)
        if predict_action:
            action = self.action_classifier(x)
        else:
            action = None
        return x, action


class TFEncoderDecoder4(nn.Module):
    def __init__(self, task_dim: int, source_dim: int, target_dim: int, n_tasks: int, embed_dim: int, nhead: int,
                 max_len: int,
                 num_encoder_layers: int, num_decoder_layers: int, dropout: float = 0.2, device=None):
        super(TFEncoderDecoder4, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.embed_layer = nn.Linear(target_dim, embed_dim, dtype=torch.float64)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        if num_encoder_layers > 0:
            e_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
            self.encoder = nn.TransformerEncoder(e_layer, num_layers=self.num_encoder_layers)

        d_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(decoder_layer=d_layer, num_layers=self.num_decoder_layers)

        self.action_classifier = ActionClassifier(embed_dim, n_tasks)
        self.lin4 = nn.Linear(embed_dim, task_dim, dtype=torch.float64)

        self.mask = torch.triu(torch.full((max_len, max_len), True), diagonal=1)
        if device:
            self.to(device)
            self.mask = torch.triu(torch.full((max_len, max_len), True, device=device), diagonal=1)

    def encode(self, source, src_mask=None, src_padding_mask=None):
        ### Encoder side
        src_emb = self.embed_layer(source) * math.sqrt(self.d_model)
        if self.num_encoder_layers > 0:
            src_emb = self.encoder(src_emb, mask=src_mask, src_key_padding_mask= src_padding_mask)
        return src_emb

    def decode(self, src_emb, target, tgt_mask, tgt_padding_mask, memory_mask=None, memory_padding_mask=None):
        if tgt_mask is None:
            tgt_mask = self.mask
        tgt_emb = self.embed_layer(target) * math.sqrt(self.d_model)
        tgt_emb = self.pos_embed(tgt_emb)
        x = self.decoder(tgt_emb, src_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                         memory_mask=memory_mask, memory_key_padding_mask=memory_padding_mask)
        x = self.lin4(x)
        return x

    def forward(self, source, target, src_mask=None, tgt_mask=None, memory_mask=None, src_padding_mask=None,
                tgt_padding_mask=None, memory_padding_mask=None, predict_action=False):
        ### Encoder side
        src_emb = self.encode(source, src_mask=src_mask, src_padding_mask=src_padding_mask)
        ### Decoder side
        x = self.decode(src_emb, target, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask, memory_mask=memory_mask,
                        memory_padding_mask=memory_padding_mask)
        if predict_action:
            action = self.action_classifier(x)
        else:
            action = None
        return x, action

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

    def forward(self, source, target, padding_mask=None):
        obj_emb = self.lin0(source) * math.sqrt(self.d_model)
        traj_emb = self.lin0(target) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        if self.num_encoder_layers > 0:
            obj_emb = self.encoder(obj_emb)
        x = self.decoder(traj_emb, obj_emb)
        x = self.lin4(x)
        return x
