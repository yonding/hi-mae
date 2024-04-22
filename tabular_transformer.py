import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TabularTransformer(nn.Module):
    def __init__(self, args):
        super(TabularTransformer, self).__init__()
        self.num_features = args.num_features
        self.dim_model = args.dim_model
        self.batch_size = args.batch_size
        self.embedding = nn.Sequential(
            nn.Linear(1, args.dim_model),
            # nn.Linear(1, 64),  
            # nn.Linear(args.dim_model, args.dim_model), 
            # nn.Linear(64, 128), 
            # nn.Linear(args.dim_model, args.dim_model) 
            # nn.Linear(128, args.dim_model) 
        )

        # self.pos_encoder = PositionalEncoding(args.dim_model, args.dropout)

        encoder_layers = TransformerEncoderLayer(
            args.dim_model, args.num_head, args.dim_ff, args.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.num_layers)

        self.cls_token = nn.Parameter(torch.rand(1, 1, args.dim_model))
        self.decoder = nn.Linear(args.dim_model, args.num_classes)
        self.imputation_decoder = nn.Sequential(
            # nn.Linear(args.dim_model, args.dim_model),  
            # nn.Linear(args.dim_model, args.dim_model), 
            nn.Linear(args.dim_model, 1) 
        )

    def forward(self, src):
        batch_size = src.size(0)
        
        src = self.embedding(src.unsqueeze(dim=-1))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        src = torch.cat((cls_tokens, src), dim=1)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        cls_output = output[:, 0, :]
        tokens = output[:,1:,:]
        tokens = self.imputation_decoder(tokens)
        tokens = tokens.squeeze(dim=-1)
        cls_output = self.decoder(cls_output)
        return cls_output, tokens


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
