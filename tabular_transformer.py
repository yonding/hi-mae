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
            nn.Linear(args.dim_model, args.dim_model), 
            nn.Linear(args.dim_model, args.dim_model) 
        )
        encoder_layers = TransformerEncoderLayer(
            args.dim_model, args.num_head, args.dim_ff, args.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.num_layers)

        # before being input to the decoder, the output of the transformer encoder is passed through a linear layer
        self.reduce_cls_dimension = nn.Linear(args.dim_model, args.num_classes if args.dataset_name != 'diabetes' else 1)
        self.reduce_feature_dimension = nn.Linear(args.dim_model, 1)

        decoder_layers = nn.TransformerDecoderLayer(
            args.dim_model, args.num_head, args.dim_ff, args.dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, args.num_layers)

        self.cls_token = nn.Parameter(torch.rand(1, 1, args.dim_model))
        self.decoder = nn.Linear(args.dim_model, args.num_classes if args.dataset_name != 'diabetes' else 1)

    def forward(self, missing_data, complete_data):
        batch_size = missing_data.size(0)

        # embedding vector for features and cls token
        missing_tokens = self.embedding(missing_data.unsqueeze(dim=-1))
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        embedding_vectors = torch.cat((cls_token, missing_tokens), dim=1)
        
        # passing through transformer encoder
        encoder_output = self.transformer_encoder(embedding_vectors)

        cls_output = self.reduce_cls_dimension(encoder_output[:, 0, :])
        features_encoder_output = encoder_output[:,1:,:]

        complete_tokens = self.embedding(complete_data.unsqueeze(dim=-1))

        decoder_output = self.transformer_decoder(complete_tokens, features_encoder_output)
        
        decoder_output = self.reduce_feature_dimension(decoder_output).squeeze(dim=-1)
        
        return cls_output, decoder_output


# class PositionalEncoding(nn.Module):
#     def __init__(self, dim_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
#         )
#         pe = torch.zeros(max_len, 1, dim_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + self.pe[: x.size(0), :]
#         return self.dropout(x)
