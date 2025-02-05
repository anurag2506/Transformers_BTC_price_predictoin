import torch 
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Pos
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        
class BTCTransformer(nn.Module):
    def __init__(self, input_features = 5,
                 d_model = 256,
                 nhead = 5,
                 num_encoder_layers = 6,
                 dim_feedforward = 1024,
                 dropout = 0.1,
                 sequence_length = 168,
                 activation = 'gelu'
                 ):
        super().__init__()
        self.sequence_length = sequence_length
        self.input_embedding = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64,1)
        )
    
    def forward(self,x):
        
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)
        last_hidden = transformer_out[:,-1,:]
        out = self.decoder(last_hidden)
        return out
    
    
def preprocess_data(df, sequence_length = 168):
    
    features = ['Open', 'High', 'Low', 'Close', "Volume"]
    normalized_data = {}
    
    for feature in features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        normalized_data[feature] = (df[feature] - min_val) / (max_val - min_val)

    X, y = [], []
    data = np.column_stack([normalized_data[f] for f in features])
    
    for i in range(len(df) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(normalized_data['close'][i + sequence_length])
    
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)



