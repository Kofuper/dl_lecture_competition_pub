import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math

class CNNLSTMClassifier(nn.Module):
    def __init__(self, num_classes, in_channels, seq_len, lstm_hidden_size=128, lstm_layers=2, p_drop=0.2):
        super(CNNLSTMClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(p_drop)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(p_drop)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(p_drop)

        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
        self.dropout4 = nn.Dropout(p_drop)
        
    def forward(self, x):
        # CNN part
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Prepare data for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_channels)
        
        # LSTM part
        x, _ = self.lstm(x)
        x = self.dropout4(x[:, -1, :])  # take the last output of the LSTM
        
        # Fully connected layer
        x = self.fc(x)
        return x


class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128, p_drop: float = 0.3) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, p_drop, kernel_size=3),
            ConvBlock(hid_dim, hid_dim, p_drop, kernel_size=3),
            ConvBlock(hid_dim, hid_dim * 2, p_drop, kernel_size=3),  # 追加の畳み込みブロック
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 2, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, p_drop: float = 0.3, kernel_size: int = 3) -> None:
        super().__init__()
        
        padding = kernel_size // 2  # パディングを整数として計算
        print(f"padding: {padding}, kernel_size: {kernel_size}")  # デバッグ出力

        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout1 = nn.Dropout(p_drop)

        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding=padding)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout2 = nn.Dropout(p_drop)

        self.skip = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else None
        self.batchnorm_skip = nn.BatchNorm1d(num_features=out_dim) if self.skip else None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        identity = X
        
        X = F.gelu(self.batchnorm1(self.conv1(X)))
        X = self.dropout1(X)
        X = self.batchnorm2(self.conv2(X))

        if self.skip:
            identity = self.batchnorm_skip(self.skip(identity))
        
        X += identity
        X = F.gelu(X)
        X = self.dropout2(X)

        return X
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, in_channels, seq_len, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.2):
        super(TransformerClassifier, self).__init__()

        self.input_linear = nn.Linear(in_channels, d_model)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(d_model)  # レイヤー正規化の追加
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.layer_norm2 = nn.LayerNorm(d_model)  # レイヤー正規化の追加
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # Change to (seq_len, batch_size, in_channels)
        x = self.input_linear(x)
        x = self.relu(x)
        x = self.layer_norm1(x)  # レイヤー正規化を適用
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.layer_norm2(x)  # レイヤー正規化を適用
        x = self.dropout(x)
        x = self.fc(x)
        return x