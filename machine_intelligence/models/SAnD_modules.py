import math

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()

        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for position in range(seq_len):
            for i in range(0, d_model, 2):
                pe[position, i] = math.sin(position / (1e4 ** ((2 * i) / d_model)))
                pe[position, i + 1] = math.cos(position / (1e4 ** ((2 * i + 1) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:

        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, features)
        :return: (batch_size, seq_len, features)
        """

        if isinstance(self.layer, nn.MultiheadAttention):
            src = x.transpose(0, 1)  # (seq_len, N, features)
            output, self.attn_weights = self.layer(src, src, src)
            output = output.transpose(0, 1)  # (N, seq_len, features)

        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)

        return output


class ResidualBlockMHA(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None
        self.attention_wo_res = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, features)
        :return: (batch_size, seq_len, features)
        """

        mask = subsequent_mask(x.shape[1]).to(x.device)

        src = x.transpose(0, 1)  # (seq_len, N, features)
        output, self.attn_weights = self.layer(src, src, src, need_weights=True, attn_mask=mask)
        output = output.transpose(0, 1)  # (N, seq_len, features)

        self.attention_output = output

        output = self.dropout(output)
        output = self.norm(x + output)

        return output


class ResidualBlockPFF(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, features)
        :return: (batch_size, seq_len, features)
        """

        output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1)
        )

    def forward(self, fw_tensor: torch.Tensor) -> torch.Tensor:
        fw_tensor = fw_tensor.transpose(1, 2)
        fw_tensor = self.conv(fw_tensor)
        fw_tensor = fw_tensor.transpose(1, 2)

        return fw_tensor


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape, dtype=np.float32), k=1)
    mask = torch.from_numpy(mask) == 0
    return mask.float()


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout=0.1):
        super().__init__()
        # self.attention_wo_res = nn.MultiheadAttention(embed_dim, num_head)
        # self.attention = ResidualBlock(nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout)
        # self.attention = ResidualBlock(self.attention_wo_res, embed_dim, p=dropout)
        self.attention = ResidualBlockMHA(nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout)
        # self.ffn = ResidualBlock(PositionWiseFeedForward(embed_dim), embed_dim, p=dropout)
        self.ffn = ResidualBlockPFF(PositionWiseFeedForward(embed_dim), embed_dim, p=dropout)
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)

        # self.attention_weights = self.attention

        return x


class DenseInterpolation(nn.Module):
    def __init__(self, seq_len: int, factor: int):
        """
        :param factor: factor M
        """
        super().__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2, )


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, factor: int, num_classes_ts: int, num_static_feat: int, num_static_hidden: int,
                 num_static_outs: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.num_classes_ts = num_classes_ts
        self.num_static_feat = num_static_feat
        self.num_static_hidden = num_static_hidden
        self.num_static_outs = num_static_outs
        self.dropout_static = dropout

        self.fc = nn.Linear(int(d_model * factor) + self.num_static_outs, num_classes_ts)

        # nn.init.normal_(self.fc.weight, std=0.02)
        # nn.init.normal_(self.fc.bias, 0)

        self.static_layers = nn.Sequential(
            nn.Linear(in_features=self.num_static_feat, out_features=self.num_static_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.num_static_hidden, out_features=self.num_static_outs),
        )

    def forward(self, x: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, int(self.factor * self.d_model))

        x_static = self.static_layers(x_static)
        x = torch.cat([x, x_static], dim=1)
        x = self.fc(x)
        return x
