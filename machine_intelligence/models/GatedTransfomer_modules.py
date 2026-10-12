import math
import torch
import torch.nn as nn
import lightning as lit


class MulitHeadAttention(lit.LightningModule):
    def __init__(self, d_model,
                 q: int,
                 v: int,
                 h: int,
                 mask: bool = False,
                 dropout: float = 0.1, ):
        super().__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = nn.Dropout(dropout)

        self.score = None
        self.score_sm = None
        # self.score = []
        # self.score_sm = []

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score
        # self.score.append(score.detach().cpu().numpy())
        # self.score = torch.cat(self.score, score)

        if self.mask and stage == "train":
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.tensor([-2 ** 32 + 1], device=self.device).expand_as(score[0]))

        score = nn.functional.softmax(score, dim=-1)

        self.score_sm = score
        # self.score_sm.append(score.detach().cpu().numpy())

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score, self.score_sm


class FeedForward(nn.Module):
    # def __init__(self, d_model: int, d_hidden: int = 512):
    def __init__(self, d_model: int, d_hidden: int = 512):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x = self.linear_2(x)

        return x


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 mask: bool = False,
                 dropout: float = 0.1,
                 ):
        super().__init__()

        self.MHA = MulitHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, dropout=dropout)
        # self.MHA = nn.MultiheadAttention(embed_dim=d_model, num_heads=h)
        # self.feedforward = FeedForward(d_model=d_hidden, d_hidden=d_hidden)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal_1 = nn.LayerNorm(d_model)
        self.layer_normal_2 = nn.LayerNorm(d_model)

    def forward(self, x, stage):
        residual = x
        x, score, score_sm = self.MHA(x, stage)
        # src = x.transpose(0, 1) # impl. for testing
        # x, score = self.MHA(x, x, x, need_weights=True)
        # x, score = self.MHA(src, src, src, need_weights=True) # impl. for testing
        # x = x.transpose(0, 1)  # (N, seq_len, features) # impl. for testing
        x = self.dropout(x)
        x = self.layer_normal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layer_normal_2(x + residual)

        return x, score, score_sm
        # return x, score
