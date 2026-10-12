import torch.nn as nn


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.num_sensors = num_features
        self.hidden_units = hidden_size
        self.num_layers = 3

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True,  # doesn't apply on hidden or cell states
            num_layers=self.num_layers,
            bidirectional=False,
            proj_size=0  # is the default
        )

        self.linear = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, x):
        # h_0, c_0 defaults are zeros, no need to do it manually
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        # x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths=x_lengths, batch_first=True, enforce_sorted=False)

        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (N,L,D∗Hout) with N: batch_size, L: sequence length,
        # D: D=2 if bidirectional=True otherwise 1, Hout: hidden_size
        # will be a packed sequence -> need to unpack that

        # lstm_out_unpacked, lstm_out_lengths = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)

        out = self.linear(h_n[0])  # dont .flatten() # this will mess up the expected input format for CE loss
        out_softmax = out.softmax(dim=0)

        return out_softmax
