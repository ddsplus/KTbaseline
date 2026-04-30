import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout


class DKTForget(Module):
    """
    DKT-Forget:
    - DKT backbone + learnable forget gate over hidden states.
    """
    def __init__(self, num_q, emb_size, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.forget_layer = Linear(self.hidden_size + self.emb_size, self.hidden_size)
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        x = q + self.num_q * r
        emb = self.interaction_emb(x)
        h, _ = self.lstm_layer(emb)

        h_prev = torch.cat([torch.zeros_like(h[:, :1, :]), h[:, :-1, :]], dim=1)
        forget_in = torch.cat([h, emb], dim=-1)
        forget_gate = torch.sigmoid(self.forget_layer(forget_in))
        h_forget = forget_gate * h + (1.0 - forget_gate) * h_prev

        y = torch.sigmoid(self.out_layer(self.dropout_layer(h_forget)))
        return y
