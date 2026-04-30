import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU


class KQN(Module):
    def __init__(self, num_q, dim_v, dim_s, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.dim_v = dim_v
        self.dim_s = dim_s
        self.hidden_size = hidden_size

        self.x_emb = Embedding(self.num_q * 2, self.dim_v)
        self.knowledge_encoder = LSTM(self.dim_v, self.dim_v, batch_first=True)
        self.out_layer = Linear(self.dim_v, self.dim_s)
        self.dropout_layer = Dropout()

        self.q_emb = Embedding(self.num_q, self.dim_v)
        self.skill_encoder = Sequential(
            Linear(self.dim_v, self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.dim_v),
            ReLU()
        )

    def forward(self, q, r, qry):
        # Knowledge State Encoding
        x = q + self.num_q * r
        x = self.x_emb(x)
        h, _ = self.knowledge_encoder(x)
        ks = self.out_layer(h)
        ks = self.dropout_layer(ks)

        # Skill Encoding
        e = self.q_emb(qry)
        o = self.skill_encoder(e)
        s = o / torch.norm(o, p=2)

        p = torch.sigmoid((ks * s).sum(-1))

        return p
