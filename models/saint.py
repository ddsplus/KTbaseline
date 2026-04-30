import torch

from torch.nn import Module, Parameter, Embedding, Linear, Transformer
from torch.nn.init import normal_


class SAINT(Module):
    def __init__(
        self, num_q, n, d, num_attn_heads, dropout, num_tr_layers=1
    ):
        super().__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_tr_layers = num_tr_layers

        self.E = Embedding(self.num_q, self.d)
        self.R = Embedding(2, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        self.S = Parameter(torch.Tensor(1, self.d))

        normal_(self.P)
        normal_(self.S)

        self.transformer = Transformer(
            self.d,
            self.num_attn_heads,
            num_encoder_layers=self.num_tr_layers,
            num_decoder_layers=self.num_tr_layers,
            dropout=self.dropout,
        )

        self.pred = Linear(self.d, 1)

    def forward(self, q, r):
        batch_size = r.shape[0]

        E = self.E(q).permute(1, 0, 2)

        R = self.R(r[:, :-1]).permute(1, 0, 2)
        S = self.S.repeat(batch_size, 1).unsqueeze(0)
        R = torch.cat([S, R], dim=0)

        P = self.P.unsqueeze(1)

        mask = self.transformer.generate_square_subsequent_mask(self.n).to(q.device)
        R = self.transformer(
            E + P, R + P, mask, mask, mask
        )
        R = R.permute(1, 0, 2)

        p = torch.sigmoid(self.pred(R)).squeeze()

        return p

    def discover_concepts(self, q, r):
        queries = torch.LongTensor([list(range(self.num_q))] * self.n)\
            .permute(1, 0)

        x = q + self.num_q * r
        x = x.repeat(self.num_q, 1)

        M = self.M(x).permute(1, 0, 2)
        E = self.E(queries).permute(1, 0, 2)
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool()

        M += P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights
