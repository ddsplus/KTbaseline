import torch

from torch.nn import Module, Embedding, Parameter, Sequential, Linear, ReLU, Dropout, GRU
from torch.nn.init import kaiming_normal_
from torch.nn.functional import one_hot


def mlp(in_size, out_size):
    return Sequential(
        Linear(in_size, out_size),
        ReLU(),
        Dropout(),
        Linear(out_size, out_size),
    )


class FlowMatcher(Module):
    def __init__(self, emb_size, hidden_size, time_size):
        super().__init__()
        self.time_mlp = Sequential(
            Linear(1, time_size),
            ReLU(),
            Linear(time_size, time_size),
            ReLU(),
        )
        self.velocity = Sequential(
            Linear(emb_size + time_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, emb_size),
        )

    def _velocity(self, x, t):
        # x: [B, D], t: [B, 1]
        t_emb = self.time_mlp(t)
        return self.velocity(torch.cat([x, t_emb], dim=-1))

    def loss(self, x0, x1):
        # x0/x1: [B, D]
        if x0.numel() == 0:
            return x0.new_tensor(0.0)
        batch = x0.shape[0]
        t = torch.rand(batch, 1, device=x0.device, dtype=x0.dtype)
        xt = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0
        pred_v = self._velocity(xt, t)
        return ((pred_v - target_v) ** 2).mean()

    def integrate(self, x0, steps):
        # x0: [B, D]
        if x0.numel() == 0:
            return x0
        x = x0
        dt = 1.0 / max(1, int(steps))
        batch = x.shape[0]
        for i in range(max(1, int(steps))):
            t_val = (i + 0.5) * dt
            t = torch.full((batch, 1), t_val, device=x.device, dtype=x.dtype)
            x = x + dt * self._velocity(x, t)
        return x


class GKTFM(Module):
    def __init__(
        self,
        num_q,
        hidden_size,
        num_attn_heads,
        method,
        fm_hidden_size=128,
        fm_time_size=32,
        fm_noise=0.1,
        fm_steps=4,
        fm_lambda=0.2,
    ):
        super().__init__()
        if method not in ["PAM"]:
            raise ValueError("gkt-fm currently supports method='PAM' only.")

        self.num_q = num_q
        self.hidden_size = hidden_size
        self.fm_noise = float(fm_noise)
        self.fm_steps = int(fm_steps)
        self.fm_lambda = float(fm_lambda)

        self.x_emb = Embedding(self.num_q * 2, self.hidden_size)
        self.q_emb = Parameter(torch.Tensor(self.num_q, self.hidden_size))
        kaiming_normal_(self.q_emb)

        self.init_h = Parameter(torch.Tensor(self.num_q, self.hidden_size))
        kaiming_normal_(self.init_h)

        self.mlp_self = mlp(self.hidden_size * 2, self.hidden_size)
        self.gru = GRU(self.hidden_size * 2, self.hidden_size, batch_first=True)

        self.A = Parameter(torch.Tensor(self.num_q, self.num_q))
        kaiming_normal_(self.A)
        self.mlp_outgo = mlp(self.hidden_size * 4, self.hidden_size)
        self.mlp_income = mlp(self.hidden_size * 4, self.hidden_size)

        self.fm = FlowMatcher(
            emb_size=self.hidden_size,
            hidden_size=int(fm_hidden_size),
            time_size=int(fm_time_size),
        )

        self.bias = Parameter(torch.Tensor(1, self.num_q, 1))
        self.out_layer = Linear(self.hidden_size, 1, bias=False)
        self.last_aux_losses = {}

    def aggregate(self, xt_emb, qt_onehot, q_emb, ht):
        xt_emb = xt_emb.unsqueeze(1).repeat(1, self.num_q, 1)
        qt_onehot = qt_onehot.unsqueeze(-1)
        return qt_onehot * torch.cat([ht, xt_emb], dim=-1) + (1 - qt_onehot) * torch.cat([ht, q_emb], dim=-1)

    def f_self(self, ht_):
        return self.mlp_self(ht_)

    def f_neighbor(self, ht_, qt):
        batch_size = qt.shape[0]
        src = ht_
        tgt = torch.gather(
            ht_,
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ht_.shape[-1]),
        )

        Aij = torch.gather(
            self.A.unsqueeze(0).repeat(batch_size, 1, 1),
            dim=1,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.A.shape[-1]),
        ).squeeze()

        outgo_part = Aij.unsqueeze(-1) * self.mlp_outgo(
            torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
        )

        Aji = torch.gather(
            self.A.unsqueeze(0).repeat(batch_size, 1, 1),
            dim=2,
            index=qt.unsqueeze(-1).unsqueeze(-1).repeat(1, self.A.shape[-1], 1),
        ).squeeze()

        income_part = Aji.unsqueeze(-1) * self.mlp_income(
            torch.cat([tgt.repeat(1, self.num_q, 1), src], dim=-1)
        )

        return outgo_part + income_part

    def update(self, ht, ht_, qt, qt_onehot):
        qt_onehot = qt_onehot.unsqueeze(-1)
        m = qt_onehot * self.f_self(ht_) + (1 - qt_onehot) * self.f_neighbor(ht_, qt)
        ht, _ = self.gru(torch.cat([m, ht], dim=-1))
        return ht

    def predict(self, ht):
        return torch.sigmoid(self.out_layer(ht) + self.bias).squeeze(-1)

    def _predict_with_denoised_qt(self, ht, qt):
        # ht: [B, Q, H], qt: [B]
        idx = qt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.hidden_size)
        x1 = torch.gather(ht, dim=1, index=idx).squeeze(1)  # [B, H]
        if self.training and self.fm_noise > 0:
            x0 = x1 + self.fm_noise * torch.randn_like(x1)
        else:
            x0 = x1

        fm_loss = self.fm.loss(x0, x1)
        xhat = self.fm.integrate(x0, self.fm_steps)

        # Avoid cloning the full hidden tensor every step; update only logits at q_t.
        logits = self.out_layer(ht) + self.bias  # [B, Q, 1]
        qt_logits = self.out_layer(xhat.unsqueeze(1))  # [B, 1, 1]
        qt_bias = torch.gather(self.bias.repeat(ht.shape[0], 1, 1), dim=1, index=qt.unsqueeze(-1).unsqueeze(-1))
        qt_logits = qt_logits + qt_bias
        logits.scatter_(1, qt.unsqueeze(-1).unsqueeze(-1), qt_logits)
        y = torch.sigmoid(logits).squeeze(-1)
        return y, fm_loss

    def forward(self, q, r, train=False):
        batch_size = q.shape[0]
        x = q + self.num_q * r

        x_emb = self.x_emb(x)
        q_emb = self.q_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        q_onehot = one_hot(q, self.num_q)

        ht = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        y = []
        fm_losses = []

        for xt_emb, qt, qt_onehot in zip(
            x_emb.permute(1, 0, 2),
            q.permute(1, 0),
            q_onehot.permute(1, 0, 2),
        ):
            ht_ = self.aggregate(xt_emb, qt_onehot, q_emb, ht)
            ht = self.update(ht, ht_, qt, qt_onehot)

            yt, fm_loss_t = self._predict_with_denoised_qt(ht, qt)
            y.append(yt)
            fm_losses.append(fm_loss_t)

        y = torch.stack(y, dim=1)

        if len(fm_losses) > 0:
            fm_loss = torch.stack(fm_losses).mean()
        else:
            fm_loss = y.new_tensor(0.0)

        aux_losses = {"fm_loss": self.fm_lambda * fm_loss}
        self.last_aux_losses = aux_losses
        if train:
            return y, None, aux_losses
        return y, None
