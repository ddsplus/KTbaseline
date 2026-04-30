import math

import torch

from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1 ** 2, dim=-1, keepdim=True)
    mean2_2 = torch.sum(mean2 ** 2, dim=-1, keepdim=True)
    mean_term = -2 * torch.matmul(mean1, mean2.transpose(-1, -2))
    mean_term = mean_term + mean1_2 + mean2_2.transpose(-1, -2)

    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    cov1_2 = torch.sum(cov1, dim=-1, keepdim=True)
    cov2_2 = torch.sum(cov2, dim=-1, keepdim=True)
    cov_term = -2 * torch.matmul(cov1_sqrt, cov2_sqrt.transpose(-1, -2))
    cov_term = cov_term + cov1_2 + cov2_2.transpose(-1, -2)
    return mean_term + cov_term


class WassersteinNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, mean_a, cov_a, mean_b, cov_b):
        cov_a = torch.nn.functional.elu(cov_a) + 1.0
        cov_b = torch.nn.functional.elu(cov_b) + 1.0

        sim_aa = -wasserstein_distance_matmul(mean_a, cov_a, mean_a, cov_a) / self.temperature
        sim_bb = -wasserstein_distance_matmul(mean_b, cov_b, mean_b, cov_b) / self.temperature
        sim_ab = -wasserstein_distance_matmul(mean_a, cov_a, mean_b, cov_b) / self.temperature

        bsz = sim_ab.shape[-1]
        sim_aa[..., range(bsz), range(bsz)] = float("-inf")
        sim_bb[..., range(bsz), range(bsz)] = float("-inf")
        logits_top = torch.cat([sim_ab, sim_aa], dim=-1)
        logits_bottom = torch.cat([sim_bb, sim_ab.transpose(-1, -2)], dim=-1)
        logits = torch.cat([logits_top, logits_bottom], dim=-2)
        labels = torch.arange(2 * bsz, device=logits.device, dtype=torch.long)
        return self.ce(logits, labels)


class UKTAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = Dropout(dropout)

        self.q_mean = Linear(hidden_size, hidden_size)
        self.k_mean = Linear(hidden_size, hidden_size)
        self.v_mean = Linear(hidden_size, hidden_size)
        self.q_cov = Linear(hidden_size, hidden_size)
        self.k_cov = Linear(hidden_size, hidden_size)
        self.v_cov = Linear(hidden_size, hidden_size)
        self.out_mean = Linear(hidden_size, hidden_size)
        self.out_cov = Linear(hidden_size, hidden_size)
        self.gamma = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def _split(self, x):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

    def forward(self, mean_x, cov_x, mask):
        q_mean = self._split(self.q_mean(mean_x))
        k_mean = self._split(self.k_mean(mean_x))
        v_mean = self._split(self.v_mean(mean_x))
        q_cov = self._split(self.q_cov(cov_x))
        k_cov = self._split(self.k_cov(cov_x))
        v_cov = self._split(self.v_cov(cov_x))

        dist = wasserstein_distance_matmul(q_mean, q_cov, k_mean, k_cov)
        scores = -dist / math.sqrt(self.head_dim)

        scores = scores.masked_fill(mask == 0, -1e32)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        seq_len = attn.shape[-1]
        pos = torch.arange(seq_len, device=attn.device)
        pos_dist = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)).float()
        pos_dist = pos_dist.unsqueeze(0).unsqueeze(0)
        gamma = -torch.nn.functional.softplus(self.gamma).unsqueeze(0)
        pos_effect = torch.clamp((gamma * pos_dist).exp(), min=1e-5, max=1e5)
        attn = attn * pos_effect
        attn = attn / torch.clamp(attn.sum(dim=-1, keepdim=True), min=1e-12)

        out_mean = torch.matmul(attn, v_mean)
        out_cov = torch.matmul(attn ** 2, v_cov)
        out_mean = self.out_mean(self._merge(out_mean))
        out_cov = self.out_cov(self._merge(out_cov))
        return out_mean, out_cov


class UKTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = UKTAttention(hidden_size, num_heads, dropout)
        self.dropout = Dropout(dropout)
        self.norm_mean_1 = nn.LayerNorm(hidden_size)
        self.norm_cov_1 = nn.LayerNorm(hidden_size)

        self.ffn_mean = nn.Sequential(
            Linear(hidden_size, d_ff),
            nn.ReLU(),
            Dropout(dropout),
            Linear(d_ff, hidden_size),
        )
        self.ffn_cov = nn.Sequential(
            Linear(hidden_size, d_ff),
            nn.ReLU(),
            Dropout(dropout),
            Linear(d_ff, hidden_size),
        )
        self.norm_mean_2 = nn.LayerNorm(hidden_size)
        self.norm_cov_2 = nn.LayerNorm(hidden_size)
        self.elu = nn.ELU()

    def forward(self, mean_x, cov_x, mask):
        attn_mean, attn_cov = self.attn(mean_x, cov_x, mask)
        mean_x = self.norm_mean_1(mean_x + self.dropout(attn_mean))
        cov_x = self.norm_cov_1(self.elu(cov_x + self.dropout(attn_cov)) + 1.0)

        ff_mean = self.ffn_mean(mean_x)
        ff_cov = self.ffn_cov(cov_x)
        mean_x = self.norm_mean_2(mean_x + self.dropout(ff_mean))
        cov_x = self.norm_cov_2(self.elu(cov_x + self.dropout(ff_cov)) + 1.0)
        return mean_x, cov_x


class UKT(Module):
    def __init__(
        self,
        num_q,
        emb_size,
        hidden_size,
        num_attn_heads,
        dropout,
        num_blocks=2,
        d_ff=256,
        use_cl=True,
        cl_weight=0.02,
        uncertainty_weight=1e-4
    ):
        super().__init__()
        self.num_q = num_q
        self.use_cl = use_cl
        self.cl_weight = cl_weight
        self.uncertainty_weight = uncertainty_weight

        self.mean_q_embed = Embedding(num_q, emb_size)
        self.cov_q_embed = Embedding(num_q, emb_size)
        self.mean_r_embed = Embedding(2, emb_size)
        self.cov_r_embed = Embedding(2, emb_size)

        self.mean_proj = Linear(emb_size, hidden_size)
        self.cov_proj = Linear(emb_size, hidden_size)
        self.dropout = Dropout(dropout)
        self.elu = nn.ELU()

        self.blocks = nn.ModuleList([
            UKTBlock(hidden_size, num_attn_heads, d_ff, dropout)
            for _ in range(num_blocks)
        ])

        self.out = nn.Sequential(
            Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_size, num_q)
        )
        self.nce = WassersteinNCELoss(temperature=0.1)

    def _encode(self, q, r):
        mean_x = self.mean_q_embed(q) + self.mean_r_embed(r)
        cov_x = self.cov_q_embed(q) + self.cov_r_embed(r)
        mean_x = self.mean_proj(mean_x)
        cov_x = self.elu(self.cov_proj(cov_x)) + 1.0

        seq_len = q.shape[1]
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_mask = (~causal_mask).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            mean_x, cov_x = blk(mean_x, cov_x, attn_mask)
        return mean_x, cov_x

    def _augment_response(self, r):
        noise = torch.bernoulli(torch.full_like(r.float(), 0.1)).long()
        return torch.bitwise_xor(r.long(), noise)

    def forward(self, q, r, train=False):
        mean_x, cov_x = self._encode(q, r)
        logits = self.out(torch.cat([mean_x, cov_x], dim=-1))
        preds = torch.sigmoid(logits)

        aux_losses = {}
        if train and self.use_cl:
            r_aug = self._augment_response(r)
            mean_aug, cov_aug = self._encode(q, r_aug)
            pooled_mean = mean_x.mean(dim=1)
            pooled_cov = cov_x.mean(dim=1)
            pooled_mean_aug = mean_aug.mean(dim=1)
            pooled_cov_aug = cov_aug.mean(dim=1)
            aux_losses["cl"] = self.nce(
                pooled_mean, pooled_cov, pooled_mean_aug, pooled_cov_aug
            ) * self.cl_weight

        aux_losses["uncertainty"] = cov_x.mean() * self.uncertainty_weight
        return preds, aux_losses
