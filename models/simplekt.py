import math

import torch
from torch import nn


class Dim:
    batch = 0
    seq = 1
    feature = 2


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(self._build_pe(max_len), requires_grad=False)

    def _build_pe(self, max_len):
        pe = 0.1 * torch.randn(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-(math.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _ensure_capacity(self, seq_len):
        if self.weight.size(1) >= seq_len:
            return
        new_weight = self._build_pe(seq_len).to(self.weight.device)
        self.weight = nn.Parameter(new_weight, requires_grad=False)

    def forward(self, x):
        self._ensure_capacity(x.size(Dim.seq))
        return self.weight[:, :x.size(Dim.seq), :]


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, memory):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, _ = self.attn(query=x, key=x, value=memory, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
        # SimpleKT constructs [first_step + shifted_steps], so effective
        # sequence length is seq_len + 1.
        self.position_emb = CosinePositionalEmbedding(d_model=d_model, max_len=seq_len + 1)

    def forward(self, q_embed_data, qa_embed_data):
        q_embed_data = q_embed_data + self.position_emb(q_embed_data)
        qa_embed_data = qa_embed_data + self.position_emb(qa_embed_data)

        x = q_embed_data
        memory = qa_embed_data
        for blk in self.blocks:
            x = blk(x, memory)
        return x


class SimpleKT(nn.Module):
    def __init__(
        self,
        num_q,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        num_attn_heads=8,
        seq_len=200,
        final_fc_dim=512,
        final_fc_dim2=256,
        separate_qa=False,
        **kwargs
    ):
        super().__init__()
        self.num_q = num_q
        self.separate_qa = separate_qa

        self.q_embed = nn.Embedding(num_q, d_model)
        if separate_qa:
            self.qa_embed = nn.Embedding(2 * num_q, d_model)
        else:
            self.qa_embed = nn.Embedding(2, d_model)

        self.model = Architecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            seq_len=seq_len,
        )

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim2, 1),
        )

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.num_q * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q, r, qshft=None, qtest=False, train=False):
        q = q.long()
        r = r.long()
        if qshft is None:
            qshft = torch.cat([q[:, 1:], q[:, -1:]], dim=1)
        qshft = qshft.long()
        # Next-step prediction:
        # - query comes from next questions (qshft)
        # - memory comes from past interactions (q, r)
        # This avoids leaking the current-step response into the same-step target.
        query_embed = self.q_embed(qshft)
        _, qa_embed_data = self.base_emb(q, r)
        d_output = self.model(query_embed, qa_embed_data)

        concat_q = torch.cat([d_output, query_embed], dim=-1)
        preds = torch.sigmoid(self.out(concat_q).squeeze(-1))

        if train:
            return preds, 0, 0
        if qtest:
            return preds, concat_q
        return preds
