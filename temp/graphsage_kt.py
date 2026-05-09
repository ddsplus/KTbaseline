"""GraphSAGE_KT 模型（中文注释版）

文件说明：
- 本文件实现一个基于 GraphSAGE + 序列编码的知识追踪（Knowledge Tracing）模型。
- 流程：先在题目图（question-question）上运行 GraphSAGE，得到题目级别的向量表示；
    然后将每个时间步的题目表示与该时间步的答题标签拼接，输入到序列编码器（例如 LSTM），
    最后用预测层计算下一题的答对概率。

设计约定（重要）：
- 该实现与外部 `loader.load_data(args)` 返回的数据结构兼容。构造时需要传入 `args` 与 `data`：
    - `data['num_ques']` : int，题目数量（题目 id 假定为 0..num_ques-1）。
    - （已移除对外部 node2vec 预训练嵌入的依赖，题目特征使用可训练的 `nn.Embedding`）

接口（示例）：
    model = GraphSAGE_KT(args, data)
    pack_pred = model(seq_lens, pad_curr, pad_answer, pad_next)

参数与张量约定：
- `seq_lens` : torch.LongTensor，shape `[batch]`，表示每个样本序列中用于预测的时间步数（通常是原始序列长度 - 1）。
- `pad_curr`, `pad_answer`, `pad_next` : 已按最大长度 pad 的张量，shape `[seq_len, batch]`，对应当前题目 id、当前答案(0/1)、下一题 id。
- 返回值为 `PackedSequence`，其中 `.data` 包含按时间展平的预测概率（float，0..1）。

实现要点与性能建议：
- GraphSAGE 的邻接矩阵以稀疏 COO 格式构造，尽量在 CPU 上使用向量化运算（例如 bincount）以减少
    GPU 上的不必要内存分配与复制。对于包含大量技能—题目 pairwise 连接的数据集，务必注意边数量可能呈
    O(N^2) 增长，必要时限制每个技能的连边数或采用近邻/采样策略。

维护与扩展提示：
- 可通过替换 `SequenceEncoder` 中的 RNN 为 Transformer 来尝试更强的序列建模能力；
- 若需要使用外部预训练题目嵌入，可在外部加载并在构造 `GraphSAGE_KT` 时把初始特征传入（当前实现已移除该分支）。

作者/维护：代码助手（已添加中文注释）
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from typing import List, Tuple, Dict
import json


# ---------------------------
# GraphSAGE 编码器（节点表示学习）
#
# 输入：
# - `features`：节点初始特征矩阵，shape [N, in_dim]
# 输出：
# - 节点表示矩阵，shape [N, hidden_dims[-1]]
#
# 算法要点：
# - 本实现使用邻居平均（row-normalized adjacency）作为聚合函数，
#   将自身表示与邻居平均拼接后通过线性层与 ReLU 更新；这是 GraphSAGE 的一种简化形式。
# - 邻接矩阵以稀疏 COO 存储，构建时尽量在 CPU 上做向量化处理，完成后转移到目标 device。
# ---------------------------
class GraphSAGEEncoder(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, hidden_dims: List[int], edge_index: Tuple[torch.Tensor, torch.Tensor], device: torch.device):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        # build normalized adjacency sparse matrix (row-stochastic: neighbor mean)
        # edge_index: (rows, cols) lists of indices for edges (both directions expected)
        rows, cols = edge_index
        assert rows.dim() == 1 and cols.dim() == 1
        # Move indices to CPU for efficient vectorized processing (avoid Python loops)
        rows_cpu = rows.cpu() if rows.is_cuda else rows
        cols_cpu = cols.cpu() if cols.is_cuda else cols
        # compute degree (per row) using bincount (fast, vectorized)
        deg = torch.bincount(rows_cpu, minlength=num_nodes).float()
        deg[deg == 0] = 1.0
        # values are 1/deg[row] using tensor indexing
        vals = (1.0 / deg[rows_cpu]).float()
        idx = torch.stack([rows_cpu.long(), cols_cpu.long()], dim=0)
        # build sparse adjacency on CPU then move to target device (reduces GPU allocations)
        adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes))
        self.register_buffer('adj', adj.coalesce().to(device))

        dims = [in_dim] + hidden_dims
        self.linears = nn.ModuleList()
        for i in range(len(dims) - 1):
            # concat of self + neighbor => dim = dims[i]*2
            lin = nn.Linear(dims[i] * 2, dims[i + 1])
            self.linears.append(lin)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [N, in_dim]
        x = features
        for lin in self.linears:
            # neighbor mean: adj @ x  (adj is row normalized)
            neigh = torch.sparse.mm(self.adj, x)
            cat = torch.cat([x, neigh], dim=1)
            x = F.relu(lin(cat))
        return x


class SequenceEncoder(nn.Module):
    """序列编码器：支持 Transformer / LSTM / GRU / RNN。

    输入：`x`，shape [seq_len, batch, input_dim]
    可选：`seq_lens`，shape [batch]，用于 Transformer 的 padding mask
    输出：序列上的隐状态 `out`，shape [seq_len, batch, hidden_dim]
    """
    def __init__(self, input_dim: int, hidden_dim: int, mode='transformer', num_layers=2, 
                 nhead=8, dropout=0.1):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if mode == 'transformer':
            # 线性投影将 input_dim 投影到 hidden_dim
            self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=nhead, 
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=False
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # 保留 RNN 支持（向后兼容）
            if mode == 'lstm':
                self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False, dropout=dropout)
            elif mode == 'gru':
                self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=False, dropout=dropout)
            else:
                self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=False, dropout=dropout)

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor = None) -> torch.Tensor:
        # x: [seq_len, batch, input_dim]
        # seq_lens: [batch]，可选，用于计算 padding mask
        
        if self.mode == 'transformer':
            # 投影输入维度
            if self.input_proj is not None:
                x = self.input_proj(x)  # [seq_len, batch, hidden_dim]
            
            seq_len, batch = x.shape[0], x.shape[1]
            device = x.device
            
            # 生成因果掩码（因果性：位置 i 只能看 i 之前的位置，不能看未来）
            # 上三角矩阵（不含对角线）的位置设为 True（表示不可访问）
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            
            # 生成 padding mask（True 表示忽略的位置，[batch, seq_len]）
            if seq_lens is not None:
                pos_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
                src_key_padding_mask = (pos_idx >= seq_lens.unsqueeze(0)).transpose(0, 1)  # [batch, seq_len]
            else:
                src_key_padding_mask = None
            
            # Transformer forward（因果mask + padding mask）
            out = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)  # [seq_len, batch, hidden_dim]
            return out
        else:
            # RNN 模式（向后兼容）
            out, _ = self.rnn(x)  # [seq_len, batch, hidden_dim]
            return out


class GraphSAGE_KT(nn.Module):
    """主模型：GraphSAGE + 序列编码器的知识追踪模型。

    forward 接口：
        `pack_predict = model(seq_lens, pad_curr, pad_answer, pad_next)`

    返回：PackedSequence（含预测概率）
    """

    def __init__(self, args, data: Dict):
        super().__init__()
        # args: should provide device, emb_dim, hidden_dim, rnn_mode, rnn_num_layer, exercise_dim
        device = torch.device(args.device)
        self.device = device
        # 用于跟踪前向调用次数（用于调试/可视化文件命名）
        self._forward_calls = 0

        # number of questions
        if 'num_ques' in data:
            self.num_questions = int(data['num_ques'])
        else:
            raise RuntimeError('data must contain num_ques')

        # initial question features: use learnable embedding only (pre-trained embeddings removed)
        feat_dim = getattr(args, 'feat_dim', args.emb_dim)
        self.initial_features = None
        #？？？
        self.learn_emb = nn.Embedding(self.num_questions, feat_dim)

        # Build question-question adjacency from dataset files (train sequences + question-skill pk)
        data_dir = os.path.join(args.data_path, args.dataset)
        max_qid = self.num_questions - 1
        edge_rows, edge_cols = build_question_question_edges(data_dir, max_qid, only_train=True)
        # add self-loops
        n = self.num_questions
        rows = torch.tensor(edge_rows, dtype=torch.long)
        cols = torch.tensor(edge_cols, dtype=torch.long)
        # ensure within range
        valid_mask = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
        rows = rows[valid_mask]; cols = cols[valid_mask]
        # add self loops
        self_loop_idx = torch.arange(0, n, dtype=torch.long)
        rows = torch.cat([rows, self_loop_idx], dim=0)
        cols = torch.cat([cols, self_loop_idx], dim=0)

        # pass edge indices as CPU tensors; GraphSAGEEncoder will handle device placement efficiently
        self.graphsage = GraphSAGEEncoder(num_nodes=n, in_dim=feat_dim, hidden_dims=[args.emb_dim],
                         edge_index=(rows, cols), device=device)

        # GraphSAGE 表示的降频更新支持（减少每次前向的 GraphSAGE 计算）
        self.gs_update_interval = int(getattr(args, 'gs_update_interval', 1))
        self._gs_forward_calls = 0
        # 缓存 q_emb，用于延迟更新场景
        self.register_buffer('q_emb_cache', torch.zeros((self.num_questions, args.emb_dim)))

        # ---------- 超图（EduLLM 风格）支持（已解耦到 KT/Code/hypergraph.py） ----------
        # 尝试动态加载独立模块 `HypergraphDualChannel`，模块内部负责 student_emb 与 hyper 投影
        try:
            from hypergraph import HypergraphDualChannel
            self.hypergraph = HypergraphDualChannel(
                data_dir, self.num_questions, feat_dim, args.emb_dim,
                device=device,
                update_interval=getattr(args, 'hg_update_interval', 1),
                use_scatter=bool(getattr(args, 'hg_use_scatter', False)),
                student_hist_max=int(getattr(args, 'hg_student_hist_max', 64)),
                student_attn_chunk_size=int(getattr(args, 'hg_student_attn_chunk_size', 32768)),
            )
            if not getattr(self.hypergraph, 'enabled', False):
                self.hypergraph = None
            else:
                # 融合层：把 GraphSAGE 输出与超图输出拼接 -> 投影回 emb_dim
                self.fuse_linear = nn.Linear(2 * args.emb_dim, args.emb_dim)
            print("HypergraphDualChannel 模块加载成功，已启用超图增强")
        except Exception as e:
            print('无法加载超图模块：', e)
            self.hypergraph = None

        # Flow Matching (无条件去噪) 支持（可选）
        self.use_fm = bool(getattr(args, 'use_fm', False))
        self.last_fm_loss = None
        if self.use_fm and self.hypergraph is not None:
            try:
                from fm_denoiser import FlowMatcher
                fm_hidden = getattr(args, 'fm_hidden', args.emb_dim * 2)
                fm_time_dim = getattr(args, 'fm_time_dim', 32)
                self.fm = FlowMatcher(emb_dim=args.emb_dim, hidden_dim=fm_hidden, time_dim=fm_time_dim)
                self.fm.to(device)
                # FM 可视化支持：是否在 forward 中保存去噪前/后的投影图
                try:
                    from scripts.fm_viz import visualize_embeddings
                    self._fm_viz_fn = visualize_embeddings
                except Exception:
                    self._fm_viz_fn = None
                self.fm_viz = bool(getattr(args, 'fm_viz', False))
                self.fm_viz_dir = getattr(args, 'fm_viz_dir', 'fm_viz')
                # 每个 epoch 最多保存的可视化图片数（训练脚本会在每个 epoch 前重置计数器）
                self.fm_viz_per_epoch = int(getattr(args, 'fm_viz_per_epoch', 3))
                self._fm_viz_saved_this_epoch = 0
            except Exception as e:
                print('无法加载 FM 模块：', e)
                self.fm = None
            self.fm_lambda = float(getattr(args, 'fm_lambda', 1.0))
            self.fm_steps = int(getattr(args, 'fm_steps', 4))
            self.fm_drop_rate = float(getattr(args, 'fm_drop_rate', 0.3))
            self.fm_noise = float(getattr(args, 'fm_noise', 0.1))
        else:
            self.fm = None

        # answer embedding (0/1) as small transform
        self.answer_embed = nn.Embedding(2, feat_dim)

        # sequence encoder (default: transformer)
        seq_input_dim = args.emb_dim + feat_dim  # concat question_emb + answer_emb
        seq_mode = getattr(args, 'seq_mode', 'transformer')  # 'transformer' or 'lstm'/'gru'/'rnn'
        if seq_mode == 'transformer':
            # Transformer 参数
            nhead = getattr(args, 'transformer_nhead', 8)
            transformer_layers = getattr(args, 'transformer_layers', 2)
            seq_dropout = getattr(args, 'transformer_dropout', 0.1)
            self.seq_encoder = SequenceEncoder(seq_input_dim, args.hidden_dim, mode='transformer',
                                              num_layers=transformer_layers, nhead=nhead, 
                                              dropout=seq_dropout)
        else:
            # RNN 兼容模式
            rnn_layers = getattr(args, 'rnn_num_layer', 1)
            rnn_dropout = getattr(args, 'rnn_dropout', 0.0)
            self.seq_encoder = SequenceEncoder(seq_input_dim, args.hidden_dim, mode=seq_mode,
                                              num_layers=rnn_layers, dropout=rnn_dropout)

        # prediction modules
        self.h2y = nn.Linear(args.hidden_dim + args.emb_dim, args.exercise_dim)
        self.y2o = nn.Linear(args.exercise_dim, 1)

        self.to(device)

    def forward(self, seq_lens, pad_curr, pad_answer, pad_next):
        # 基础题目特征（来自 learnable embedding）
        q_feats = self.learn_emb.weight  # [Q, feat_dim]
        # 增加前向调用计数，供可视化/调试使用
        try:
            self._forward_calls += 1
        except Exception:
            self._forward_calls = 1

        # 如果有独立的超图模块，则使用它来得到 hyper 投影并与 GraphSAGE 输出融合
        # 下面计算 GraphSAGE 与（可选的）超图输出，均支持降频更新缓存以提高效率
        device = q_feats.device

        # GraphSAGE 表示（支持降频更新）
        if self.gs_update_interval <= 1:
            q_emb = self.graphsage(q_feats)  # 每步更新
        else:
            if (self._gs_forward_calls % self.gs_update_interval) == 0:
                q_emb_new = self.graphsage(q_feats)
                try:
                    self.q_emb_cache.copy_(q_emb_new.detach())
                except Exception:
                    self.q_emb_cache = q_emb_new.detach()
                q_emb = q_emb_new
            else:
                q_emb = self.q_emb_cache.to(device)
            self._gs_forward_calls += 1

        # 当前 mini-batch 涉及的题目 id（用于子图超图计算）
        seq_len = pad_curr.size(0)
        arange = torch.arange(seq_len, device=device).unsqueeze(1)
        valid_mask = arange < seq_lens.unsqueeze(0)
        if valid_mask.any():
            batch_curr_q = pad_curr[valid_mask].long()
            batch_next_q = pad_next[valid_mask].long()
            batch_qids = torch.unique(torch.cat([batch_curr_q, batch_next_q], dim=0))
            valid_q = (batch_qids >= 0) & (batch_qids < self.num_questions)
            batch_qids = batch_qids[valid_q]
        else:
            batch_qids = torch.zeros((0,), device=device, dtype=torch.long)

        # 如果有超图模块，则先计算 hyper 投影（用于后续 FM 或早融合）
        hyper_proj_sub = None
        if getattr(self, 'hypergraph', None) is not None:
            try:
                if batch_qids.numel() > 0:
                    hyper_proj_sub = self.hypergraph(q_feats, q_ids=batch_qids)  # [B, emb_dim]
            except Exception as e:
                print('超图前向失败，回退到 GraphSAGE：', e)

        final_q_emb = q_emb

        # ----- FM 无条件去噪（先对超图去噪，再早融合） -----
        self.last_fm_loss = None
        fm_applied = False
        if getattr(self, 'fm', None) is not None and getattr(self, 'hypergraph', None) is not None and valid_mask.any():
            try:
                curr_qids = pad_curr[valid_mask].long()
                next_qids = pad_next[valid_mask].long()
                used_qids = torch.unique(torch.cat([curr_qids, next_qids], dim=0))
                valid_used = (used_qids >= 0) & (used_qids < self.num_questions)
                used_qids = used_qids[valid_used]

                if used_qids.numel() > 0:
                    drop_rate = self.fm_drop_rate if self.training else 0.0
                    x0_hyper_sub = self.hypergraph.compute_proj_with_member_dropout(
                        q_feats,
                        drop_rate=drop_rate,
                        q_ids=used_qids,
                    )
                    x1_hyper_sub = None
                    if hyper_proj_sub is not None and batch_qids.numel() > 0:
                        qid_to_local = torch.full((self.num_questions,), -1, device=device, dtype=torch.long)
                        qid_to_local[batch_qids] = torch.arange(batch_qids.numel(), device=device, dtype=torch.long)
                        used_local = qid_to_local[used_qids]
                        if (used_local >= 0).all():
                            x1_hyper_sub = hyper_proj_sub[used_local]
                    if x1_hyper_sub is None:
                        x1_hyper_sub = self.hypergraph(q_feats, q_ids=used_qids)

                    denoised_hyper_sub = self.fm.integrate(x0_hyper_sub, steps=self.fm_steps)

                    final_q_emb = q_emb.clone()
                    final_q_emb[used_qids] = self.fuse_linear(
                        torch.cat([q_emb[used_qids], denoised_hyper_sub], dim=-1)
                    )
                    fm_applied = True

                    # 训练损失按有效 next 位置对齐，保持与预测目标一致
                    if next_qids.numel() > 0:
                        qid_to_local_used = torch.full((self.num_questions,), -1, device=device, dtype=torch.long)
                        qid_to_local_used[used_qids] = torch.arange(used_qids.numel(), device=device, dtype=torch.long)
                        next_local = qid_to_local_used[next_qids]
                        valid_local = next_local >= 0
                        if valid_local.any():
                            x0_loss = x0_hyper_sub[next_local[valid_local]]
                            x1_loss = x1_hyper_sub[next_local[valid_local]]
                            self.last_fm_loss = self.fm.flow_matching_loss(x0_loss, x1_loss)

                            # 训练阶段可视化：保存 x0（noisy）和 x1（clean）对比
                            if self.training and getattr(self, 'fm_viz', False) and getattr(self, '_fm_viz_fn', None) is not None:
                                try:
                                    per_epoch = int(getattr(self, 'fm_viz_per_epoch', 3))
                                    saved_count = int(getattr(self, '_fm_viz_saved_this_epoch', 0))
                                    if saved_count < per_epoch:
                                        vis_n = min(1024, x0_loss.shape[0])
                                        vis_x0 = x0_loss[:vis_n]  # noisy 版本
                                        vis_x1 = x1_loss[:vis_n]  # clean 版本
                                        out_path = os.path.abspath(os.path.join(self.fm_viz_dir, f'fm_train_{self._forward_calls}_{saved_count}.png'))
                                        _ = self._fm_viz_fn(vis_x0, vis_x1, out_path=out_path)
                                        self._fm_viz_saved_this_epoch = saved_count + 1
                                except Exception:
                                    pass
            except Exception as e:
                print('FM 前置去噪失败，继续使用原始题目表示：', e)

        # 如果未进行 FM，则按超图原始投影做早融合
        if getattr(self, 'hypergraph', None) is not None and not fm_applied:
            try:
                if hyper_proj_sub is not None and batch_qids.numel() > 0:
                    final_q_emb = q_emb.clone()
                    final_q_emb[batch_qids] = self.fuse_linear(
                        torch.cat([q_emb[batch_qids], hyper_proj_sub], dim=-1)
                    )
                else:
                    final_q_emb = q_emb
            except Exception as e:
                print('超图融合失败，回退到 GraphSAGE：', e)
                final_q_emb = q_emb

        denoised_q_emb = final_q_emb

        # 序列编码器输入使用去噪后的题目表示
        curr_emb = F.embedding(pad_curr, denoised_q_emb)  # [seq, batch, emb_dim]
        next_emb = F.embedding(pad_next, denoised_q_emb)  # [seq, batch, emb_dim]

        ans_emb = F.embedding(pad_answer.long(), self.answer_embed.weight)  # [seq, batch, feat_dim]

        interact = torch.cat([curr_emb, ans_emb], dim=-1)  # [seq, batch, emb_dim+feat_dim]

        # 序列编码器前向（传递 seq_lens 用于 Transformer 的 padding mask）
        ks_emb = self.seq_encoder(interact, seq_lens=seq_lens)  # [seq, batch, hidden_dim]

        y = F.relu(self.h2y(torch.cat((ks_emb, next_emb), -1)))
        prediction = torch.sigmoid(self.y2o(y)).squeeze(-1)

        pack_predict = pack_padded_sequence(prediction, seq_lens.cpu().long(), enforce_sorted=True)
        return pack_predict


def build_question_question_edges(data_dir: str, max_qid: int, only_train: bool = True) -> Tuple[List[int], List[int]]:
    """
    Build question-question edge lists from:
      - train_ques.txt sequences: connect consecutive questions in each student sequence
      - kg_pk.edgelist: connect questions that share the same skill (pairwise)
    Returns (rows, cols) lists (both directions added later by caller if needed)
    """
    rows = []
    cols = []

    train_q = os.path.join(data_dir, 'train_ques.txt')
    if os.path.exists(train_q):
        with open(train_q, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip() != '']
        i = 0
        while i < len(lines):
            try:
                seq_len = int(lines[i])
            except Exception:
                break
            if i + 1 < len(lines):
                qs = [int(x) for x in lines[i + 1].split(',') if x != '']
                # connect consecutive pairs
                for a, b in zip(qs[:-1], qs[1:]):
                    if a <= max_qid and b <= max_qid:
                        rows.append(a); cols.append(b)
                        rows.append(b); cols.append(a)
            i += 3

    # skill-based edges (kg_pk.edgelist): group questions by skill
    kg_pk = os.path.join(data_dir, 'kg_pk.edgelist')
    if os.path.exists(kg_pk):
        skill2qs = {}
        with open(kg_pk, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                a, b = [int(x) for x in s.split(',')[:2]]
                # keep question ids <= max_qid
                if a <= max_qid and b <= max_qid:
                    # case both are questions (rare)
                    q1, q2 = a, b
                    rows.append(q1); cols.append(q2)
                    rows.append(q2); cols.append(q1)
                else:
                    # assume one is question, the other skill (skill id > max_qid)
                    if a <= max_qid:
                        q, skill = a, b
                    elif b <= max_qid:
                        q, skill = b, a
                    else:
                        continue
                    skill2qs.setdefault(skill, []).append(q)
        # connect questions that share a skill (pairwise)
        for skill, qlist in skill2qs.items():
            if len(qlist) <= 1:
                continue
            # connect all pairs (undirected)
            for i in range(len(qlist)):
                for j in range(i + 1, len(qlist)):
                    a = qlist[i]; b = qlist[j]
                    rows.append(a); cols.append(b)
                    rows.append(b); cols.append(a)

    return rows, cols


if __name__ == '__main__':
    print('This module defines GraphSAGE_KT model. Import and use in your training script.')
