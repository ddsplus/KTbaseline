"""hypergraph.py

轻量级的双通道超图编码器（EduLLM 风格），与主模型解耦。

功能：
- 从预处理产生的 JSON 文件加载超边成员（仅基于训练用户）
- 注册用于向量化聚合的索引/度为 buffer，以便随模型迁移到 device
- 提供前向接口 `forward(q_feats)`：输入题目初始特征（[Q, feat_dim]），输出超图投影表示 [Q, emb_dim]

实现要点：
- 模块自带 `student_emb`（训练学生嵌入），并在没有文件时自动禁用（enabled=False）
- 聚合使用 `index_add_` 向量化实现，避免在 Python 层循环
"""
from __future__ import annotations
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HypergraphDualChannel(nn.Module):
    """双通道超图编码器（正/负），返回对题目的投影表示。

    构造参数：
    - data_dir: 包含 user_map.json, hg_pos.json, hg_neg.json 的目录
    - num_questions: Q
    - feat_dim: 输入特征维度（题目/学生共享）
    - emb_dim: 输出嵌入维度（与 GraphSAGE emb_dim 对齐）
    - device: 可选 torch.device（模块会在 model.to(device) 时被移动）
    """

    def __init__(self, data_dir: str, num_questions: int, feat_dim: int, emb_dim: int,
                 device: Optional[torch.device] = None, update_interval: int = 1,
                 use_scatter: bool = False, use_attention: bool = True,
                 student_hist_max: int = 64, student_attn_chunk_size: int = 32768):
        super().__init__()
        self.enabled = False
        self.data_dir = data_dir
        self.num_questions = int(num_questions)
        self.feat_dim = int(feat_dim)
        self.emb_dim = int(emb_dim)
        # 学生侧 query-aware 注意力的显存保护参数：限制历史长度并按三元组分块计算
        self.student_hist_max = int(student_hist_max)
        self.student_attn_chunk_size = max(1, int(student_attn_chunk_size))

        user_map_path = os.path.join(data_dir, 'user_map.json')
        hg_pos_path = os.path.join(data_dir, 'hg_pos.json')
        hg_neg_path = os.path.join(data_dir, 'hg_neg.json')

        if not (os.path.exists(user_map_path) and os.path.exists(hg_pos_path) and os.path.exists(hg_neg_path)):
            # 缺文件则禁用超图模块
            return

        try:
            with open(user_map_path, 'r', encoding='utf-8') as f:
                um = json.load(f)
            if isinstance(um, dict) and 'user_map' in um:
                user_map = um['user_map']
                num_students = int(um.get('num_students', len(user_map)))
            else:
                user_map = um
                num_students = int(len(user_map))

            with open(hg_pos_path, 'r', encoding='utf-8') as f:
                hg_pos = json.load(f)
            with open(hg_neg_path, 'r', encoding='utf-8') as f:
                hg_neg = json.load(f)

            # 扁平化为成员索引列表（全局节点 id = Q + student_idx）
            pos_node_idx_list = []
            pos_he_idx_list = []
            neg_node_idx_list = []
            neg_he_idx_list = []
            for q_str, s_list in hg_pos.items():
                qid = int(q_str)
                if s_list:
                    for s in s_list:
                        sid = int(s)
                        pos_node_idx_list.append(self.num_questions + sid)
                        pos_he_idx_list.append(qid)
            for q_str, s_list in hg_neg.items():
                qid = int(q_str)
                if s_list:
                    for s in s_list:
                        sid = int(s)
                        neg_node_idx_list.append(self.num_questions + sid)
                        neg_he_idx_list.append(qid)

            pos_node_idx = torch.tensor(pos_node_idx_list, dtype=torch.long)
            pos_he_idx = torch.tensor(pos_he_idx_list, dtype=torch.long)
            neg_node_idx = torch.tensor(neg_node_idx_list, dtype=torch.long)
            neg_he_idx = torch.tensor(neg_he_idx_list, dtype=torch.long)

            # 每条超边的度（仅统计学生数）
            if pos_he_idx.numel() > 0:
                pos_deg = torch.bincount(pos_he_idx, minlength=self.num_questions).float()
            else:
                pos_deg = torch.zeros(self.num_questions, dtype=torch.float)

            if neg_he_idx.numel() > 0:
                neg_deg = torch.bincount(neg_he_idx, minlength=self.num_questions).float()
            else:
                neg_deg = torch.zeros(self.num_questions, dtype=torch.float)

            # 注册 buffer，随 model.to(device) 一起移动
            self.register_buffer('pos_node_idx', pos_node_idx)
            self.register_buffer('pos_he_idx', pos_he_idx)
            self.register_buffer('pos_deg', pos_deg)
            self.register_buffer('neg_node_idx', neg_node_idx)
            self.register_buffer('neg_he_idx', neg_he_idx)
            self.register_buffer('neg_deg', neg_deg)

            # Rasch 参数缓存：题目难度 B_i 与学生能力 theta_n
            eps = 1e-4
            item_total = pos_deg + neg_deg
            item_total[item_total == 0] = 1.0
            item_rate = (pos_deg / item_total).clamp(eps, 1.0 - eps)
            rasch_item_difficulty = -torch.log(item_rate / (1.0 - item_rate))

            pos_sid = pos_node_idx - self.num_questions
            neg_sid = neg_node_idx - self.num_questions
            if pos_sid.numel() > 0:
                pos_cnt = torch.bincount(pos_sid, minlength=num_students).float()
            else:
                pos_cnt = torch.zeros((num_students,), dtype=torch.float)
            if neg_sid.numel() > 0:
                neg_cnt = torch.bincount(neg_sid, minlength=num_students).float()
            else:
                neg_cnt = torch.zeros((num_students,), dtype=torch.float)

            stu_total = pos_cnt + neg_cnt
            stu_total[stu_total == 0] = 1.0
            stu_rate = (pos_cnt / stu_total).clamp(eps, 1.0 - eps)
            rasch_student_ability = torch.log(stu_rate / (1.0 - stu_rate))

            # 中心化以缓解尺度不一致
            rasch_item_difficulty = rasch_item_difficulty - rasch_item_difficulty.mean()
            rasch_student_ability = rasch_student_ability - rasch_student_ability.mean()

            self.register_buffer('rasch_item_difficulty', rasch_item_difficulty)
            self.register_buffer('rasch_student_ability', rasch_student_ability)

            # student -> history questions 映射（按 student 排序，便于向量化分组）
            pos_hist_sid = pos_node_idx - self.num_questions
            neg_hist_sid = neg_node_idx - self.num_questions

            if pos_hist_sid.numel() > 0:
                pos_order = torch.argsort(pos_hist_sid)
                pos_hist_sid = pos_hist_sid[pos_order]
                pos_hist_qid = pos_he_idx[pos_order]
                if self.student_hist_max > 0:
                    pos_hist_cnt_full = torch.bincount(pos_hist_sid, minlength=num_students)
                    pos_hist_start_full = torch.cumsum(pos_hist_cnt_full, dim=0) - pos_hist_cnt_full
                    pos_local_rank = torch.arange(pos_hist_sid.numel(), dtype=torch.long) - pos_hist_start_full[pos_hist_sid]
                    pos_keep = pos_local_rank < self.student_hist_max
                    pos_hist_sid = pos_hist_sid[pos_keep]
                    pos_hist_qid = pos_hist_qid[pos_keep]
                pos_hist_cnt = torch.bincount(pos_hist_sid, minlength=num_students)
            else:
                pos_hist_sid = torch.zeros((0,), dtype=torch.long)
                pos_hist_qid = torch.zeros((0,), dtype=torch.long)
                pos_hist_cnt = torch.zeros((num_students,), dtype=torch.long)

            if neg_hist_sid.numel() > 0:
                neg_order = torch.argsort(neg_hist_sid)
                neg_hist_sid = neg_hist_sid[neg_order]
                neg_hist_qid = neg_he_idx[neg_order]
                if self.student_hist_max > 0:
                    neg_hist_cnt_full = torch.bincount(neg_hist_sid, minlength=num_students)
                    neg_hist_start_full = torch.cumsum(neg_hist_cnt_full, dim=0) - neg_hist_cnt_full
                    neg_local_rank = torch.arange(neg_hist_sid.numel(), dtype=torch.long) - neg_hist_start_full[neg_hist_sid]
                    neg_keep = neg_local_rank < self.student_hist_max
                    neg_hist_sid = neg_hist_sid[neg_keep]
                    neg_hist_qid = neg_hist_qid[neg_keep]
                neg_hist_cnt = torch.bincount(neg_hist_sid, minlength=num_students)
            else:
                neg_hist_sid = torch.zeros((0,), dtype=torch.long)
                neg_hist_qid = torch.zeros((0,), dtype=torch.long)
                neg_hist_cnt = torch.zeros((num_students,), dtype=torch.long)

            self.register_buffer('pos_hist_sid', pos_hist_sid)
            self.register_buffer('pos_hist_qid', pos_hist_qid)
            self.register_buffer('pos_hist_cnt', pos_hist_cnt)
            self.register_buffer('neg_hist_sid', neg_hist_sid)
            self.register_buffer('neg_hist_qid', neg_hist_qid)
            self.register_buffer('neg_hist_cnt', neg_hist_cnt)

            # 学生嵌入（仅训练用户）
            self.num_students = int(num_students)
            self.student_emb = nn.Embedding(self.num_students, self.feat_dim)

            # 学生侧双通道 query-aware attention：Q/K/V + 拼接融合
            # 第一跳（学生聚合题目历史）的投影矩阵
            self.W_Q = nn.Linear(2 * self.feat_dim, self.feat_dim)
            self.W_K = nn.Linear(self.feat_dim, self.feat_dim)
            self.W_V = nn.Linear(self.feat_dim, self.feat_dim)
            self.student_fuse = nn.Linear(2 * self.feat_dim, self.feat_dim)

            # 第二跳（题目聚合学生）的投影矩阵：仅 K、V，Q 不投影
            self.W_K2 = nn.Linear(self.feat_dim, self.feat_dim)
            self.W_V2 = nn.Linear(self.feat_dim, self.feat_dim)

            # 双通道 Attention-only 拼接 -> 投影层（仅拼接 pos attn 与 neg attn，2*feat_dim -> emb_dim）
            self.hyper_proj = nn.Linear(2 * self.feat_dim, self.emb_dim)
            # 两跳 attention 的通道区分偏置（作用于 key 空间）
            self.h1_attn_key_bias = nn.Embedding(2, self.feat_dim)
            self.h2_attn_key_bias = nn.Embedding(2, self.feat_dim)
            nn.init.zeros_(self.h1_attn_key_bias.weight)
            nn.init.zeros_(self.h2_attn_key_bias.weight)

            # 缓存与更新频率控制：减少每次前向对超图进行全量计算
            self.update_interval = max(1, int(update_interval))
            self._forward_calls = 0
            # hyper 投影缓存（buffer，随 model.to(device) 一起移动）
            self.register_buffer('hyper_proj_cache', torch.zeros((self.num_questions, self.emb_dim)))

            # 尝试导入 torch_scatter 加速（若用户请求且已安装），包括 scatter_max 用于 group-wise max pooling
            self.use_scatter = False
            self._scatter_add = None
            self._scatter_max = None
            if use_scatter:
                try:
                    from torch_scatter import scatter_add, scatter_max
                    self._scatter_add = scatter_add
                    self._scatter_max = scatter_max
                    self.use_scatter = True
                except Exception:
                    self.use_scatter = False
            # attention 聚合开关（可用于消融实验）
            self.use_attention = bool(use_attention)

            self.enabled = True
        except Exception as e:
            print('HypergraphDualChannel 加载失败：', e)
            self.enabled = False

    def _student_history_query_attention_channel_active(self, q_feats: torch.Tensor,
                                                        q_current_feats: torch.Tensor,
                                                        curr_sid_local: torch.Tensor, curr_he_local: torch.Tensor,
                                                        hist_sid: torch.Tensor, hist_qid: torch.Tensor,
                                                        sid_to_local: torch.Tensor,
                                                        s_emb_active: torch.Tensor,
                                                        channel_idx: int = 0) -> torch.Tensor:
        """学生聚合题目历史的单通道 query-aware attention（第一跳）。
        
        核心逻辑（标准 KT attention）：
        1. 构造边级 query：q_{u,q} = W_Q([x_q || e_u])，其中 x_q 是题目 embedding，e_u 是学生 embedding
        2. 按学生 MAX pooling 聚合得到 per-student query
        3. 使用 per-student query 在历史题目上做 attention
        4. 输出：per-student 的历史聚合表示
        
        复杂度：O(E_current·D + E_history·D)，完全保持稀疏计算
        
        参数：
        - q_feats: [Q, D] 全量题目特征（用于历史 lookup）
        - q_current_feats: [B, D] 当前子图题目特征（用于 query 构造）
        - curr_sid_local: [E_curr] 当前边的学生本地索引
        - curr_he_local: [E_curr] 当前边的题目本地索引
        - hist_sid: [E_hist] 历史边的学生 id
        - hist_qid: [E_hist] 历史边的题目 id
        - sid_to_local: [S] 全局学生 id → 本地索引的映射
        - s_emb_active: [A, D] 活跃学生的 embedding
        
        返回: [A, D] 活跃学生的历史聚合表示
        """
        device = q_feats.device
        A = int(s_emb_active.size(0))  # 活跃学生数
        D = q_feats.size(1)  # 特征维度
        scale = float(D) ** 0.5  # attention scale

        # 初始化输出：[A, D]
        out = torch.zeros((A, D), device=device, dtype=q_feats.dtype)
        
        # 边界条件检查
        if A == 0 or curr_he_local.numel() == 0 or hist_qid.numel() == 0:
            return out

        # 设备同步
        curr_sid_local_dev = curr_sid_local if curr_sid_local.device == device else curr_sid_local.to(device)
        curr_he_local_dev = curr_he_local if curr_he_local.device == device else curr_he_local.to(device)
        hist_sid_dev = hist_sid if hist_sid.device == device else hist_sid.to(device)
        hist_qid_dev = hist_qid if hist_qid.device == device else hist_qid.to(device)
        sid_to_local_dev = sid_to_local if sid_to_local.device == device else sid_to_local.to(device)

        # ========== 第一步：构造 per-student query（question-aware）==========
        # 统计每个学生在当前子图中有多少条边（用于 has_curr mask）
        curr_cnt = torch.bincount(curr_sid_local_dev, minlength=A).to(device=device, dtype=q_feats.dtype)
        has_curr = curr_cnt > 0  # [A] 布尔 mask：哪些学生有当前题目

        # 映射历史学生到本地索引
        hist_sid_local = sid_to_local_dev[hist_sid_dev]
        keep = hist_sid_local >= 0  # 只保留活跃学生的历史
        if keep.any():
            hist_sid_local = hist_sid_local[keep]
            hist_qid_dev = hist_qid_dev[keep]
        else:
            return out

        # 只保留具有当前题目的学生的历史
        keep_curr = has_curr[hist_sid_local]
        if keep_curr.any():
            hist_sid_local = hist_sid_local[keep_curr]
            hist_qid_dev = hist_qid_dev[keep_curr]
        else:
            return out

        # ========== 第二步：边级 query 构造 ==========
        # Q：当前题目特征 [E_curr, D]
        curr_q_emb = q_current_feats[curr_he_local_dev]
        
        # Q：边级学生特征 [E_curr, D]
        curr_s_emb = s_emb_active[curr_sid_local_dev]
        
        # 拼接：[E_curr, 2D]
        query_concat = torch.cat([curr_q_emb, curr_s_emb], dim=-1)
        
        # 投影得到边级 query：[E_curr, D]
        query_edges = self.W_Q(query_concat)
        
        # ========== 第三步：按学生 MAX pooling 聚合 query ==========
        q_query_active = torch.zeros((A, D), device=device, dtype=q_feats.dtype)
        try:
            # 优先使用 scatter_reduce mean（若 PyTorch 支持），对边级 query 做均值池化
            q_query_active.scatter_reduce_(0, curr_sid_local_dev.unsqueeze(1).expand(-1, D),
                                          query_edges, reduce='mean')
        except Exception:
            # fallback 1：使用 torch_scatter 的 add（求和），再除以计数得到均值
            _scatter_add = getattr(self, '_scatter_add', None)
            if _scatter_add is not None:
                try:
                    q_query_active = _scatter_add(query_edges, curr_sid_local_dev, dim=0, dim_size=A)
                    q_query_active = q_query_active / curr_cnt.unsqueeze(1).clamp_min(1.0)
                except Exception:
                    # fallback 2：应急用 index_add_ + 除以计数
                    q_query_active = torch.zeros((A, D), device=device, dtype=q_feats.dtype)
                    q_query_active.index_add_(0, curr_sid_local_dev, query_edges)
                    q_query_active = q_query_active / curr_cnt.unsqueeze(1).clamp_min(1.0)
        
        # 取出每条历史边对应的 per-student query：[E_hist, D]
        q_query = q_query_active[hist_sid_local]
        
        # ========== 第四步：历史 attention 计算 ==========
        # K：历史题目特征 [E_hist, D]
        q_history = q_feats[hist_qid_dev]
        ch_bias = self.h1_attn_key_bias.weight[int(channel_idx)].to(device=device, dtype=q_feats.dtype)
        q_key = self.W_K(q_history) + ch_bias.unsqueeze(0)  # [E_hist, D] 投影后的 key + 通道偏置
        q_value = self.W_V(q_history)  # [E_hist, D] 投影后的 value
        
        # 计算 attention 分数：(Q · K) / scale → [E_hist]
        score = (q_query * q_key).sum(dim=-1) / scale

        # 获取聚合工具
        _scatter_add = getattr(self, '_scatter_add', None)
        _scatter_max = getattr(self, '_scatter_max', None)
        use_scatter = getattr(self, 'use_scatter', False)

        # ========== 第五步：softmax 归一化（数值稳定） ==========
        # 按学生统计最大分数（用于稳定指数）
        max_vals = None
        if use_scatter and _scatter_max is not None:
            try:
                res = _scatter_max(score, hist_sid_local, dim=0, dim_size=A)
                max_vals = res[0] if isinstance(res, (tuple, list)) else res
            except Exception:
                max_vals = None
        if max_vals is None:
            try:
                max_vals = torch.full((A,), float('-inf'), device=device, dtype=score.dtype)
                max_vals.scatter_reduce_(0, hist_sid_local, score, reduce='amax')
            except Exception:
                max_vals = None

        # exp(score - max) for stability
        if max_vals is not None:
            exp_score = (score - max_vals[hist_sid_local]).exp()
        else:
            exp_score = score.exp()

        # 计算权重分母：sum(exp) 按学生分组
        if use_scatter and _scatter_add is not None:
            try:
                sum_exp = _scatter_add(exp_score, hist_sid_local, dim=0, dim_size=A)
            except Exception:
                sum_exp = torch.zeros((A,), device=device, dtype=exp_score.dtype)
                sum_exp.index_add_(0, hist_sid_local, exp_score)
        else:
            sum_exp = torch.zeros((A,), device=device, dtype=exp_score.dtype)
            sum_exp.index_add_(0, hist_sid_local, exp_score)

        # softmax 权重：α = exp(score) / sum(exp)
        alpha = exp_score / sum_exp[hist_sid_local].clamp_min(1e-6)
        
        # ========== 第六步：加权聚合 ==========
        weighted = alpha.unsqueeze(-1) * q_value  # [E_hist, D]

        # 按学生聚合加权值
        if use_scatter and _scatter_add is not None:
            try:
                out = _scatter_add(weighted, hist_sid_local, dim=0, dim_size=A)
            except Exception:
                out.index_add_(0, hist_sid_local, weighted)
        else:
            out.index_add_(0, hist_sid_local, weighted)

        # 对没有当前题目的学生置零
        return out * has_curr.to(dtype=q_feats.dtype).unsqueeze(1)

    def _group_attention_pool_local(self, q_sub: torch.Tensor, s_active: torch.Tensor,
                                    sid_local: torch.Tensor, he_local: torch.Tensor,
                                    num_q: int, channel_idx: int = 0) -> torch.Tensor:
        """子图上的题目聚合学生 attention pooling（第二跳）。
        
        计算逻辑：
        - Query（题目特征）：直接使用，不投影
        - Key/Value（学生特征）：通过 W_K2、W_V2 投影
        
        参数：
        - q_sub: [B, D] 子图题目特征
        - s_active: [A, D] 活跃学生特征
        - sid_local: [E] 边级学生本地索引
        - he_local: [E] 边级题目本地索引
        - num_q: 子图题目数 B
        
        返回: [B, D] 聚合后的题目表示
        """
        device = q_sub.device
        D = q_sub.size(1)
        scale = float(D) ** 0.5

        # 初始化输出张量：[B, D]
        out = torch.zeros((num_q, D), device=device, dtype=q_sub.dtype)
        
        # 空边处理
        if he_local.numel() == 0:
            return out

        # 设备同步
        sid_local_dev = sid_local if sid_local.device == device else sid_local.to(device)
        he_local_dev = he_local if he_local.device == device else he_local.to(device)
        
        # 提取边级特征
        # src: [E, D] 边级学生特征
        src = s_active[sid_local_dev]
        
        # q_for_members: [E, D] 边级题目特征（Query，不投影）
        q_for_members = q_sub[he_local_dev]
        
        # 计算注意力分数：[E]
        # score = (Q · K) / scale = (q_for_members · W_K2(src)) / scale
        ch_bias = self.h2_attn_key_bias.weight[int(channel_idx)].to(device=device, dtype=q_sub.dtype)
        src_key = self.W_K2(src) + ch_bias.unsqueeze(0)  # [E, D] 投影后的学生 key + 通道偏置
        score = (q_for_members * src_key).sum(dim=-1) / scale
        
        # 获取聚合工具
        _scatter_add = getattr(self, '_scatter_add', None)
        _scatter_max = getattr(self, '_scatter_max', None)
        use_scatter = getattr(self, 'use_scatter', False)

        # 计算组内最大值（数值稳定性）：[B]
        max_vals = None
        if use_scatter and _scatter_max is not None:
            try:
                res = _scatter_max(score, he_local_dev, dim=0, dim_size=num_q)
                max_vals = res[0] if isinstance(res, (tuple, list)) else res
            except Exception:
                max_vals = None
        
        if max_vals is None:
            try:
                max_vals = torch.full((num_q,), float('-inf'), device=device, dtype=score.dtype)
                max_vals.scatter_reduce_(0, he_local_dev, score, reduce='amax')
            except Exception:
                max_vals = None

        # 计算注意力权重（使用 max 技巧保证数值稳定性）
        if max_vals is not None:
            exp_score = (score - max_vals[he_local_dev]).exp()  # [E]
        else:
            exp_score = score.exp()

        # 计算权重和：sum(exp(score)) 按题目分组 → [B]
        if use_scatter and _scatter_add is not None:
            try:
                sum_exp = _scatter_add(exp_score, he_local_dev, dim=0, dim_size=num_q)
            except Exception:
                sum_exp = torch.zeros((num_q,), device=device, dtype=exp_score.dtype)
                sum_exp.index_add_(0, he_local_dev, exp_score)
        else:
            sum_exp = torch.zeros((num_q,), device=device, dtype=exp_score.dtype)
            sum_exp.index_add_(0, he_local_dev, exp_score)

        # 注意力权重：α = exp(score) / sum_exp → [E]
        alpha = exp_score / sum_exp[he_local_dev].clamp_min(1e-6)
        
        # 投影学生值并加权：[E, D] = α * W_V2(src)
        src_value = self.W_V2(src)  # [E, D] 投影后的学生 value
        weighted = src_value * alpha.unsqueeze(1)  # [E, D]

        # 按题目聚合加权学生特征：[B, D]
        if use_scatter and _scatter_add is not None:
            try:
                out = _scatter_add(weighted, he_local_dev, dim=0, dim_size=num_q)
            except Exception:
                out.index_add_(0, he_local_dev, weighted)
        else:
            out.index_add_(0, he_local_dev, weighted)
        
        return out

    def _group_max_pool_local(self, s_active: torch.Tensor,
                              sid_local: torch.Tensor, he_local: torch.Tensor,
                              num_q: int) -> torch.Tensor:
        """子图上的题目聚合学生 max pooling（辅助函数，目前未使用）。
        
        逻辑：按题目对学生特征做 element-wise max aggregation
        返回: [B, D] 聚合后的题目表示
        """
        device = s_active.device
        D = s_active.size(1)
        out = torch.zeros((num_q, D), device=device, dtype=s_active.dtype)
        
        if he_local.numel() == 0:
            return out

        # 设备同步
        sid_local_dev = sid_local if sid_local.device == device else sid_local.to(device)
        he_local_dev = he_local if he_local.device == device else he_local.to(device)
        src = s_active[sid_local_dev]  # [E, D]

        # 尝试使用 torch_scatter 的 max
        _scatter_max = getattr(self, '_scatter_max', None)
        use_scatter = getattr(self, 'use_scatter', False)
        if use_scatter and _scatter_max is not None:
            try:
                res = _scatter_max(src, he_local_dev, dim=0, dim_size=num_q)
                out = res[0] if isinstance(res, (tuple, list)) else res
                return out
            except Exception:
                pass

        # fallback：使用 scatter_reduce amax
        try:
            out = torch.full((num_q, D), float('-inf'), device=device, dtype=s_active.dtype)
            idx = he_local_dev.unsqueeze(1).expand(-1, D)
            out.scatter_reduce_(0, idx, src, reduce='amax')
            out[out == float('-inf')] = 0.0
            return out
        except Exception:
            # 最后保底：按题目迭代（可能较慢）
            unique_hes = torch.unique(he_local_dev)
            for h in unique_hes:
                m = (he_local_dev == h)
                if m.any():
                    out[h] = src[m].max(dim=0).values
            return out

    def _normalize_q_ids(self, q_ids, device: torch.device) -> torch.Tensor:
        """规范化题目 id：转为去重的有效张量。
        
        参数：
        - q_ids: int、list、tensor 等多种格式
        - device: 目标设备
        
        返回: [B] 去重后的有效题目 id 张量
        """
        if torch.is_tensor(q_ids):
            ids = q_ids.to(device=device, dtype=torch.long)
        else:
            ids = torch.tensor(q_ids, device=device, dtype=torch.long)
        
        # 扁平化
        ids = ids.reshape(-1)
        if ids.numel() == 0:
            return ids
        
        # 去重
        ids = torch.unique(ids)
        
        # 过滤有效范围 [0, Q)
        valid = (ids >= 0) & (ids < self.num_questions)
        return ids[valid]

    def _compute_hyper_proj_for_qids(self, q_feats: torch.Tensor, q_ids: torch.Tensor,
                                     drop_rate: float = 0.0,
                                     slip_intensity: float = 0.0,
                                     guess_intensity: float = 0.0) -> torch.Tensor:
        """仅在指定题目子图上计算超图投影（核心计算接口）。
        
        整体流程（两跳传播）：
        ========
        第一跳：学生 → 聚合其历史题目 → 得到学生表示（s_pos / s_neg）
        第二跳：题目 ← 聚合其超边中的学生 → 得到题目表示（E_pos_attn / E_neg_attn）
        融合：拼接两通道 → 投影得到最终超图表示
        
        参数：
        - q_feats: [Q, D] 全量题目特征
        - q_ids: 子图题目 id（一维张量或列表）
        - drop_rate: 成员 dropout 概率（0 = 不 dropout）
        - slip_intensity: 失误噪声强度（0-1）
        - guess_intensity: 猜测噪声强度（0-1）
        
        返回: [B, emb_dim] 子图题目的超图投影，B = len(unique(q_ids))
        """
        device = q_feats.device
        Q = self.num_questions
        S = self.num_students
        D = q_feats.size(1)

        # ========== 规范化 q_ids ==========
        q_ids_dev = self._normalize_q_ids(q_ids, device)
        B = int(q_ids_dev.numel())
        if B == 0:
            return torch.zeros((0, self.emb_dim), device=device, dtype=q_feats.dtype)

        # 提取子图题目特征：[B, D]
        q_sub = q_feats[q_ids_dev]
        
        # 创建题目掩码：用于快速过滤边
        q_mask = torch.zeros((Q,), dtype=torch.bool, device=device)
        q_mask[q_ids_dev] = True

        # 获取全量的正负超边
        pos_node_idx = self.pos_node_idx if self.pos_node_idx.device == device else self.pos_node_idx.to(device)
        pos_he_idx = self.pos_he_idx if self.pos_he_idx.device == device else self.pos_he_idx.to(device)
        neg_node_idx = self.neg_node_idx if self.neg_node_idx.device == device else self.neg_node_idx.to(device)
        neg_he_idx = self.neg_he_idx if self.neg_he_idx.device == device else self.neg_he_idx.to(device)

        # ========== 通道过滤器：仅保留子图内的边 ==========
        def _filter_channel(node_idx: torch.Tensor, he_idx: torch.Tensor):
            """根据题目掩码和 dropout 过滤超边。"""
            if he_idx.numel() == 0:
                return torch.zeros((0,), device=device, dtype=torch.long), torch.zeros((0,), device=device, dtype=torch.long)
            
            # 保留题目在子图中的边
            keep = q_mask[he_idx]
            
            # 应用 dropout（可选）
            if float(drop_rate) > 0.0:
                keep = keep & (torch.rand(he_idx.shape, device=device) > float(drop_rate))
            
            return node_idx[keep], he_idx[keep]

        # 过滤出子图内的边
        pos_node_sub, pos_he_sub = _filter_channel(pos_node_idx, pos_he_idx)
        neg_node_sub, neg_he_sub = _filter_channel(neg_node_idx, neg_he_idx)

        # ========== Rasch 噪声：失误/猜测翻转 ==========
        if float(slip_intensity) > 0.0 or float(guess_intensity) > 0.0:
            try:
                rasch_item = self.rasch_item_difficulty
                rasch_student = self.rasch_student_ability
                if rasch_item.device != device:
                    rasch_item = rasch_item.to(device)
                if rasch_student.device != device:
                    rasch_student = rasch_student.to(device)

                if float(slip_intensity) > 0.0 and pos_he_sub.numel() > 0:
                    pos_sid_local = pos_node_sub - Q
                    p_pos = torch.sigmoid(rasch_student[pos_sid_local] - rasch_item[pos_he_sub])
                    slip_prob = p_pos * float(slip_intensity)
                    slip_mask = torch.rand(p_pos.shape, device=device) < slip_prob
                    if slip_mask.any():
                        flip_idx = torch.nonzero(slip_mask, as_tuple=True)[0]
                        flip_pos_node = pos_node_sub[flip_idx]
                        flip_pos_he = pos_he_sub[flip_idx]

                        keep_pos = ~slip_mask
                        pos_node_sub = pos_node_sub[keep_pos]
                        pos_he_sub = pos_he_sub[keep_pos]

                        neg_node_sub = torch.cat([neg_node_sub, flip_pos_node], dim=0) if neg_node_sub.numel() > 0 else flip_pos_node
                        neg_he_sub = torch.cat([neg_he_sub, flip_pos_he], dim=0) if neg_he_sub.numel() > 0 else flip_pos_he

                if float(guess_intensity) > 0.0 and neg_he_sub.numel() > 0:
                    neg_sid_local = neg_node_sub - Q
                    p_neg = torch.sigmoid(rasch_student[neg_sid_local] - rasch_item[neg_he_sub])
                    guess_prob = (1.0 - p_neg) * float(guess_intensity)
                    guess_mask = torch.rand(p_neg.shape, device=device) < guess_prob
                    if guess_mask.any():
                        flip_idx = torch.nonzero(guess_mask, as_tuple=True)[0]
                        flip_neg_node = neg_node_sub[flip_idx]
                        flip_neg_he = neg_he_sub[flip_idx]

                        keep_neg = ~guess_mask
                        neg_node_sub = neg_node_sub[keep_neg]
                        neg_he_sub = neg_he_sub[keep_neg]

                        pos_node_sub = torch.cat([pos_node_sub, flip_neg_node], dim=0) if pos_node_sub.numel() > 0 else flip_neg_node
                        pos_he_sub = torch.cat([pos_he_sub, flip_neg_he], dim=0) if pos_he_sub.numel() > 0 else flip_neg_he
            except Exception:
                pass

        # ========== 建立索引映射 ==========
        # 题目全局 id → 子图本地 id
        qid_to_local = torch.full((Q,), -1, device=device, dtype=torch.long)
        qid_to_local[q_ids_dev] = torch.arange(B, device=device, dtype=torch.long)

        # 找出子图中的活跃学生（有边连接的学生）
        active_sid_parts = []
        if pos_node_sub.numel() > 0:
            active_sid_parts.append(pos_node_sub - Q)
        if neg_node_sub.numel() > 0:
            active_sid_parts.append(neg_node_sub - Q)

        # 空图处理
        zeros_q = torch.zeros((B, D), device=device, dtype=q_feats.dtype)
        if len(active_sid_parts) == 0:
            # 无学生连接该子图 → 返回零投影
            hyper_concat = torch.cat([zeros_q, zeros_q], dim=-1)
            return self.hyper_proj(hyper_concat)

        # 活跃学生集合
        active_sid = torch.unique(torch.cat(active_sid_parts, dim=0))
        A = int(active_sid.numel())  # 活跃学生数
        
        # 学生全局 id → 本地 id
        sid_to_local = torch.full((S,), -1, device=device, dtype=torch.long)
        sid_to_local[active_sid] = torch.arange(A, device=device, dtype=torch.long)

        # ========== 本地索引转换 ==========
        def _to_local(node_sub: torch.Tensor, he_sub: torch.Tensor):
            """将全局边转换为本地索引。"""
            if he_sub.numel() == 0:
                return torch.zeros((0,), device=device, dtype=torch.long), torch.zeros((0,), device=device, dtype=torch.long)
            
            # 学生本地索引
            sid_local = sid_to_local[node_sub - Q]
            # 题目本地索引
            he_local = qid_to_local[he_sub]
            # 保留有效索引
            valid = (sid_local >= 0) & (he_local >= 0)
            return sid_local[valid], he_local[valid]

        # 转换正负通道的索引
        pos_sid_local, pos_he_local = _to_local(pos_node_sub, pos_he_sub)
        neg_sid_local, neg_he_local = _to_local(neg_node_sub, neg_he_sub)

        # ========== 获取活跃学生的 embedding ==========
        s_emb_active = self.student_emb.weight.to(device)[active_sid]  # [A, D]
        s_pos = torch.zeros((A, D), device=device, dtype=q_feats.dtype)
        s_neg = torch.zeros((A, D), device=device, dtype=q_feats.dtype)

        # ========== 第一跳：学生聚合题目历史 ==========
        # 正通道：学生聚合正向题目历史
        if pos_he_local.numel() > 0 and self.pos_hist_qid.numel() > 0:
            s_pos = self._student_history_query_attention_channel_active(
                q_feats,  # 全量特征用于历史 lookup
                q_sub,    # 子图特征用于 query 构造
                pos_sid_local,
                pos_he_local,
                self.pos_hist_sid,
                self.pos_hist_qid,
                sid_to_local,
                s_emb_active,
                channel_idx=0,
            )

        # 负通道：学生聚合负向题目历史
        if neg_he_local.numel() > 0 and self.neg_hist_qid.numel() > 0:
            s_neg = self._student_history_query_attention_channel_active(
                q_feats,
                q_sub,
                neg_sid_local,
                neg_he_local,
                self.neg_hist_sid,
                self.neg_hist_qid,
                sid_to_local,
                s_emb_active,
                channel_idx=1,
            )

        # 融合双通道学生表示：[A, D]
        s_from_q = self.student_fuse(torch.cat([s_pos, s_neg], dim=-1))
        
        # 丰富学生 embedding：student embedding + 从题目学到的表示
        s_refined_active = s_emb_active + s_from_q

        # ========== 第二跳：题目聚合学生 ==========
        # 正通道：题目聚合正向超边中的学生
        if getattr(self, 'use_attention', True):
            E_pos_attn = self._group_attention_pool_local(q_sub, s_refined_active, 
                                                         pos_sid_local, pos_he_local, B, channel_idx=0)
            
            # 负通道：题目聚合负向超边中的学生
            E_neg_attn = self._group_attention_pool_local(q_sub, s_refined_active, 
                                                         neg_sid_local, neg_he_local, B, channel_idx=1)
        else:
            # attention 关闭时用零张量（用于消融实验）
            E_pos_attn = zeros_q
            E_neg_attn = zeros_q

        # ========== 融合与投影 ==========
        # 拼接两通道：[B, 2D]
        hyper_concat = torch.cat([E_pos_attn, E_neg_attn], dim=-1)
        
        # 最终投影到 emb_dim：[B, emb_dim]
        return self.hyper_proj(hyper_concat)

    def forward(self, q_feats: torch.Tensor, q_ids=None) -> torch.Tensor:
        """主前向接口：支持全图或子图计算。
        
        两种模式：
        1. 子图模式（推荐）：q_ids != None
           输入当前批次的题目 id → 仅计算这些题目的超图投影
           复杂度 O(E_sub·D)，内存高效
        
        2. 全图模式（带缓存）：q_ids == None
           计算全量题目的超图投影（支持间隔更新缓存）
           复杂度 O(E·D)，用于离线投影计算
        
        参数：
        - q_feats: [Q, D] 全量题目特征
        - q_ids: 可选，子图题目 id（int/tensor/list）
        
        返回：
        - 子图模式：[B, emb_dim]，B = len(unique(q_ids))
        - 全图模式：[Q, emb_dim]（支持缓存）
        """
        # 模块状态检查
        if not self.enabled:
            raise RuntimeError('HypergraphDualChannel 未启用或文件缺失')

        device = q_feats.device
        
        # ========== 子图模式：仅计算指定题目的投影 ==========
        if q_ids is not None:
            out = self._compute_hyper_proj_for_qids(q_feats, q_ids, drop_rate=0.0)
            self._forward_calls += 1
            return out

        # ========== 全图模式：支持缓存机制减少重复计算 ==========
        # 根据 update_interval 决定是否使用缓存
        if (self._forward_calls % self.update_interval) != 0 and self.hyper_proj_cache is not None:
            # 使用缓存
            out = self.hyper_proj_cache.to(device)
            self._forward_calls += 1
            return out

        # 需要更新缓存：计算全量投影
        all_q_ids = torch.arange(self.num_questions, device=device, dtype=torch.long)
        hyper_proj_q = self._compute_hyper_proj_for_qids(q_feats, all_q_ids, drop_rate=0.0)

        # 更新缓存
        try:
            self.hyper_proj_cache.copy_(hyper_proj_q.detach())
        except Exception:
            self.hyper_proj_cache = hyper_proj_q.detach()

        self._forward_calls += 1
        return hyper_proj_q

    def compute_proj_with_member_dropout(self, q_feats: torch.Tensor, drop_rate: float = 0.3, q_ids=None,
                                         slip_intensity: float = 0.0, guess_intensity: float = 0.0) -> torch.Tensor:
        """计算超图投影，同时随机丢弃超边成员（数据增强）。
        
        应用场景：
        - 模拟学生知识图的不完整性
        - 作为数据增强提升泛化性
        - 用于鲁棒性评估
        
        参数：
        - q_feats: [Q, D] 题目特征
        - drop_rate: 成员 dropout 概率（0-1）
        - q_ids: 可选，子图题目 id
        - slip_intensity: 失误噪声强度（0-1）
        - guess_intensity: 猜测噪声强度（0-1）
        
        返回:
        - q_ids 为 None：[Q, emb_dim]
        - q_ids 不为 None：[B, emb_dim]，B = len(unique(q_ids))
        """
        if not self.enabled:
            raise RuntimeError('HypergraphDualChannel 未启用或缺少文件')
        
        device = q_feats.device
        
        if q_ids is None:
            # 全量投影
            q_ids_dev = torch.arange(self.num_questions, device=device, dtype=torch.long)
        else:
            # 子图投影
            q_ids_dev = self._normalize_q_ids(q_ids, device)
        
        return self._compute_hyper_proj_for_qids(q_feats, q_ids_dev, drop_rate=float(drop_rate),
                            slip_intensity=float(slip_intensity),
                            guess_intensity=float(guess_intensity))

if __name__ == '__main__':
    print('HypergraphDualChannel module')
