import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from models.utils import match_seq_len

# 修改默认数据集路径
DATASET_DIR = "datasets/ASSIST2017/"

class ASSIST2017(Dataset):  # 类名改为ASSIST2017
    def __init__(self, seq_len, dataset_dir=DATASET_DIR, data_file="anonymized_full_release_competition_dataset.csv") -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, data_file
        )

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
                self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        # 修正编码为utf-8，并处理可能的编码问题
        try:
            df = pd.read_csv(self.dataset_path, encoding='utf-8')
        except UnicodeDecodeError:
            # 如果utf-8失败，尝试其他常见编码
            df = pd.read_csv(self.dataset_path, encoding='latin1')
        
        # ASSIST2017的列名：studentId, skill, correct
        # 删除skill为空的行
        df = df.dropna(subset=["skill"])
        
        # 按照studentId和可能的顺序列排序（ASSIST2017有startTime和endTime）
        # 如果存在startTime，按时间排序；否则按原始顺序
        if "startTime" in df.columns:
            df = df.sort_values(by=["studentId", "startTime"])
        elif "endTime" in df.columns:
            df = df.sort_values(by=["studentId", "endTime"])
        else:
            # 如果没有时间列，按原始索引排序
            df = df.sort_values(by=["studentId"])
        
        # 提取唯一的学生和技能
        u_list = np.unique(df["studentId"].values)
        q_list = np.unique(df["skill"].values)
        
        # 创建映射
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        
        q_seqs = []
        r_seqs = []
        
        # 按学生分组处理
        for u in u_list:
            df_u = df[df["studentId"] == u]
            
            # 获取技能序列（转换为索引）
            q_seq = np.array([q2idx[q] for q in df_u["skill"].values])
            
            # 获取正确性序列（确保是整数类型）
            r_seq = df_u["correct"].values.astype(int)
            
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
        
        # 保存预处理结果
        os.makedirs(self.dataset_dir, exist_ok=True)  # 确保目录存在
        
        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)
        
        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx