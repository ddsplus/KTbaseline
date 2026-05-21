import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from models.utils import match_seq_len


DATASET_DIR = "datasets/statics2011/"


class Statics2011(Dataset):
    def __init__(self, seq_len, datset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.seq_len = seq_len

        self.dataset_dir = datset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, os.path.join(
                
                "AllData_student_step_2011F.csv"
            )
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

        if self.seq_len:
            self.q_seqs, self.r_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path, sep=None, engine="python")
        df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]

        normalized_col_map = {
            col.lower().replace("_", " ").replace("-", " ").strip(): col
            for col in df.columns
        }

        def _resolve_by_normalized(candidates):
            for name in candidates:
                key = name.lower().replace("_", " ").replace("-", " ").strip()
                if key in normalized_col_map:
                    return normalized_col_map[key]
            return None

        def _pick_column(candidates):
            for col in candidates:
                if col in df.columns:
                    return col
            return _resolve_by_normalized(candidates)

        problem_col = _pick_column(["Problem Name", "problem_name", "problem"])
        step_col = _pick_column(["Step Name", "step_name", "step"])
        user_col = _pick_column([
            "Anon Student Id", "Anon Student ID", "Student Id", "student_id",
            "user_id", "user"
        ])
        time_col = _pick_column(["Time", "First Transaction Time", "timestamp", "time"])
        outcome_col = _pick_column(["Outcome", "First Attempt", "Label", "correct"])
        attempt_col = _pick_column(["Attempt At Step", "attempt_at_step", "attempt"])
        response_type_col = _pick_column(["Student Response Type", "response_type"])
        corrects_col = _pick_column(["Corrects", "corrects"])
        incorrects_col = _pick_column(["Incorrects", "incorrects"])
        hints_col = _pick_column(["Hints", "hints"])

        required_cols = [user_col, problem_col, step_col]
        if any(col is None for col in required_cols):
            missing = []
            if user_col is None:
                missing.append("Anon Student Id")
            if problem_col is None:
                missing.append("Problem Name")
            if step_col is None:
                missing.append("Step Name")
            raise ValueError(
                "Missing required columns: {}. Available columns: {}".format(
                    missing, list(df.columns)
                )
            )

        df = df.dropna(subset=[problem_col, step_col])
        if time_col is not None:
            df = df.sort_values(by=[time_col])

        if attempt_col is not None:
            df = df[df[attempt_col] == 1]
        if response_type_col is not None:
            df = df[df[response_type_col] == "ATTEMPT"]

        if outcome_col is None:
            def derive_outcome(row):
                corrects = int(row[corrects_col]) if corrects_col and pd.notna(row[corrects_col]) else 0
                incorrects = int(row[incorrects_col]) if incorrects_col and pd.notna(row[incorrects_col]) else 0
                hints = int(row[hints_col]) if hints_col and pd.notna(row[hints_col]) else 0
                if corrects > 0 and incorrects == 0 and hints == 0:
                    return "CORRECT"
                return "INCORRECT"
            df["__outcome__"] = df.apply(derive_outcome, axis=1)
            outcome_col = "__outcome__"
        else:
            if outcome_col == "First Attempt":
                normalized = df[outcome_col].astype(str).str.strip().str.lower()
                df["__outcome__"] = np.where(normalized == "correct", "CORRECT", "INCORRECT")
                outcome_col = "__outcome__"
            else:
                df = df.dropna(subset=[outcome_col])

        kcs = []
        for _, row in df.iterrows():
            kcs.append("{}_{}".format(row[problem_col], row[step_col]))

        df["KC"] = kcs

        u_list = np.unique(df[user_col].values)
        q_list = np.unique(df["KC"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []
        for u in u_list:
            u_df = df[df[user_col] == u]

            q_seqs.append([q2idx[q] for q in u_df["KC"].values])
            r_seqs.append((u_df[outcome_col].astype(str).str.upper().values == "CORRECT").astype(int))

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
