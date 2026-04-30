import csv
import os
import pickle

import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset

from models.utils import match_seq_len


DATASET_DIR = "datasets/XES3G5M/"
RAW_TRAIN_FILE = "train.csv"
RAW_TEST_FILE = "test.csv"


class XES3G5M(Dataset):
    """
    XES3G5M loader for current KTbaseline framework.

    Raw input files (under DATASET_DIR):
    - train.csv
    - test.csv

    Required columns in each CSV row:
    - uid
    - questions   (comma-separated ints)
    - concepts    (comma-separated ints)
    - responses   (comma-separated ints)
    """
    def __init__(
        self,
        seq_len,
        dataset_dir=DATASET_DIR,
        raw_train_file=RAW_TRAIN_FILE,
        raw_test_file=RAW_TEST_FILE
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.raw_train_path = os.path.join(self.dataset_dir, raw_train_file)
        self.raw_test_path = os.path.join(self.dataset_dir, raw_test_file)

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
            (
                self.q_seqs,
                self.r_seqs,
                self.q_list,
                self.u_list,
                self.q2idx,
                self.u2idx
            ) = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = match_seq_len(
                self.q_seqs, self.r_seqs, seq_len
            )

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def _read_csv_users(self, csv_path):
        users = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = str(row["uid"])
                questions = [
                    int(x) for x in row["questions"].split(",") if x.strip()
                ]
                concepts = [
                    int(x) for x in row["concepts"].split(",") if x.strip()
                ]
                responses = [
                    int(x) for x in row["responses"].split(",") if x.strip()
                ]
                if len(questions) == len(concepts) == len(responses):
                    users[uid] = (questions, concepts, responses)
        return users

    def _build_maps(self, train_users, test_users):
        q2c = defaultdict(set)
        train_q, train_c = set(), set()
        test_q, test_c = set(), set()

        for _, (questions, concepts, responses) in train_users.items():
            for q, c, r in zip(questions, concepts, responses):
                if q <= 0 or c <= 0 or r < 0:
                    continue
                q2c[q].add(c)
                train_q.add(q)
                train_c.add(c)

        for _, (questions, concepts, responses) in test_users.items():
            for q, c, r in zip(questions, concepts, responses):
                if q <= 0 or c <= 0 or r < 0:
                    continue
                test_q.add(q)
                test_c.add(c)

        all_q = sorted(train_q.union(test_q))
        all_c = sorted(train_c.union(test_c))
        qid_map = {q: i for i, q in enumerate(all_q)}
        cid_map = {c: i for i, c in enumerate(all_c)}
        return dict(q2c), qid_map, cid_map

    def preprocess(self):
        if not os.path.exists(self.raw_train_path):
            raise FileNotFoundError(
                f"Raw train CSV not found: {self.raw_train_path}"
            )
        if not os.path.exists(self.raw_test_path):
            raise FileNotFoundError(
                f"Raw test CSV not found: {self.raw_test_path}"
            )

        os.makedirs(self.dataset_dir, exist_ok=True)

        train_users = self._read_csv_users(self.raw_train_path)
        test_users = self._read_csv_users(self.raw_test_path)
        _, qid_map, _ = self._build_maps(train_users, test_users)

        all_users = dict(train_users)
        for uid, val in test_users.items():
            if uid in all_users:
                q0, c0, r0 = all_users[uid]
                q1, c1, r1 = val
                all_users[uid] = (q0 + q1, c0 + c1, r0 + r1)
            else:
                all_users[uid] = val

        u_list = np.array(
            sorted(
                all_users.keys(),
                key=lambda x: int(x) if str(x).isdigit() else 0
            )
        )
        q_list = np.array(sorted(qid_map.values()))
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = dict(qid_map)

        q_seqs = []
        r_seqs = []
        for uid in u_list:
            questions, concepts, responses = all_users[uid]
            filtered_q = []
            filtered_r = []
            for q, c, r in zip(questions, concepts, responses):
                if q <= 0 or c <= 0 or r < 0:
                    continue
                if q not in qid_map:
                    continue
                filtered_q.append(qid_map[q])
                filtered_r.append(int(r))
            if len(filtered_q) == 0:
                continue
            q_seqs.append(np.array(filtered_q))
            r_seqs.append(np.array(filtered_r))

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
