import csv
import os
import pickle
import json

import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset

from models.utils import match_seq_len


DATASET_DIR = "datasets/XES3G5M/"
RAW_TRAIN_FILE = "train.csv"
RAW_TEST_FILE = "test.csv"
OFFICIAL_SPLIT_META_FILE = "official_split_meta.json"


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
        self.train_indices_path = os.path.join(self.dataset_dir, "train_indices.pkl")
        self.test_indices_path = os.path.join(self.dataset_dir, "test_indices.pkl")
        self.official_split_meta_path = os.path.join(
            self.dataset_dir, OFFICIAL_SPLIT_META_FILE
        )

        has_cached_sequences = os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl"))
        has_official_split_cache = (
            os.path.exists(self.train_indices_path)
            and os.path.exists(self.test_indices_path)
            and os.path.exists(self.official_split_meta_path)
        )
        has_strict_user_split_cache = False
        if has_official_split_cache:
            try:
                with open(self.official_split_meta_path, "r", encoding="utf-8") as f:
                    split_meta = json.load(f)
                has_strict_user_split_cache = (
                    split_meta.get("split") == "strict_user_disjoint"
                )
            except (ValueError, OSError):
                has_strict_user_split_cache = False

        if has_cached_sequences and has_strict_user_split_cache:
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

        train_users_raw = self._read_csv_users(self.raw_train_path)
        test_users_raw = self._read_csv_users(self.raw_test_path)

        train_uids = set(train_users_raw.keys())
        test_uids = set(test_users_raw.keys())
        overlap_uids = train_uids.intersection(test_uids)

        if overlap_uids:
            train_users = {
                uid: seq for uid, seq in train_users_raw.items()
                if uid not in overlap_uids
            }
            test_users = dict(test_users_raw)
        else:
            train_users = dict(train_users_raw)
            test_users = dict(test_users_raw)

        _, qid_map, _ = self._build_maps(train_users, {})

        q_list = np.array(sorted(qid_map.values()))
        q2idx = dict(qid_map)

        q_seqs = []
        r_seqs = []
        sequence_users = []
        train_indices = []
        test_indices = []

        def _append_split_sequences(users, split_name):
            for uid in sorted(users.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                questions, concepts, responses = users[uid]
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

                seq_idx = len(q_seqs)
                q_seqs.append(np.array(filtered_q))
                r_seqs.append(np.array(filtered_r))
                sequence_users.append("{}::{}".format(split_name, uid))
                if split_name == "train":
                    train_indices.append(seq_idx)
                else:
                    test_indices.append(seq_idx)

        _append_split_sequences(train_users, "train")
        _append_split_sequences(test_users, "test")

        u_list = np.array(sequence_users)
        u2idx = {u: idx for idx, u in enumerate(u_list)}

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
        with open(self.train_indices_path, "wb") as f:
            pickle.dump(train_indices, f)
        with open(self.test_indices_path, "wb") as f:
            pickle.dump(test_indices, f)
        split_meta = {
            "split": "strict_user_disjoint",
            "train_user_count": len(train_users),
            "test_user_count": len(test_users),
            "overlap_user_count": len(overlap_uids),
            "overlap_users_assigned_to": "test",
        }
        with open(self.official_split_meta_path, "w", encoding="utf-8") as f:
            json.dump(split_meta, f)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx
