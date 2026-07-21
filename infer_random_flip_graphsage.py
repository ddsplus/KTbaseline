#!/usr/bin/env python3
"""
Random flip inference for GraphSAGE_KT.

This script evaluates a trained GraphSAGE_KT checkpoint under answer-flip
noise only: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 by default.
"""

import argparse
import csv
import math
import os
import pickle
import random

import numpy as np
import torch
from sklearn import metrics
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Subset

from data_loaders.algebra2005 import Algebra2005
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.assist2017 import ASSIST2017
from data_loaders.statics2011 import Statics2011
from data_loaders.xes3g5m import XES3G5M
from models.utils import collate_fn
from temp.graphsage_kt import GraphSAGE_KT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random flip inference for GraphSAGE_KT"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Path to the dataset root directory. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g. ASSIST2009, ASSIST2015, Algebra2005, Statics2011, ASSIST2017, XES3G5M.",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=3,
        help="Accepted for compatibility; not used by this repo's loaders.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=200,
        help="Sequence chunk length used for evaluation.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Used only when cached split indices are unavailable.",
    )

    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--exercise_dim", type=int, default=128)
    parser.add_argument("--rnn_mode", type=str, default="lstm")
    parser.add_argument("--rnn_num_layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=33)

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved GraphSAGE_KT .pt state_dict.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
    )

    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Random flip ratios to test.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional CSV output path. Defaults to this script directory.",
    )

    parser.add_argument("--hg-update-interval", dest="hg_update_interval", type=int, default=1)
    parser.add_argument("--hg-use-scatter", dest="hg_use_scatter", action="store_true")
    parser.add_argument("--gs-update-interval", dest="gs_update_interval", type=int, default=1)
    parser.add_argument("--use-fm", dest="use_fm", action="store_true")
    parser.add_argument("--fm-lambda", dest="fm_lambda", type=float, default=1.0)
    parser.add_argument("--fm-steps", dest="fm_steps", type=int, default=4)
    parser.add_argument("--fm-hidden", dest="fm_hidden", type=int, default=256)
    parser.add_argument("--fm-time-dim", dest="fm_time_dim", type=int, default=32)
    parser.add_argument("--fm-drop-rate", dest="fm_drop_rate", type=float, default=0.3)
    parser.add_argument("--fm-noise", dest="fm_noise", type=float, default=0.1)
    parser.add_argument("--fm-viz", dest="fm_viz", action="store_true", default=False)
    parser.add_argument("--fm-viz-dir", dest="fm_viz_dir", type=str, default="fm_viz")
    parser.add_argument("--fm-viz-per-epoch", dest="fm_viz_per_epoch", type=int, default=1)

    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def auto_detect_data_path():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(repo_dir, "datasets"),
        os.path.join(repo_dir, "data"),
        os.path.join(repo_dir, "Data"),
        os.path.join(repo_dir, "..", "datasets"),
        os.path.join(repo_dir, "..", "data"),
        os.path.join(repo_dir, "..", "Data"),
    ]

    for path in possible_paths:
        if os.path.isdir(os.path.abspath(path)):
            return os.path.abspath(path)

    raise FileNotFoundError(
        "Could not auto-detect dataset root directory. Please pass --data_path explicitly."
    )


def resolve_dataset_dir(data_path, dataset_name):
    candidate_names = [
        dataset_name,
        dataset_name.upper(),
        dataset_name.lower(),
    ]

    aliases = {
        "ASSIST2009": ["ASSIST2009", "assist2009"],
        "ASSIST2015": ["ASSIST2015", "assist2015"],
        "ASSIST2017": ["ASSIST2017", "assist2017"],
        "Algebra2005": ["Algebra2005", "algebra2005", "algebra_2005_2006"],
        "Statics2011": ["Statics2011", "statics2011"],
        "XES3G5M": ["XES3G5M", "xes3g5m"],
    }
    candidate_names.extend(aliases.get(dataset_name, []))

    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        dataset_dir = os.path.abspath(os.path.join(data_path, name))
        if os.path.isdir(dataset_dir):
            return dataset_dir

    raise FileNotFoundError(
        "Dataset '{}' was not found under '{}'.".format(dataset_name, os.path.abspath(data_path))
    )


def build_dataset(dataset_name, seq_len, dataset_dir):
    if dataset_name == "ASSIST2009":
        return ASSIST2009(seq_len, dataset_dir=dataset_dir)
    if dataset_name == "ASSIST2015":
        return ASSIST2015(seq_len, dataset_dir=dataset_dir)
    if dataset_name == "Algebra2005":
        return Algebra2005(seq_len, datset_dir=dataset_dir)
    if dataset_name == "Statics2011":
        return Statics2011(seq_len, datset_dir=dataset_dir)
    if dataset_name == "ASSIST2017":
        return ASSIST2017(seq_len, dataset_dir=dataset_dir)
    if dataset_name == "XES3G5M":
        return XES3G5M(seq_len, dataset_dir=dataset_dir)
    raise ValueError("Unsupported dataset: {}".format(dataset_name))


def load_split_indices(dataset, train_ratio, seed):
    train_indices_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
    test_indices_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")

    if os.path.exists(train_indices_path) and os.path.exists(test_indices_path):
        with open(train_indices_path, "rb") as f:
            train_indices = pickle.load(f)
        with open(test_indices_path, "rb") as f:
            test_indices = pickle.load(f)
        return train_indices, test_indices

    if dataset.__class__.__name__ == "XES3G5M":
        raise FileNotFoundError(
            "Official XES3G5M split indices not found. Please regenerate preprocessing first."
        )

    train_size = int(len(dataset) * train_ratio)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator, device="cpu").tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    with open(train_indices_path, "wb") as f:
        pickle.dump(train_indices, f)
    with open(test_indices_path, "wb") as f:
        pickle.dump(test_indices, f)

    return train_indices, test_indices


def graphsage_collate_fn(batch):
    data = collate_fn(batch)
    q, r, qshft, rshft, mask = data[:5]

    seq_lens = mask.sum(dim=1).long()
    sort_idx = torch.argsort(seq_lens, descending=True)

    q = q.index_select(0, sort_idx).long()
    r = r.index_select(0, sort_idx).float()
    qshft = qshft.index_select(0, sort_idx).long()
    rshft = rshft.index_select(0, sort_idx).float()
    seq_lens = seq_lens.index_select(0, sort_idx)

    pad_curr = q.transpose(0, 1).contiguous()
    pad_answer = r.transpose(0, 1).contiguous()
    pad_next = qshft.transpose(0, 1).contiguous()
    label = rshft.transpose(0, 1).contiguous()
    pack_label = pack_padded_sequence(label, seq_lens.cpu(), enforce_sorted=True)

    return seq_lens, pad_curr, pad_answer, pad_next, pack_label


def perturb_answers_random_flip(seq_lens, pad_answer, noise_level):
    if float(noise_level) <= 0.0:
        return pad_answer.clone()

    noisy_answers = pad_answer.clone().float()
    seq_len, batch_size = noisy_answers.shape[0], noisy_answers.shape[1]

    for b in range(batch_size):
        actual_len = min(int(seq_lens[b].item()), seq_len)
        if actual_len <= 0:
            continue

        candidate = np.arange(actual_len)
        num_to_flip = max(1, int(candidate.size * float(noise_level)))
        num_to_flip = min(num_to_flip, int(candidate.size))
        flip_indices = np.random.choice(candidate, size=num_to_flip, replace=False)
        noisy_answers[flip_indices, b] = 1.0 - noisy_answers[flip_indices, b]

    return noisy_answers


def run_inference_with_random_flip(model, data_loader, noise_level, seed, device):
    set_seed(seed)
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for seq_lens, pad_curr, pad_answer, pad_next, pack_label in data_loader:
            seq_lens = seq_lens.to(device)
            pad_curr = pad_curr.to(device)
            pad_answer = pad_answer.to(device)
            pad_next = pad_next.to(device)

            noisy_pad_answer = perturb_answers_random_flip(seq_lens, pad_answer, noise_level)
            pack_pred = model(seq_lens, pad_curr, noisy_pad_answer, pad_next)

            y_true_all.append(
                pack_label.data.detach().cpu().contiguous().view(-1).numpy()
            )
            y_pred_all.append(
                pack_pred.data.detach().cpu().contiguous().view(-1).numpy()
            )

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
    return y_true, y_pred


def calc_metrics(y_true, y_pred):
    out = {"auc": float("nan"), "acc": float("nan"), "rmse": float("nan")}
    if y_true.size == 0:
        return out

    try:
        out["auc"] = float(metrics.roc_auc_score(y_true, y_pred))
    except Exception:
        pass

    try:
        out["acc"] = float(((y_pred >= 0.5) == (y_true >= 0.5)).mean())
    except Exception:
        pass

    try:
        out["rmse"] = float(math.sqrt(np.mean((y_pred - y_true) ** 2)))
    except Exception:
        pass

    return out


def write_results_csv(path, results_list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["noise_level", "auc", "acc", "rmse", "num_samples"])
        for result in results_list:
            writer.writerow([
                result["noise_level"],
                "{:.6f}".format(result["metrics"]["auc"]),
                "{:.6f}".format(result["metrics"]["acc"]),
                "{:.6f}".format(result["metrics"]["rmse"]),
                result["num_samples"],
            ])


def resolve_output_csv_path(output_csv, dataset, split):
    if output_csv:
        return output_csv

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = "flip_results_{}_{}.csv".format(dataset, split)
    return os.path.join(script_dir, filename)


def infer_seq_mode_from_state(state, args):
    has_rnn = any(key.startswith("seq_encoder.rnn.") for key in state.keys())
    has_transformer = any(key.startswith("seq_encoder.transformer.") for key in state.keys())

    if has_rnn and not has_transformer:
        return args.rnn_mode
    if has_transformer and not has_rnn:
        return "transformer"
    return getattr(args, "seq_mode", args.rnn_mode)


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.data_path:
        args.data_path = auto_detect_data_path()
        print("Auto-detected data path: {}".format(args.data_path))

    dataset_dir = resolve_dataset_dir(args.data_path, args.dataset)
    print("Using dataset dir: {}".format(dataset_dir))

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError("Model not found: {}".format(args.model_path))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.device = str(device)

    try:
        state = torch.load(args.model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model_path, map_location=device)

    args.seq_mode = infer_seq_mode_from_state(state, args)
    args.use_fm = bool(args.use_fm or any(key.startswith("fm.") for key in state.keys()))
    args.transformer_nhead = getattr(args, "transformer_nhead", 8)
    args.transformer_layers = getattr(args, "transformer_layers", 2)
    args.transformer_dropout = getattr(args, "transformer_dropout", 0.1)
    args.rnn_dropout = getattr(args, "rnn_dropout", 0.0)

    dataset = build_dataset(args.dataset, args.max_seq_len, dataset_dir)
    train_indices, test_indices = load_split_indices(
        dataset, train_ratio=args.train_ratio, seed=args.seed
    )
    eval_indices = train_indices if args.split == "train" else test_indices
    eval_dataset = Subset(dataset, eval_indices)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=graphsage_collate_fn,
    )

    data_info = {"num_ques": int(dataset.num_q)}
    model = GraphSAGE_KT(args, data_info).to(device)
    model.load_state_dict(state, strict=True)

    print("Model: {}".format(args.model_path))
    print("Dataset: {}".format(args.dataset))
    print("Split: {}".format(args.split))
    print("Seq Mode: {}".format(args.seq_mode))
    print("Use FM: {}".format(bool(args.use_fm)))
    print("=" * 72)
    print("{:<12} {:<14} {:<14} {:<14} {:<8}".format("Noise Level", "AUC", "ACC", "RMSE", "Samples"))
    print("=" * 72)

    results_list = []
    for noise_level in sorted(float(level) for level in args.noise_levels):
        y_true, y_pred = run_inference_with_random_flip(
            model, eval_loader, noise_level, args.seed, device
        )
        metric = calc_metrics(y_true, y_pred)
        result = {
            "noise_level": noise_level,
            "metrics": metric,
            "num_samples": int(y_true.size),
        }
        results_list.append(result)
        print(
            "{:<12.1f} {:<14.6f} {:<14.6f} {:<14.6f} {:<8}".format(
                noise_level,
                metric["auc"],
                metric["acc"],
                metric["rmse"],
                y_true.size,
            )
        )

    print("=" * 72)

    output_csv_path = resolve_output_csv_path(args.output_csv, args.dataset, args.split)
    write_results_csv(output_csv_path, results_list)
    print("Results saved to: {}".format(output_csv_path))


if __name__ == "__main__":
    main()
