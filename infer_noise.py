import argparse
import csv
import json
import math
import os
import pickle
import random

import numpy as np
import torch
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

from data_loaders.algebra2005 import Algebra2005
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.assist2017 import ASSIST2017
from data_loaders.statics2011 import Statics2011
from data_loaders.xes3g5m import XES3G5M
from models.dkt import DKT
from models.dkt_forget import DKTForget
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.gkt import MHA, PAM
from models.gkt_fm import GKTFM
from models.kqn import KQN
from models.saint import SAINT
from models.sakt import SAKT
from models.simplekt import SimpleKT
from models.utils import collate_fn
from trainers.kt_trainer import _forward_for_batch, _move_batch_to_model_device


def parse_args():
    parser = argparse.ArgumentParser(description="Noise inference for KT models")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the saved model directory or checkpoint file."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name. If omitted, it is inferred from the checkpoint directory."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Random seed for deterministic perturbation."
    )
    parser.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Noise levels (flip/drop ratios) to test."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional CSV output path for grouped perturbation results."
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="",
        help="Optional TXT output path for grouped perturbation results."
    )
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


def build_dataset(dataset_name, seq_len):
    if dataset_name == "ASSIST2009":
        return ASSIST2009(seq_len)
    if dataset_name == "ASSIST2015":
        return ASSIST2015(seq_len)
    if dataset_name == "Algebra2005":
        return Algebra2005(seq_len)
    if dataset_name == "Statics2011":
        return Statics2011(seq_len)
    if dataset_name == "ASSIST2017":
        return ASSIST2017(seq_len)
    if dataset_name == "XES3G5M":
        return XES3G5M(seq_len)
    raise ValueError("Unsupported dataset_name: {}".format(dataset_name))


def build_model(model_name, dataset, model_config, device):
    if model_name == "dkt":
        return DKT(dataset.num_q, **model_config).to(device)
    if model_name == "dkt-f":
        return DKTForget(dataset.num_q, **model_config).to(device)
    if model_name == "dkt+":
        return DKTPlus(dataset.num_q, **model_config).to(device)
    if model_name == "dkvmn":
        return DKVMN(dataset.num_q, **model_config).to(device)
    if model_name == "sakt":
        return SAKT(dataset.num_q, **model_config).to(device)
    if model_name == "kqn":
        return KQN(dataset.num_q, **model_config).to(device)
    if model_name == "saint":
        return SAINT(dataset.num_q, **model_config).to(device)
    if model_name == "ukt":
        from models.ukt import UKT
        return UKT(dataset.num_q, n_pid=0, **model_config).to(device)
    if model_name == "robustkt":
        from models.robustkt import Robustkt
        return Robustkt(dataset.num_q, n_pid=0, **model_config).to(device)
    if model_name == "simplekt":
        return SimpleKT(dataset.num_q, **model_config).to(device)
    if model_name == "gkt":
        if model_config["method"] == "PAM":
            return PAM(dataset.num_q, **model_config).to(device)
        if model_config["method"] == "MHA":
            return MHA(dataset.num_q, **model_config).to(device)
        raise ValueError("Unsupported GKT method: {}".format(model_config["method"]))
    if model_name == "gkt-fm":
        return GKTFM(dataset.num_q, **model_config).to(device)
    raise ValueError("Unsupported model_name: {}".format(model_name))


def infer_model_name(ckpt_dir):
    return os.path.basename(os.path.dirname(ckpt_dir.rstrip(os.sep)))


def resolve_ckpt_dir(ckpt_path):
    if os.path.isfile(ckpt_path):
        return os.path.dirname(ckpt_path)
    return ckpt_path


def load_checkpoint(model, ckpt_dir):
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    legacy_path = os.path.join(ckpt_dir, "model.ckpt")

    if os.path.exists(best_path):
        state_dict = torch.load(best_path, map_location="cpu")
    elif os.path.exists(legacy_path):
        state_dict = torch.load(legacy_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            "No checkpoint found in {}. Expected best_model.pt or model.ckpt".format(
                ckpt_dir
            )
        )

    model.load_state_dict(state_dict)
    return model


def load_split_indices(dataset, train_ratio, seed):
    train_indices_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
    test_indices_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")

    if os.path.exists(train_indices_path) and os.path.exists(test_indices_path):
        with open(train_indices_path, "rb") as f:
            train_indices = pickle.load(f)
        with open(test_indices_path, "rb") as f:
            test_indices = pickle.load(f)
    else:
        if dataset.__class__.__name__ == "XES3G5M":
            raise FileNotFoundError(
                "Official XES3G5M split indices not found. "
                "Please remove cached XES3G5M pkl files and rerun preprocessing."
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


def _clone_optional_tensor(x):
    if x is None:
        return None
    return x.clone()


def _rebuild_batch(q, r, qshft, rshft, m, pid, pidshft, keep_masks):
    seqs = []
    batch_size = int(q.shape[0])

    for b in range(batch_size):
        actual_len = int(m[b].sum().item())
        keep_mask = keep_masks[b][:actual_len]
        if keep_mask.size == 0 or not keep_mask.any():
            keep_mask = np.zeros((actual_len,), dtype=bool)
            if actual_len > 0:
                keep_mask[0] = True

        keep_idx = torch.as_tensor(np.flatnonzero(keep_mask), device=q.device, dtype=torch.long)
        if keep_idx.numel() == 0:
            continue

        seq_item = {
            "q": q[b, :actual_len].index_select(0, keep_idx),
            "r": r[b, :actual_len].index_select(0, keep_idx),
            "qshft": qshft[b, :actual_len].index_select(0, keep_idx),
            "rshft": rshft[b, :actual_len].index_select(0, keep_idx),
        }
        if pid is not None:
            seq_item["pid"] = pid[b, :actual_len].index_select(0, keep_idx)
            seq_item["pidshft"] = pidshft[b, :actual_len].index_select(0, keep_idx)
        seqs.append(seq_item)

    if not seqs:
        return q, r, qshft, rshft, m, pid, pidshft

    seqs.sort(key=lambda item: int(item["q"].shape[0]), reverse=True)
    new_q = pad_sequence([item["q"] for item in seqs], batch_first=True, padding_value=0.0).to(q.device)
    new_r = pad_sequence([item["r"] for item in seqs], batch_first=True, padding_value=0.0).to(r.device)
    new_qshft = pad_sequence([item["qshft"] for item in seqs], batch_first=True, padding_value=0.0).to(qshft.device)
    new_rshft = pad_sequence([item["rshft"] for item in seqs], batch_first=True, padding_value=0.0).to(rshft.device)
    new_m = pad_sequence(
        [torch.ones(item["q"].shape[0], dtype=torch.bool) for item in seqs],
        batch_first=True,
        padding_value=False
    ).to(m.device)

    new_pid = None
    new_pidshft = None
    if pid is not None:
        new_pid = pad_sequence([item["pid"] for item in seqs], batch_first=True, padding_value=0.0).to(pid.device)
        new_pidshft = pad_sequence(
            [item["pidshft"] for item in seqs],
            batch_first=True,
            padding_value=0.0
        ).to(pidshft.device)

    return new_q, new_r, new_qshft, new_rshft, new_m, new_pid, new_pidshft


def perturb_batch(q, r, qshft, rshft, m, pid, pidshft, noise_level, noise_mode):
    noise_mode = str(noise_mode).lower().strip()
    if noise_mode == "clean" or float(noise_level) <= 0.0:
        return q, r, qshft, rshft, m, _clone_optional_tensor(pid), _clone_optional_tensor(pidshft)

    if noise_mode in {"flip", "c2w", "w2c"}:
        noisy_r = r.clone().float()
        batch_size = int(r.shape[0])

        for b in range(batch_size):
            valid_idx = torch.nonzero(m[b].bool(), as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            ans_slice = noisy_r[b, valid_idx]
            if noise_mode == "flip":
                candidate = torch.arange(valid_idx.numel(), device=valid_idx.device)
            elif noise_mode == "c2w":
                candidate = torch.nonzero(ans_slice > 0.5, as_tuple=True)[0]
            else:
                candidate = torch.nonzero(ans_slice <= 0.5, as_tuple=True)[0]

            if candidate.numel() == 0:
                continue

            num_to_flip = max(1, int(candidate.numel() * float(noise_level)))
            num_to_flip = min(num_to_flip, int(candidate.numel()))
            perm = torch.randperm(candidate.numel(), device=candidate.device)[:num_to_flip]
            flip_positions = valid_idx[candidate[perm]]
            noisy_r[b, flip_positions] = 1.0 - noisy_r[b, flip_positions]

        return q, noisy_r, qshft, rshft, m, _clone_optional_tensor(pid), _clone_optional_tensor(pidshft)

    if noise_mode == "drop":
        keep_masks = []
        batch_size = int(q.shape[0])
        for b in range(batch_size):
            actual_len = int(m[b].sum().item())
            if actual_len <= 1:
                keep_masks.append(np.ones((max(actual_len, 1),), dtype=bool))
                continue

            num_to_drop = max(1, int(actual_len * float(noise_level)))
            num_to_drop = min(num_to_drop, actual_len - 1)
            drop_idx = np.random.choice(actual_len, size=num_to_drop, replace=False)
            keep_mask = np.ones((actual_len,), dtype=bool)
            keep_mask[drop_idx] = False
            keep_masks.append(keep_mask)

        return _rebuild_batch(q, r, qshft, rshft, m, pid, pidshft, keep_masks)

    raise ValueError("Unsupported noise_mode: {}".format(noise_mode))


def run_inference_with_noise(model_name, model, data_loader, noise_level, seed, noise_mode="clean"):
    set_seed(seed)
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for data in data_loader:
            q, r, qshft, rshft, m, pid, pidshft = _move_batch_to_model_device(model, data)
            q, noisy_r, qshft, rshft, m, pid, pidshft = perturb_batch(
                q, r, qshft, rshft, m, pid, pidshft, noise_level, noise_mode
            )
            pred = _forward_for_batch(model_name, model, q, noisy_r, qshft, pid=pid)

            seq_len = min(pred.shape[1], m.shape[1])
            pred = pred[:, :seq_len]
            m_aligned = m[:, :seq_len]
            r_aligned = noisy_r[:, :seq_len]
            rshft_aligned = rshft[:, :seq_len]

            y_pred = torch.masked_select(pred, m_aligned).detach().cpu().contiguous().view(-1).numpy()
            if model_name in ["dkvmn", "saint"]:
                y_true = torch.masked_select(r_aligned, m_aligned).float().detach().cpu().contiguous().view(-1).numpy()
            else:
                y_true = torch.masked_select(rshft_aligned, m_aligned).detach().cpu().contiguous().view(-1).numpy()

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
    return y_true, y_pred


def write_results_csv(path, results_list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "mode", "noise_level", "auc", "acc", "rmse", "num_samples"])
        for result in results_list:
            writer.writerow([
                result["group"],
                result["mode"],
                result["noise_level"],
                "{:.6f}".format(result["metrics"]["auc"]),
                "{:.6f}".format(result["metrics"]["acc"]),
                "{:.6f}".format(result["metrics"]["rmse"]),
                result["num_samples"],
            ])


def write_results_txt(path, results_list, ckpt_path, dataset_name, split):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Model: {}\n".format(ckpt_path))
        f.write("Dataset: {}\n".format(dataset_name))
        f.write("Split: {}\n".format(split))
        f.write("=" * 96 + "\n")
        f.write(
            "{:<18} {:<10} {:<8} {:<14} {:<14} {:<14} {:<8}\n".format(
                "Group", "Mode", "Level", "AUC", "ACC", "RMSE", "Samples"
            )
        )
        f.write("=" * 96 + "\n")
        for result in results_list:
            metric = result["metrics"]
            f.write(
                "{:<18} {:<10} {:<8.1f} {:<14.6f} {:<14.6f} {:<14.6f} {:<8}\n".format(
                    result["group"],
                    result["mode"],
                    result["noise_level"],
                    metric["auc"],
                    metric["acc"],
                    metric["rmse"],
                    result["num_samples"],
                )
            )


def main():
    args = parse_args()
    set_seed(args.seed)

    ckpt_dir = resolve_ckpt_dir(args.ckpt_path)

    with open(os.path.join(ckpt_dir, "model_config.json")) as f:
        model_config = json.load(f)
    with open(os.path.join(ckpt_dir, "train_config.json")) as f:
        train_config = json.load(f)

    model_name = infer_model_name(ckpt_dir)
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = os.path.basename(ckpt_dir.rstrip(os.sep))

    seq_len = train_config["seq_len"]
    dataset = build_dataset(dataset_name, seq_len)

    train_indices, test_indices = load_split_indices(dataset, train_config["train_ratio"], args.seed)
    if args.split == "train":
        eval_dataset = Subset(dataset, train_indices)
    else:
        eval_dataset = Subset(dataset, test_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, dataset, model_config, device)
    model = load_checkpoint(model, ckpt_dir)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    positive_levels = [float(x) for x in sorted(args.noise_levels) if float(x) > 0.0]
    eval_plan = [("clean", "clean", 0.0)]
    eval_plan.extend(("drop_{:.1f}".format(level), "drop", level) for level in positive_levels)
    eval_plan.extend(("c2w_{:.1f}".format(level), "c2w", level) for level in positive_levels)
    eval_plan.extend(("w2c_{:.1f}".format(level), "w2c", level) for level in positive_levels)

    print("Model: {}".format(args.ckpt_path))
    print("Dataset: {}".format(dataset_name))
    print("Split: {}".format(args.split))
    print("=" * 88)
    print("{:<18} {:<8} {:<14} {:<14} {:<14} {:<8}".format("Group", "Level", "AUC", "ACC", "RMSE", "Samples"))
    print("=" * 88)

    results_list = []
    for group_name, noise_mode, noise_level in eval_plan:
        y_true, y_pred = run_inference_with_noise(
            model_name, model, eval_loader, noise_level, args.seed, noise_mode=noise_mode
        )
        metric = calc_metrics(y_true, y_pred)
        result = {
            "group": group_name,
            "mode": noise_mode,
            "noise_level": noise_level,
            "metrics": metric,
            "num_samples": int(y_true.size),
        }
        results_list.append(result)
        print(
            "{:<18} {:<8.1f} {:<14.6f} {:<14.6f} {:<14.6f} {:<8}".format(
                group_name,
                noise_level,
                metric["auc"],
                metric["acc"],
                metric["rmse"],
                y_true.size,
            )
        )

    print("=" * 88)

    if args.output_csv:
        write_results_csv(args.output_csv, results_list)
        print("Results saved to: {}".format(args.output_csv))

    if args.output_txt:
        write_results_txt(args.output_txt, results_list, args.ckpt_path, dataset_name, args.split)
        print("TXT report saved to: {}".format(args.output_txt))


if __name__ == "__main__":
    main()
