import argparse
import json
import os
import pickle

import numpy as np
import torch
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
from models.ukt import UKT
from models.utils import collate_fn
from trainers.kt_trainer import _forward_for_batch, _move_batch_to_model_device


NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def calc_binary_auc_acc(y_true, y_score, threshold=0.5):
    from sklearn import metrics

    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    acc = metrics.accuracy_score(
        y_true=y_true,
        y_pred=np.where(y_score >= threshold, 1, 0)
    )
    return auc, acc


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
        return UKT(dataset.num_q, **model_config).to(device)
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


def load_split_indices(dataset, train_ratio):
    train_indices_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
    test_indices_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")

    if os.path.exists(train_indices_path) and os.path.exists(test_indices_path):
        with open(train_indices_path, "rb") as f:
            train_indices = pickle.load(f)
        with open(test_indices_path, "rb") as f:
            test_indices = pickle.load(f)
    else:
        train_size = int(len(dataset) * train_ratio)
        generator = torch.Generator(device="cpu")
        indices = torch.randperm(len(dataset), generator=generator, device="cpu").tolist()
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        with open(train_indices_path, "wb") as f:
            pickle.dump(train_indices, f)
        with open(test_indices_path, "wb") as f:
            pickle.dump(test_indices, f)

    return train_indices, test_indices


def flip_response_noise(r, m, noise_ratio, generator=None):
    if noise_ratio <= 0:
        return r

    noisy_r = r.clone()
    valid_positions = torch.nonzero(m.bool(), as_tuple=False)
    if valid_positions.numel() == 0:
        return noisy_r

    num_flips = int(round(valid_positions.shape[0] * noise_ratio))
    if num_flips <= 0:
        return noisy_r

    if generator is None:
        perm = torch.randperm(valid_positions.shape[0], device=r.device)
    else:
        perm = torch.randperm(valid_positions.shape[0], device=r.device, generator=generator)

    flip_positions = valid_positions[perm[:num_flips]]
    row_idx = flip_positions[:, 0]
    col_idx = flip_positions[:, 1]
    selected_values = noisy_r[row_idx, col_idx]
    noisy_r[row_idx, col_idx] = torch.where(
        selected_values > 0.5,
        torch.zeros_like(selected_values),
        torch.ones_like(selected_values)
    )
    return noisy_r


def evaluate_with_noise(model_name, model, test_loader, noise_ratio, device):
    y_true_all = []
    y_score_all = []
    noise_generator = torch.Generator(device=device)

    with torch.no_grad():
        model.eval()
        for data in test_loader:
            q, r, qshft, rshft, m = _move_batch_to_model_device(model, data)
            noisy_r = flip_response_noise(r, m, noise_ratio, generator=noise_generator)

            if model_name in ["dkvmn", "saint"]:
                y_true = torch.masked_select(r, m).detach().cpu().numpy()
            else:
                y_true = torch.masked_select(rshft, m).detach().cpu().numpy()

            pred = _forward_for_batch(model_name, model, q, noisy_r, qshft)
            y_score = torch.masked_select(pred, m).detach().cpu().numpy()

            y_true_all.append(y_true)
            y_score_all.append(y_score)

    y_true_all = np.concatenate(y_true_all)
    y_score_all = np.concatenate(y_score_all)
    auc, acc = calc_binary_auc_acc(y_true=y_true_all, y_score=y_score_all)
    return auc, acc


def main(ckpt_path, dataset_name=None):
    ckpt_dir = resolve_ckpt_dir(ckpt_path)

    with open(os.path.join(ckpt_dir, "model_config.json")) as f:
        model_config = json.load(f)
    with open(os.path.join(ckpt_dir, "train_config.json")) as f:
        train_config = json.load(f)

    model_name = infer_model_name(ckpt_dir)
    if dataset_name is None:
        dataset_name = os.path.basename(ckpt_dir.rstrip(os.sep))

    seq_len = train_config["seq_len"]
    dataset = build_dataset(dataset_name, seq_len)

    _, test_indices = load_split_indices(dataset, train_config["train_ratio"])
    test_dataset = Subset(dataset, test_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, dataset, model_config, device)
    model = load_checkpoint(model, ckpt_dir)

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    results = []
    for noise_ratio in NOISE_LEVELS:
        auc, acc = evaluate_with_noise(model_name, model, test_loader, noise_ratio, device)
        results.append((noise_ratio, auc, acc))

    print("noise_ratio\tauc\tacc")
    for noise_ratio, auc, acc in results:
        print("{:.1f}\t{:.6f}\t{:.6f}".format(noise_ratio, auc, acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    main(args.ckpt_path, args.dataset_name)