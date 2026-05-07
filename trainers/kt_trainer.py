import os

import numpy as np
import torch

from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


def calc_binary_auc_acc(y_true, y_score, threshold=0.5):
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    acc = metrics.accuracy_score(
        y_true=y_true,
        y_pred=np.where(y_score >= threshold, 1, 0)
    )
    return auc, acc


def _forward_for_batch(model_name, model, q, r, qshft):
    if model_name in ["dkt", "dkt+", "dkt-f"]:
        y = model(q.long(), r.long())
        y = (y * one_hot(qshft.long(), model.num_q)).sum(-1)
        return y
    if model_name == "ukt":
        y, _ = model(q.long(), r.long(), train=False)
        y = (y * one_hot(qshft.long(), model.num_q)).sum(-1)
        return y
    if model_name == "gkt":
        y, _ = model(q.long(), r.long())
        seq_len = min(y.shape[1], qshft.shape[1])
        y = y[:, :seq_len, :]
        qshft = qshft[:, :seq_len]
        y = (y * one_hot(qshft.long(), num_classes=y.shape[-1])).sum(-1)
        return y
    if model_name == "dkvmn":
        p, _ = model(q.long(), r.long())
        return p
    if model_name == "sakt":
        p, _ = model(q.long(), r.long(), qshft.long())
        return p
    if model_name == "kqn":
        return model(q.long(), r.long(), qshft.long())
    if model_name == "saint":
        return model(q.long(), r.long())
    if model_name == "simplekt":
        return model(q.long(), r.long(), qshft.long())
    raise ValueError("Unsupported model_name: {}".format(model_name))


def _train_loss(model_name, model, pred, q, r, qshft, rshft, m):
    if model_name == "dkt+":
        y = model(q.long(), r.long())
        y_curr = (y * one_hot(q.long(), model.num_q)).sum(-1)
        y_next = (y * one_hot(qshft.long(), model.num_q)).sum(-1)

        y_curr = torch.masked_select(y_curr, m)
        y_next = torch.masked_select(y_next, m)
        r_masked = torch.masked_select(r, m)
        rshft_masked = torch.masked_select(rshft, m)

        loss_w1 = torch.masked_select(
            torch.norm(y[:, 1:] - y[:, :-1], p=1, dim=-1),
            m[:, 1:]
        )
        loss_w2 = torch.masked_select(
            (torch.norm(y[:, 1:] - y[:, :-1], p=2, dim=-1) ** 2),
            m[:, 1:]
        )

        return (
            binary_cross_entropy(y_next, rshft_masked)
            + model.lambda_r * binary_cross_entropy(y_curr, r_masked)
            + model.lambda_w1 * loss_w1.mean() / model.num_q
            + model.lambda_w2 * loss_w2.mean() / model.num_q
        )
    if model_name == "ukt":
        y, aux_losses = model(q.long(), r.long(), train=True)
        y_next = (y * one_hot(qshft.long(), model.num_q)).sum(-1)
        y_next = torch.masked_select(y_next, m)
        target = torch.masked_select(rshft, m)
        loss = binary_cross_entropy(y_next, target)
        for _, aux in aux_losses.items():
            loss = loss + aux
        return loss

    pred_masked = torch.masked_select(pred, m)
    if model_name in ["dkvmn", "saint"]:
        target = torch.masked_select(r, m).float()
    else:
        target = torch.masked_select(rshft, m)

    return binary_cross_entropy(pred_masked, target)


def _eval_arrays(model_name, pred, r, rshft, m):
    pred = torch.masked_select(pred, m).detach().cpu().numpy()
    if model_name in ["dkvmn", "saint"]:
        target = torch.masked_select(r, m).float().detach().cpu().numpy()
    else:
        target = torch.masked_select(rshft, m).detach().cpu().numpy()
    return target, pred


def _move_batch_to_model_device(model, batch):
    device = next(model.parameters()).device
    return tuple(x.to(device) for x in batch)


def train_model(model_name, model, train_loader, test_loader, num_epochs, opt, ckpt_path):
    aucs = []
    loss_means = []
    max_auc = 0

    for i in range(1, num_epochs + 1):
        epoch_losses = []

        for data in train_loader:
            q, r, qshft, rshft, m = _move_batch_to_model_device(model, data)
            model.train()

            pred = _forward_for_batch(model_name, model, q, r, qshft)
            loss = _train_loss(model_name, model, pred, q, r, qshft, rshft, m)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            for data in test_loader:
                q, r, qshft, rshft, m = _move_batch_to_model_device(model, data)
                model.eval()

                pred = _forward_for_batch(model_name, model, q, r, qshft)
                y_true, y_score = _eval_arrays(model_name, pred, r, rshft, m)

                auc, acc = calc_binary_auc_acc(y_true=y_true, y_score=y_score)
                loss_mean = np.mean(epoch_losses)

                print(
                    "Epoch: {},   Test AUC: {},   Test ACC: {},   Loss Mean: {}"
                    .format(i, auc, acc, loss_mean)
                )

                if auc > max_auc:
                    torch.save(
                        model.state_dict(),
                        os.path.join(ckpt_path, "model.ckpt")
                    )
                    max_auc = auc

                aucs.append(auc)
                loss_means.append(loss_mean)

    return aucs, loss_means
