import numpy as np

from sklearn import metrics


def calc_binary_auc_acc(y_true, y_score, threshold=0.5):
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    acc = metrics.accuracy_score(
        y_true=y_true,
        y_pred=np.where(y_score >= threshold, 1, 0)
    )

    return auc, acc
