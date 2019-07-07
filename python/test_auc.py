import numpy as np
import sklearn
from sklearn import metrics
import sys

def eval_auc(pred_fname, label_fname, label_idx):
    pred = np.genfromtxt(pred_fname, delimiter=",", dtype=np.float)
    label = np.genfromtxt(label_fname, delimiter=",", dtype=np.float)[:, label_idx]
    pred = 1.0 / (1.0 + np.exp(-pred))
    print(metrics.roc_auc_score(label, pred))

def eval_rmse(pred_fname, label_fname, label_idx):
    pred = np.genfromtxt(pred_fname, delimiter=",", dtype=np.float)
    label = np.genfromtxt(label_fname, delimiter=",", dtype=np.float)[:, label_idx]
    print(np.sqrt(np.mean((pred - label) ** 2)))


if __name__ == "__main__":
    if sys.argv[4] == "auc":
        eval_auc(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif sys.argv[4] == "rmse":
        eval_rmse(sys.argv[1], sys.argv[2], int(sys.argv[3]))
