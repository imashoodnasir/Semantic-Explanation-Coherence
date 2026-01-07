from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_prob, num_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))

    # auc: binary or ovr for multiclass
    try:
        if num_classes == 2:
            # y_prob shape: (n,2) or (n,)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                score = roc_auc_score(y_true, y_prob[:,1])
            else:
                score = roc_auc_score(y_true, y_prob)
        else:
            score = roc_auc_score(y_true, y_prob, multi_class="ovr")
        out["auc"] = float(score)
    except Exception:
        out["auc"] = float("nan")

    return out
