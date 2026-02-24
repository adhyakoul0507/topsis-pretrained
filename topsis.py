# topsis.py — Pure TOPSIS Algorithm Implementation
import numpy as np
import pandas as pd


def run_topsis(models: list, criteria: list, weights: list) -> pd.DataFrame:
    """
    Run TOPSIS on a list of models.

    Parameters
    ----------
    models   : list of dicts  (each dict has keys matching criteria)
    criteria : list of dicts  with keys: 'key', 'type' ('benefit'|'cost')
    weights  : list of floats (raw, will be normalised internally)

    Returns
    -------
    pd.DataFrame sorted by TOPSIS score descending, with columns:
        name, score, d_best, d_worst, rank, + original criterion columns
    """
    keys   = [c["key"]  for c in criteria]
    types  = [c["type"] for c in criteria]

    # ── Step 1: Build decision matrix (rows=models, cols=criteria)
    X = np.array([[m[k] for k in keys] for m in models], dtype=float)
    n_models, n_criteria = X.shape

    # ── Step 2: Normalize weights
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # ── Step 3: Normalise decision matrix (Euclidean norm per column)
    col_norms = np.sqrt((X ** 2).sum(axis=0))
    col_norms[col_norms == 0] = 1          # avoid div-by-zero
    R = X / col_norms

    # ── Step 4: Weighted normalised matrix
    V = R * w

    # ── Step 5: Ideal best (A+) and ideal worst (A-)
    A_best  = np.where([t == "benefit" for t in types], V.max(axis=0), V.min(axis=0))
    A_worst = np.where([t == "benefit" for t in types], V.min(axis=0), V.max(axis=0))

    # ── Step 6: Euclidean distance to A+ and A-
    D_best  = np.sqrt(((V - A_best)  ** 2).sum(axis=1))
    D_worst = np.sqrt(((V - A_worst) ** 2).sum(axis=1))

    # ── Step 7: Closeness coefficient  Ci = D- / (D+ + D-)
    denom = D_best + D_worst
    denom[denom == 0] = 1e-12
    scores = D_worst / denom

    # ── Step 8: Build results DataFrame
    results = pd.DataFrame(models)
    results["score"]   = scores.round(4)
    results["d_best"]  = D_best.round(5)
    results["d_worst"] = D_worst.round(5)
    results = results.sort_values("score", ascending=False).reset_index(drop=True)
    results.index += 1                     # rank starts at 1
    results.index.name = "rank"

    # Store weighted matrix rows for display
    results["_weighted"] = [V[i].tolist() for i in range(n_models)]

    return results, A_best, A_worst, w
