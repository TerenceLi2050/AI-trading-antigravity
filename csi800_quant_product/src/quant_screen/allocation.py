import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def hierarchical_weights(returns: pd.DataFrame) -> pd.Series | None:
    clean = returns.dropna()
    names = clean.columns
    if len(names) == 1:
        return pd.Series(1.0, index=names)
    if len(clean) < max(50, 2 * len(names)):
        return None
    covariance = LedoitWolf().fit(clean).covariance_
    diagonal = np.diag(covariance)
    if (diagonal <= 0).any():
        return None
    corr = np.clip(covariance / np.sqrt(np.outer(diagonal, diagonal)), -1, 1)
    distance = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    weights = np.ones(len(names))

    def variance(cluster: np.ndarray) -> float:
        subset = covariance[np.ix_(cluster, cluster)]
        inverse = 1 / np.diag(subset)
        normalized = inverse / inverse.sum()
        return float(normalized @ subset @ normalized)

    pending = [order]
    while pending:
        group = pending.pop()
        if len(group) < 2:
            continue
        first, second = np.split(group, [len(group) // 2])
        a, b = variance(first), variance(second)
        weights[first] *= b / (a + b)
        weights[second] *= a / (a + b)
        pending.extend([first, second])
    return pd.Series(weights, index=names)
