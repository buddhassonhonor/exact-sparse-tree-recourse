"""Core single-tree recourse helpers used by the public benchmark."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

ACTION_EPS = 1e-8


def get_leaf_boxes(clf: DecisionTreeClassifier) -> dict[int, dict[str, dict[int, float]]]:
    tree = clf.tree_
    boxes: dict[int, dict[str, dict[int, float]]] = {}
    stack = [(0, {}, {})]
    while stack:
        node, lb, ub = stack.pop()
        if tree.children_left[node] == tree.children_right[node]:
            boxes[node] = {"lb": dict(lb), "ub": dict(ub)}
            continue
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        left_ub = dict(ub)
        left_ub[feature] = min(left_ub.get(feature, float("inf")), threshold)
        right_lb = dict(lb)
        right_lb[feature] = max(right_lb.get(feature, -float("inf")), threshold)
        stack.append((tree.children_left[node], dict(lb), left_ub))
        stack.append((tree.children_right[node], right_lb, dict(ub)))
    return boxes


def actionable_indices(n_features: int, immutable_idx: np.ndarray) -> np.ndarray:
    mask = np.ones(n_features, dtype=bool)
    mask[immutable_idx] = False
    return np.where(mask)[0]


def nearest_positive_train(x0: np.ndarray, positives: np.ndarray) -> np.ndarray | None:
    if len(positives) == 0:
        return None
    return positives[int(np.argmin(np.linalg.norm(positives - x0, axis=1)))]


def choose_support(x0: np.ndarray, nearest: np.ndarray | None, actionable: np.ndarray,
                   max_changes: int | None) -> np.ndarray:
    if max_changes is None or max_changes >= len(actionable):
        return actionable.copy()
    if nearest is None:
        return actionable[:max_changes].copy()
    order = np.argsort(-np.abs(nearest[actionable] - x0[actionable]))
    return actionable[order[:max_changes]].copy()


def apply_support(x0: np.ndarray, candidate: np.ndarray | None,
                  support: np.ndarray) -> np.ndarray | None:
    if candidate is None:
        return None
    out = x0.copy()
    out[support] = candidate[support]
    return out


def build_nn_target(x0: np.ndarray, nearest: np.ndarray | None,
                    support: np.ndarray) -> np.ndarray | None:
    return apply_support(x0, nearest, support)


def constrained_repair(clf: DecisionTreeClassifier, candidate: np.ndarray | None,
                       target: np.ndarray | None, support: np.ndarray,
                       steps: int) -> np.ndarray | None:
    if candidate is None:
        return None
    if int(clf.predict(candidate.reshape(1, -1))[0]) == 1:
        return candidate
    if target is None or int(clf.predict(target.reshape(1, -1))[0]) != 1 or steps <= 0:
        return None
    lo, hi = 0.0, 1.0
    best = target.copy()
    for _ in range(max(1, steps)):
        mid = (lo + hi) / 2
        probe = candidate.copy()
        probe[support] = (1 - mid) * candidate[support] + mid * target[support]
        if int(clf.predict(probe.reshape(1, -1))[0]) == 1:
            best, hi = probe, mid
        else:
            lo = mid
    return best


def scaled_cost(x0: np.ndarray, candidate: np.ndarray, scale: np.ndarray,
                metric: str) -> float:
    delta = (candidate.astype(np.float64) - x0.astype(np.float64)) / scale
    return float(np.sum(np.abs(delta))) if metric == "weighted_l1" else float(np.linalg.norm(delta))


def changed_feature_indices(x0: np.ndarray, candidate: np.ndarray | None) -> np.ndarray:
    if candidate is None:
        return np.array([], dtype=int)
    return np.where(np.abs(candidate.astype(np.float64) - x0.astype(np.float64)) > ACTION_EPS)[0]


def format_changed_features(x0: np.ndarray, xcf: np.ndarray | None,
                            feature_names: Sequence[str], max_items: int = 6) -> str:
    indices = changed_feature_indices(x0, xcf)
    pieces = [f"{feature_names[j]}:{float(xcf[j] - x0[j]):+.3f}" for j in indices[:max_items]]
    if len(indices) > max_items:
        pieces.append(f"+{len(indices) - max_items} more")
    return " | ".join(pieces)


def leaf_centroid_candidate(x0: np.ndarray, Xtr: np.ndarray, leaf_ids: np.ndarray,
                            positive_leaves: Sequence[int]) -> np.ndarray | None:
    best_distance = float("inf")
    best = None
    for leaf in positive_leaves:
        members = Xtr[leaf_ids == leaf]
        if len(members) == 0:
            continue
        candidate = np.mean(members, axis=0).astype(np.float32)
        distance = float(np.linalg.norm(candidate.astype(np.float64) - x0.astype(np.float64)))
        if distance < best_distance:
            best_distance, best = distance, candidate
    return best
