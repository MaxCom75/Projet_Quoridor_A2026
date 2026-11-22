from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import List, Optional

from game_state_quoridor import GameStateQuoridor

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "logistic_weights.json")

FEATURE_NAMES = [
    "my_dist",
    "opp_dist",
    "my_turn",
    "my_walls",
    "opp_walls",
]

def extract_features(state: GameStateQuoridor, player) -> List[float]:
    opponent = state._opponent(player)
    my_dist = state._shortest_path(player)
    opp_dist = state._shortest_path(opponent)
    my_turn = 1.0 if state.get_active_player().get_id() == player.id else 0.0
    return [
        float(my_dist),
        float(opp_dist),
        my_turn,
        float(state.rep.remaining_walls[player.id]),
        float(state.rep.remaining_walls[opponent.id]),
    ]

_weights_cache: Optional[tuple] = None

def load_weights(path: str = WEIGHTS_PATH) -> tuple:
    global _weights_cache
    if _weights_cache is None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data["feature_names"] != FEATURE_NAMES:
            raise ValueError
        _weights_cache = (data["bias"], data["weights"])
    return _weights_cache

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

def win_probability(state: GameStateQuoridor, player) -> float:
    bias, weights = load_weights()
    z = bias + sum(w * x for w, x in zip(weights, extract_features(state, player)))
    return _sigmoid(z)

def evaluate(state: GameStateQuoridor, player) -> float:
    if state.is_done():
        return 1.0 if state.get_player_score(player) == 1.0 else -1.0
    return 2.0 * win_probability(state, player) - 1.0

def _game_samples(path: str):
    with open(path, encoding="utf-8") as f:
        game = json.load(f)
    summary = game.get("final_summary", {})
    winners = summary.get("winners_id", [])
    if summary.get("status") != "done" or len(winners) != 1:
        return None
    winner = winners[0]
    samples = []
    for step in game["steps"]:
        state = GameStateQuoridor.from_json(step)
        if state.is_done():
            continue
        for player in state.players:
            samples.append((extract_features(state, player), 1.0 if player.get_id() == winner else 0.0))
    return samples or None

def load_dataset(data_dir: str):
    from multiprocessing import Pool

    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json"))
    games = []
    skipped = 0
    with Pool() as pool:
        for i, samples in enumerate(pool.imap(_game_samples, files, chunksize=4)):
            if samples is None:
                skipped += 1
            else:
                games.append(samples)
            if (i + 1) % 200 == 0:
                print(f"  parsed {i + 1}/{len(files)} games", flush=True)
    print(f"Loaded {len(games)} games ({skipped} skipped: unfinished or no single winner)")
    return games

def fit_logistic(X, y, l2: float = 1e-3, iters: int = 50):
    import numpy as np

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = np.hstack([np.ones((len(X), 1)), (X - mean) / std])

    w = np.zeros(Xs.shape[1])
    reg = l2 * len(X) * np.eye(Xs.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        grad = Xs.T @ (p - y) + reg @ w
        hess = (Xs * (p * (1 - p))[:, None]).T @ Xs + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break

    weights = w[1:] / std
    bias = w[0] - float((w[1:] * mean / std).sum())
    return float(bias), [float(v) for v in weights]

def _metrics(p, y):
    import numpy as np

    eps = 1e-12
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return logloss, acc, auc

def _baseline_predict(X):
    import numpy as np

    race = X[:, 1] - X[:, 0] + 0.5 * X[:, 2]
    return 1.0 / (1.0 + np.exp(-race))

def train(data_dir: str, test_fraction: float = 0.2, seed: int = 0, l2: float = 1e-3):
    import numpy as np

    games = load_dataset(data_dir)
    random.Random(seed).shuffle(games)
    n_test = max(1, int(len(games) * test_fraction))
    test_games, train_games = games[:n_test], games[n_test:]

    def to_arrays(gs):
        X = np.array([x for g in gs for x, _ in g], dtype=float)
        y = np.array([lab for g in gs for _, lab in g], dtype=float)
        return X, y

    X_tr, y_tr = to_arrays(train_games)
    X_te, y_te = to_arrays(test_games)
    print(f"Train: {len(train_games)} games / {len(X_tr)} samples   Test: {len(test_games)} games / {len(X_te)} samples")

    bias, weights = fit_logistic(X_tr, y_tr, l2=l2)
    p_te = 1.0 / (1.0 + np.exp(-(bias + X_te @ np.array(weights))))

    ll, acc, auc = _metrics(p_te, y_te)
    bll, bacc, bauc = _metrics(_baseline_predict(X_te), y_te)
    print("\nTest set           log-loss  accuracy   AUC")
    print(f"  logistic model     {ll:.4f}    {acc:.3f}    {auc:.3f}")
    print(f"  race-only baseline {bll:.4f}    {bacc:.3f}    {bauc:.3f}")

    print(f"\n  bias = {bias:+.4f}")
    for name, w in zip(FEATURE_NAMES, weights):
        print(f"  {name:<18} {w:+.4f}")

    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    bias, weights = fit_logistic(X_all, y_all, l2=l2)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "bias": bias,
            "weights": weights,
            "n_games": len(games),
            "n_samples": int(len(X_all)),
            "test_metrics": {"log_loss": ll, "accuracy": acc, "auc": auc},
        }, f, indent=2)
    print(f"\nFinal weights (fit on all {len(games)} games) written to {WEIGHTS_PATH}")

if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "past_game_data"))
