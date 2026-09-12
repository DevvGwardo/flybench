"""The readout is the bottleneck, not the wiring.

probe3 showed max|corr| = 0.48 between the input and a single descending neuron
when the input drives the cells that synapse directly onto them. A readout that
used just that one neuron would score R^2 ~ 0.23. So the signal is there and the
linear readout over all 1,314 features is throwing it away.

Suspects: the ridge penalty was far too small for 1,300 noise features, the
features drift slowly between train and test, and there were too few samples.

This picks the readout properly, on a validation split, and compares three
readouts on the same recorded activity:
    ridge       all 1,314 features, lambda chosen on validation
    best-1      the single descending neuron most correlated with the input
    pca+k       top-k principal components, then ridge

    python flybench/probe5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fly_brain import DATA, FlyBrain  # noqa: E402

TRACE_TAU = 0.1
WASHOUT, TRAIN, VAL, TEST = 500, 3500, 1000, 1000
TOTAL = WASHOUT + TRAIN + VAL + TEST
HERE = Path(__file__).resolve().parent
LAMBDAS = np.logspace(-4, 5, 10)


def fit(Xn, y, tr, lam):
    A = Xn[tr].T @ Xn[tr] + lam * len(Xn[tr]) * np.eye(Xn.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xn[tr].T @ y[tr])


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    import scipy.sparse as sp
    brain = FlyBrain(device="cpu")
    W = sp.load_npz(DATA / "weights.npz")
    desc = np.flatnonzero(brain.superclass == "descending_neuron")
    slot = np.full(brain.n, -1, np.int64)
    slot[desc] = np.arange(len(desc))
    enc = np.unique(W.tocsc()[:, desc].indices)[:30000].astype(np.int64)
    rng = np.random.default_rng(7)
    w = rng.choice(np.array([-1.0, 1.0], np.float32), size=len(enc))
    u = (np.random.default_rng(1234).random(TOTAL).astype(np.float32) - 0.5) * 2.0
    brain.tonic, brain.gain = 0.14, 3.0
    brain.reset(64)
    decay = np.float32(np.exp(-brain.dt / TRACE_TAU))
    X = np.empty((TOTAL, len(desc)), np.float32)
    trace = np.zeros(len(desc), np.float32)
    for t, ut in enumerate(u):
        brain.v[enc] += np.float32(ut * 0.4) * w
        fired = brain.step()
        trace *= decay
        s = slot[fired]
        trace[s[s >= 0]] += 1.0
        X[t] = trace
    np.save(HERE / "activity_pre_desc.npy", X.astype(np.float16))

    tr = slice(WASHOUT, WASHOUT + TRAIN)
    va = slice(WASHOUT + TRAIN, WASHOUT + TRAIN + VAL)
    te = slice(WASHOUT + TRAIN + VAL, TOTAL)
    print(f"recorded {X.shape}, input corr max {max(abs(np.corrcoef(X[WASHOUT:,j],u[WASHOUT:])[0,1]) for j in range(0,X.shape[1],7)):.3f}")

    # ---- 1. best single neuron, chosen on TRAIN only
    c = np.array([abs(np.corrcoef(X[tr, j], u[tr])[0, 1]) for j in range(X.shape[1])])
    c = np.nan_to_num(c)
    j = int(c.argmax())
    mu, sd = X[tr, j].mean(), X[tr, j].std() + 1e-6
    pred = ((X[te, j] - mu) / sd) * np.sign(np.corrcoef(X[tr, j], u[tr])[0, 1])
    yte = (u[te] - u[tr].mean()) / u[tr].std()
    print(f"best-1 neuron (train corr {c[j]:.3f}): test R2 = {r2(pred, yte):.4f}")

    # ---- 2. ridge, lambda on validation
    mu, sd = X[tr].mean(0), X[tr].std(0)
    pos = sd[sd > 0]
    sd = np.maximum(sd, 0.05 * pos.mean())
    Xn = (X - mu) / sd
    ym, ys = u[tr].mean(), u[tr].std()
    y = (u - ym) / ys
    best = None
    for lam in LAMBDAS:
        ww = fit(Xn, y, tr, lam)
        vr = r2(Xn[va] @ ww, y[va])
        print(f"  ridge lam={lam:9.4f} val R2 {vr:8.4f}")
        if best is None or vr > best[0]:
            best = (vr, lam, ww)
    vr, lam, ww = best
    print(f"ridge lam={lam:g} val {vr:.4f} -> TEST R2 {r2(Xn[te] @ ww, y[te]):.4f}")

    # ---- 3. PCA + ridge
    mu2 = X[tr].mean(0)
    vt = np.linalg.svd(X[tr] - mu2, full_matrices=False)[2]
    for k in (5, 20, 50, 100, 200):
        P = vt[:k].T.astype(np.float32)
        Z = (X - mu2) @ P
        zmu, zsd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
        Zn = (Z - zmu) / zsd
        b = None
        for lm in LAMBDAS:
            ww = fit(Zn, y, tr, lm)
            r = r2(Zn[va] @ ww, y[va])
            if b is None or r > b[0]:
                b = (r, lm, ww)
        print(f"  pca k={k:4d} lam={b[1]:8.4f} val {b[0]:7.4f} -> TEST R2 {r2(Zn[te] @ b[2], y[te]):.4f}")


if __name__ == "__main__":
    main()
