"""Which neurons can you drive and still be heard at the output?

Weights are normalised so each neuron's incoming synapses sum to 1. Median
in-degree is ~153, so one presynaptic spike moves a neuron by ~1/153 of
threshold. Input injected into a random 0.6% of the brain therefore attenuates
to nothing within a couple of hops, and the descending neurons never see it.

So the input population has to be chosen by the wiring. Candidates:
    all2k / all30k   random subsets of the whole brain (the textbook W_in)
    visual_proj      the 9,201 visual projection neurons
    loom_chase       LPLC2, LC4, LPLC1, LC10a - the route fly.ai proved reaches DNp01
    pre_desc         the neurons that synapse DIRECTLY onto descending neurons

Diagnostic per config: the largest |correlation| between the input u(t) and any
descending neuron's trace, and the held-out R^2 for decoding u(t) at zero lag.
If the input reaches the output at all, both go up.

    python flybench/probe3.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fly_brain import DATA, FlyBrain  # noqa: E402

TRACE_TAU = 0.1
WASHOUT, TRAIN, TEST = 400, 1200, 800
TOTAL = WASHOUT + TRAIN + TEST
HERE = Path(__file__).resolve().parent


def standardise(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    pos = sd[sd > 0]
    return mu, np.maximum(sd, 0.05 * pos.mean() if len(pos) else 1.0)


def r2(Xn, y, tr, te, lam=1e-2):
    A = Xn[tr].T @ Xn[tr] + lam * len(Xn[tr]) * np.eye(Xn.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xn[tr].T @ y[tr])
    return float(1 - ((y[te] - Xn[te] @ w) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())


def pools(brain, W):
    out = {}
    rng = np.random.default_rng(7)
    out["all2k"] = rng.choice(brain.n, 2000, replace=False)
    out["all30k"] = rng.choice(brain.n, 30000, replace=False)
    out["visual_proj"] = np.flatnonzero(brain.superclass == "visual_projection")
    types = ["LPLC2", "LC4", "LPLC1", "LC10a"]
    out["loom_chase"] = np.flatnonzero(np.isin(brain.cell_type, types))
    desc = np.flatnonzero(brain.superclass == "descending_neuron")
    Wc = W.tocsc()
    pre = np.unique(Wc[:, desc].indices)
    out["pre_desc"] = pre[:30000] if len(pre) > 30000 else pre
    return out


def main():
    brain = FlyBrain(device="cpu")
    W = sparse.load_npz(DATA / "weights.npz")
    desc = np.flatnonzero(brain.superclass == "descending_neuron")
    slot = np.full(brain.n, -1, np.int64)
    slot[desc] = np.arange(len(desc))
    P = pools(brain, W)
    u = (np.random.default_rng(1234).random(TOTAL).astype(np.float32) - 0.5) * 2.0
    tr, te = slice(WASHOUT, WASHOUT + TRAIN), slice(WASHOUT + TRAIN, TOTAL)
    decay = np.float32(np.exp(-brain.dt / TRACE_TAU))
    print(f"populations: " + ", ".join(f"{k}={len(v):,}" for k, v in P.items()))
    print(f"{'pool':>12} {'n':>7} {'scale':>6} {'max|corr|':>10} {'mean|corr|':>10} {'inst R2':>9} {'desc Hz':>8}")
    rows = []
    for name, idx in P.items():
        rng = np.random.default_rng(7)
        enc = np.asarray(idx, np.int64)
        w = rng.choice(np.array([-1.0, 1.0], np.float32), size=len(enc))
        for scale in (0.05, 0.15, 0.4):
            brain.tonic, brain.gain = 0.14, 3.0
            brain.reset(64)
            t0 = time.time()
            X = np.empty((TOTAL, len(desc)), np.float32)
            trace = np.zeros(len(desc), np.float32)
            for t, ut in enumerate(u):
                brain.v[enc] += np.float32(ut * scale) * w
                fired = brain.step()
                trace *= decay
                s = slot[fired]
                trace[s[s >= 0]] += 1.0
                X[t] = trace
            c = np.array([np.corrcoef(X[WASHOUT:, j], u[WASHOUT:])[0, 1] for j in range(X.shape[1])])
            c = np.nan_to_num(c)
            mu, sd = standardise(X[tr])
            Xn = (X - mu) / sd
            inst = r2(Xn, u, tr, te)
            dhz = float(X[WASHOUT:].mean(0).mean()) * (1 - decay) / brain.dt
            rows.append(dict(pool=name, n=len(enc), scale=scale, max_corr=round(float(abs(c).max()), 4),
                             mean_corr=round(float(abs(c).mean()), 4), inst_r2=round(inst, 4),
                             desc_hz=round(dhz, 2), wall_s=round(time.time() - t0, 1)))
            print(f"{name:>12} {len(enc):7,} {scale:6.2f} {abs(c).max():10.4f} {abs(c).mean():10.4f} "
                  f"{inst:9.4f} {dhz:8.2f}")
    rows = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}
            for r in rows]
    (HERE / "probe3_results.json").write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["inst_r2"])
    print(f"\nbest: {best['pool']} scale={best['scale']} inst R2={best['inst_r2']:.3f} "
          f"max|corr|={best['max_corr']:.4f}")


if __name__ == "__main__":
    main()
