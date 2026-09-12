"""The background drive hypothesis.

Reproducing fly.ai's own inject.py result needs `eye_drive=0.45` on every
photoreceptor at every step. Without it, stimulating the looming detectors does
nothing at all. So the brain is only responsive when it is already being driven
- ongoing activity is what puts it near threshold, and the task signal is a
modulation on top.

That is also how a real nervous system works, so this is not a hack: a constant
background on the fly's own visual system, plus the task signal modulating the
neurons the wiring actually listens to.

    python flybench/probe4.py
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
LOOM_CHASE = ["LPLC2", "LC4", "LPLC1", "LC10a"]


def standardise(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    pos = sd[sd > 0]
    return mu, np.maximum(sd, 0.05 * pos.mean() if len(pos) else 1.0)


def r2(Xn, y, tr, te, lam=1e-2):
    A = Xn[tr].T @ Xn[tr] + lam * len(Xn[tr]) * np.eye(Xn.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xn[tr].T @ y[tr])
    return float(1 - ((y[te] - Xn[te] @ w) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())


def main():
    brain = FlyBrain(device="cpu")
    W = sparse.load_npz(DATA / "weights.npz")
    desc = np.flatnonzero(brain.superclass == "descending_neuron")
    slot = np.full(brain.n, -1, np.int64)
    slot[desc] = np.arange(len(desc))
    rng = np.random.default_rng(7)
    pools = {
        "loom_chase": np.flatnonzero(np.isin(brain.cell_type, LOOM_CHASE)),
        "all30k": rng.choice(brain.n, 30000, replace=False),
        "visual_proj": np.flatnonzero(brain.superclass == "visual_projection"),
        "pre_desc": np.unique(W.tocsc()[:, desc].indices)[:30000],
    }
    eye = brain.visual
    u = (np.random.default_rng(1234).random(TOTAL).astype(np.float32) - 0.5) * 2.0
    tr, te = slice(WASHOUT, WASHOUT + TRAIN), slice(WASHOUT + TRAIN, TOTAL)
    decay = np.float32(np.exp(-brain.dt / TRACE_TAU))
    print(f"pools: " + ", ".join(f"{k}={len(v):,}" for k, v in pools.items()))
    print(f"{'bg':>10} {'enc pool':>12} {'n':>7} {'scale':>6} {'max|corr|':>10} {'inst R2':>9} {'desc Hz':>8} {'net Hz':>8}")
    rows = []
    configs = [
        (None, "loom_chase", 0.8), (None, "all30k", 0.15),
        (0.28, "loom_chase", 0.8), (0.28, "loom_chase", 0.3),
        (0.28, "all30k", 0.15), (0.28, "all30k", 0.4),
        (0.28, "visual_proj", 0.3), (0.28, "pre_desc", 0.3),
        (0.5, "all30k", 0.3),
    ]
    for bg, pname, scale in configs:
        enc = np.asarray(pools[pname], np.int64)
        w = rng.choice(np.array([-1.0, 1.0], np.float32), size=len(enc))
        brain.tonic, brain.gain = 0.14, 3.0
        brain.reset(64)
        t0 = time.time()
        X = np.empty((TOTAL, len(desc)), np.float32)
        trace = np.zeros(len(desc), np.float32)
        net = 0
        for t, ut in enumerate(u):
            if bg:
                brain.v[eye] += np.float32(bg)
            brain.v[enc] += np.float32(ut * scale) * w
            fired = brain.step()
            net += len(fired)
            trace *= decay
            s = slot[fired]
            trace[s[s >= 0]] += 1.0
            X[t] = trace
        c = np.nan_to_num(np.array([np.corrcoef(X[WASHOUT:, j], u[WASHOUT:])[0, 1]
                                    for j in range(X.shape[1])]))
        mu, sd = standardise(X[tr])
        Xn = (X - mu) / sd
        inst = r2(Xn, u, tr, te)
        dhz = float(X[WASHOUT:].mean(0).mean()) * (1 - decay) / brain.dt
        nhz = net / TOTAL / brain.dt
        rows.append(dict(bg=bg or 0.0, pool=pname, n=int(len(enc)), scale=scale,
                         max_corr=float(abs(c).max()), inst_r2=inst,
                         desc_hz=float(dhz), net_hz=float(nhz), wall_s=round(time.time() - t0, 1)))
        print(f"{str(bg):>10} {pname:>12} {len(enc):7,} {scale:6.2f} {abs(c).max():10.4f} "
              f"{inst:9.4f} {dhz:8.2f} {nhz:8.0f}")
    (HERE / "probe4_results.json").write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["inst_r2"])
    print(f"\nbest: bg={best['bg']} {best['pool']} scale={best['scale']} "
          f"inst R2={best['inst_r2']:.3f} max|corr|={best['max_corr']:.4f}")


if __name__ == "__main__":
    main()
