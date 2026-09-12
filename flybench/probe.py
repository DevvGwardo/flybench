"""Where can you actually get a signal INTO this brain?

fly.ai's own finding is that the photoreceptor route dies at the lamina: real
lamina neurons use graded signals and photoreceptors release histamine, which is
inhibitory. So a naive encoder onto `ol_sensory` (which contains the
photoreceptors) may be a dead end.

This searches every plausible input population and dynamics regime, and reports
two things per configuration:
    instant R^2  - can a linear readout of the 1,314 descending neurons tell what
                   the input is RIGHT NOW (zero lag)? This is the gate: if the
                   input does not reach the output cables at all, nothing else
                   can work.
    memory       - memory capacity, sum of R^2 for recovering u(t-k), k=1..20.

    python flybench/probe.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fly_brain import FlyBrain  # noqa: E402

TRACE_TAU = 0.1
WASHOUT, TRAIN, TEST = 400, 1200, 800
DELAYS = 20
K_ENC = 1000
POOLS = ["vnc_sensory", "cb_sensory", "ascending_neuron", "sensory_ascending", "ol_sensory"]
HERE = Path(__file__).resolve().parent


def standardise(Xtr):
    """Floor the scale so near-silent neurons cannot blow the readout up."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    pos = sd[sd > 0]
    floor = 0.05 * pos.mean() if len(pos) else 1.0
    return mu, np.maximum(sd, floor)


def ridge_r2(Xn, y, tr, te, lam=1e-2):
    A = Xn[tr].T @ Xn[tr] + lam * len(Xn[tr]) * np.eye(Xn.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xn[tr].T @ y[tr])
    pred = Xn[te] @ w
    return float(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())


def record(brain, enc, w, u, scale, decay, slot, n_desc):
    out = np.empty((len(u), n_desc), np.float32)
    trace = np.zeros(n_desc, np.float32)
    for t, ut in enumerate(u):
        brain.v[enc] += np.float32(ut * scale) * w
        fired = brain.step()
        trace *= decay
        s = slot[fired]
        trace[s[s >= 0]] += 1.0
        out[t] = trace
    return out


def main():
    total = WASHOUT + TRAIN + TEST
    brain = FlyBrain(device="cpu")
    decay = np.float32(np.exp(-brain.dt / TRACE_TAU))
    desc = np.flatnonzero(brain.superclass == "descending_neuron")
    slot = np.full(brain.n, -1, np.int64)
    slot[desc] = np.arange(len(desc))
    u = np.random.default_rng(1234).random(total).astype(np.float32)
    tr, te = slice(WASHOUT, WASHOUT + TRAIN), slice(WASHOUT + TRAIN, total)
    print(f"{total} steps, {len(desc)} descending neurons, {K_ENC} input cells")
    print(f"{'pool':>18} {'tonic':>6} {'gain':>5} {'scale':>6} {'inst R2':>9} {'mem cap':>8} {'Hz':>7}")
    rows = []
    for pool, tonic, gain, scale in itertools.product(
            POOLS, (0.10, 0.14), (3.0, 6.0), (0.3, 1.0)):
        if brain.superclass is None or not (brain.superclass == pool).any():
            continue
        brain.tonic, brain.gain = tonic, gain
        rng = np.random.default_rng(7)
        idx = np.flatnonzero(brain.superclass == pool)
        enc = rng.choice(idx, size=min(K_ENC, len(idx)), replace=False).astype(np.int64)
        w = rng.choice(np.array([-1.0, 1.0], np.float32), size=len(enc))
        brain.reset(64)
        t0 = time.time()
        X = record(brain, enc, w, u, scale, decay, slot, len(desc))
        mu, sd = standardise(X[tr])
        Xn = (X - mu) / sd
        inst = ridge_r2(Xn, u, tr, te)
        mc = 0.0
        for k in range(1, DELAYS + 1):
            mc += max(0.0, ridge_r2(Xn, np.roll(u, k), tr, te))
        hz = float((X[WASHOUT:].sum(0) / (len(u) - WASHOUT)) .mean() / (1 - decay)) / brain.dt
        rows.append(dict(pool=pool, tonic=tonic, gain=gain, scale=scale,
                         inst_r2=round(inst, 4), memory_capacity=round(mc, 3),
                         mean_hz=round(hz, 1), wall_s=round(time.time() - t0, 1)))
        print(f"{pool:>18} {tonic:6.2f} {gain:5.1f} {scale:6.2f} {inst:9.4f} {mc:8.3f} {hz:7.1f}")
    (HERE / "probe_results.json").write_text(json.dumps(rows, indent=2))
    rows = [r for r in rows if r["inst_r2"] > 0]
    best = max(rows, key=lambda r: r["memory_capacity"]) if rows else None
    print()
    if not best:
        print("NO ROUTE IN. Input never reaches the descending neurons in any configuration.")
    else:
        print(f"BEST: {best['pool']} tonic={best['tonic']} gain={best['gain']} scale={best['scale']}")
        print(f"  instant R2 {best['inst_r2']:.3f} | memory capacity {best['memory_capacity']:.2f} delays "
              f"| {best['mean_hz']} Hz")


if __name__ == "__main__":
    main()
