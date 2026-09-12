"""Find the operating point: where is the brain responsive but not seizing?

The LIF neurons sit at v = 5*tonic at rest (tonic 0.14 -> 0.70, threshold 1.0).
That leaves only 0.30 of headroom, so an injection of 0.3+ volts fires every
neuron downstream and the network saturates. The usable input range is tiny and
it has to be found, not assumed.

Also: the drive is centred (u - mean). A DC offset just parks the network
against the threshold and carries no information.

    python flybench/probe2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fly_brain import FlyBrain  # noqa: E402

TRACE_TAU = 0.1
WASHOUT, TRAIN, TEST = 400, 1200, 800
TOTAL = WASHOUT + TRAIN + TEST
HERE = Path(__file__).resolve().parent


def standardise(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    pos = sd[sd > 0]
    floor = 0.05 * pos.mean() if len(pos) else 1.0
    return mu, np.maximum(sd, floor)


def r2(Xn, y, tr, te, lam=1e-2):
    A = Xn[tr].T @ Xn[tr] + lam * len(Xn[tr]) * np.eye(Xn.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xn[tr].T @ y[tr])
    return float(1 - ((y[te] - Xn[te] @ w) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())


def run(brain, enc, w, u, scale, slot, n_desc):
    X = np.empty((len(u), n_desc), np.float32)
    trace = np.zeros(n_desc, np.float32)
    decay = np.float32(np.exp(-brain.dt / TRACE_TAU))
    total_spikes = 0
    for t, ut in enumerate(u):
        brain.v[enc] += np.float32(ut * scale) * w
        fired = brain.step()
        total_spikes += len(fired)
        trace *= decay
        s = slot[fired]
        trace[s[s >= 0]] += 1.0
        X[t] = trace
    return X, total_spikes / len(u) / brain.dt


def main():
    brain = FlyBrain(device="cpu")
    desc = np.flatnonzero(brain.superclass == "descending_neuron")
    slot = np.full(brain.n, -1, np.int64)
    slot[desc] = np.arange(len(desc))
    all_idx = np.arange(brain.n)
    tr, te = slice(WASHOUT, WASHOUT + TRAIN), slice(WASHOUT + TRAIN, TOTAL)
    rng = np.random.default_rng(7)
    enc = rng.choice(all_idx, size=2000, replace=False).astype(np.int64)
    w = rng.choice(np.array([-1.0, 1.0], np.float32), size=len(enc))
    u = (np.random.default_rng(1234).random(TOTAL).astype(np.float32) - 0.5) * 2.0
    rows = []
    print(f"{'tonic':>6} {'gain':>5} {'scale':>6} {'rest Hz':>8} {'driven Hz':>10} {'desc Hz':>8} {'inst R2':>9} {'MC':>7}")
    for tonic in (0.06, 0.10, 0.14):
        brain.tonic = tonic
        for gain in (2.0, 3.0):
            brain.gain = gain
            # resting rate with no injection
            brain.reset(64)
            X0, rest = run(brain, enc[:1], w[:1] * 0.0, np.zeros(600, np.float32), 0.0, slot, len(desc))
            for scale in (0.0, 0.02, 0.05, 0.1, 0.2):
                brain.reset(64)
                X, net = run(brain, enc, w, u, scale, slot, len(desc))
                mu, sd = standardise(X[tr])
                Xn = (X - mu) / sd
                inst = r2(Xn, u, tr, te)
                mc = sum(max(0.0, r2(Xn, np.roll(u, k), tr, te)) for k in range(1, 11))
                dhz = float(X[WASHOUT:].sum(0).mean() / (1 - np.exp(-brain.dt / TRACE_TAU)) / brain.dt)
                rows.append(dict(tonic=tonic, gain=gain, scale=scale, rest_hz=round(rest, 2),
                                 net_hz=round(net, 1), desc_hz=round(dhz, 2),
                                 inst_r2=round(inst, 4), mc=round(mc, 3)))
                print(f"{tonic:6.2f} {gain:5.1f} {scale:6.3f} {rest:8.2f} {net:10.1f} "
                      f"{dhz:8.2f} {inst:9.4f} {mc:7.3f}")
    (HERE / "probe2_results.json").write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r["scale"] > 0]
    best = max(ok, key=lambda r: r["inst_r2"])
    print(f"\nbest instantaneous decode: tonic={best['tonic']} gain={best['gain']} "
          f"scale={best['scale']} -> inst R2 {best['inst_r2']:.3f}, MC {best['mc']:.2f}, "
          f"desc {best['desc_hz']} Hz, net {best['net_hz']} Hz")


if __name__ == "__main__":
    main()
