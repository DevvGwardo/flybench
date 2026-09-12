"""FlyDrive race: sober drivers only. Does the real wiring steer better than a
random network of the same size?

drive.py answers "what do global parameter changes do to the fly". This answers
the harder one: is the connectome's *topology* worth anything behind the wheel?

Every model gets the identical treatment:
  * its own encoder, taken from its own wiring - the cells that synapse onto the
    same four motor neurons (steer_L/R, forward_L/R);
  * its own decoder, cloned from the same PD teacher on the same roads;
  * its own decoder gain, tuned on the same held-out roads;
  * the same four evaluation roads, starts and seeds.

So the only difference between two rows is which network is sitting in the
driver's seat. `linear` is the teacher itself - a PD controller with no brain at
all, which is the ceiling.

    python flybench/drive_race.py --steps 2400 --seeds 4 --train-eps 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from arena import load_real, build_control  # noqa: E402
from drive import (SOBER, calibrate_channels, motor_encoder, r2,  # noqa: E402
                   run_episode, fit_readout, teacher, Road, Car, V, DT, LANE)

OUT = HERE / "drive_race.json"
GAINS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3)


def fit_decoder(W, meta, desc, enc, chan, w, steps, n_eps, chmu, chsd, seed0=100):
    """Behaviour-clone the PD teacher on sober runs. Same recipe for every model."""
    Xs, ys = [], []
    for i in range(n_eps):
        rng = np.random.default_rng(seed0 + i)
        ep = run_episode(W, meta, desc, enc, chan, w, SOBER, steps, seed0 + i,
                         road_phase=float(rng.uniform(0, 400)),
                         x0=float(np.random.default_rng(seed0 + i + 900).uniform(-1.2, 1.2)),
                         record=True, chmu=chmu, chsd=chsd, teacher_noise=0.18)
        Xs.append(ep["feats"][200:])
        ys.append(ep["targets"][200:])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    tr = slice(0, int(len(X) * 0.7))
    va = slice(int(len(X) * 0.7), len(X))
    mu, sd = X[tr].mean(0), X[tr].std(0)
    pos = sd[sd > 0]
    sd = np.maximum(sd, 0.05 * pos.mean() if len(pos) else 1.0)
    Xn = (X - mu) / sd
    best = max(((r2(Xn[va] @ fit_readout(Xn, y, tr, lam), y[va]), lam)
                for lam in (0.1, 1.0, 10.0, 100.0, 1000.0)), key=lambda c: c[0])
    return fit_readout(Xn, y, tr, best[1]), mu, sd, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--train-eps", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=["fly", "random", "shuffled"])
    args = ap.parse_args()

    t0 = time.time()
    W_fly, sc = load_real()
    meta = np.load(Path.home() / "fly-data" / "brain.npz", allow_pickle=True)
    desc = np.flatnonzero(sc == "descending_neuron")
    chmu, chsd = calibrate_channels()

    seeds = [(i, float(np.random.default_rng(200 + i).uniform(0, 400)),
              float(np.random.default_rng(700 + i).uniform(-1.0, 1.0)))
             for i in range(args.seeds)]
    val = [(i, float(np.random.default_rng(800 + i).uniform(0, 400)),
            float(np.random.default_rng(900 + i).uniform(-1.0, 1.0))) for i in range(2)]

    rows = []
    for model in args.models:
        W = W_fly if model == "fly" else build_control(model)
        enc, chan, w = motor_encoder(W, meta)
        print(f"\n=== {model} ({W.nnz:,} synapses, encoder {len(enc)} cells) ===")
        readout, mu, sd, val_r2 = fit_decoder(W, meta, desc, enc, chan, w,
                                              args.steps, args.train_eps, chmu, chsd)

        def run(i, ph, x0, g, seed_off=300):
            return run_episode(W, meta, desc, enc, chan, w, SOBER, args.steps,
                               seed_off + i, ph, x0, readout=readout, trace_tau=0.1,
                               zmu=mu, zsd=sd, decoder_gain=g, chmu=chmu, chsd=chsd)

        g_scores = [(float(np.mean([run(i, ph, x0, g, 800)["mean_abs_err"]
                                    for i, ph, x0 in val])), g) for g in GAINS]
        best_err, best_g = min(g_scores)
        print(f"  decoder val R2 {val_r2:.3f}   gain x{best_g:g} "
              f"(mean|e| {best_err:.3f} m)")
        print("  gain sweep: " + "  ".join(f"x{g:g}->{m:.2f}" for m, g in g_scores))

        accs = [run(i, ph, x0, best_g) for i, ph, x0 in seeds]
        per = [a["mean_abs_err"] for a in accs]
        lost = sum(1 for v in per if v > 1.0)
        row = dict(model=model, decoder_val_r2=round(val_r2, 3), gain=best_g,
                   encoder_n=int(len(enc)),
                   mean_abs_err=round(float(np.mean(per)), 3),
                   median_abs_err=round(float(np.median(per)), 3),
                   lost_road=lost, seeds=len(accs),
                   in_lane=round(float(np.mean([a["in_lane"] for a in accs])), 3),
                   crashes=sum(1 for a in accs if a["crash_at"] >= 0),
                   spikes_per_step=round(float(np.mean([a["spikes_per_step"] for a in accs])), 0),
                   travelled=round(float(np.mean([a["travelled"] for a in accs])), 0),
                   per_seed=[round(v, 2) for v in per])
        rows.append(row)
        print(f"  {model:>9}: median|e| {row['median_abs_err']:6.3f} m  "
              f"mean {row['mean_abs_err']:6.3f}  lost {lost:2d}/{len(accs)}  "
              f"in-lane {row['in_lane']:.2f}  {row['spikes_per_step']:7.0f} spikes/step")

    # the ceiling: the PD teacher itself, no brain, on the same roads
    tch = []
    for i, ph, x0 in seeds:  # the ceiling: a PD controller with no brain
        car = Car(Road(phase=ph), offset=x0)
        errs = []
        for _ in range(args.steps):
            car.drive(teacher(car))
            errs.append(car.errors()[0])
        tch.append(float(np.abs(np.array(errs)).mean()))
    rows.append(dict(model="linear (PD teacher)", mean_abs_err=round(float(np.mean(tch)), 3),
                     in_lane=1.0, crashes=0, spikes_per_step=0, travelled=args.steps * V * DT,
                     per_seed=[round(t, 3) for t in tch]))
    print(f"\n  {'PD teacher':>9}: mean|e| {np.mean(tch):6.3f} m  (no brain at all)")

    OUT.write_text(json.dumps(dict(steps=args.steps, seeds=args.seeds, rows=rows), indent=2))
    print(f"\n{time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
