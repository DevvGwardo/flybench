"""Does the fly's prediction advantage survive when activity is matched?

The fly connectome runs at ~16,000 spikes/step and the random control at
~30,000, with the same synapse count. So the fly's better one-step prediction
could just be an operating-point artifact: less activity, less mixing, a slower
and more predictable reservoir.

This sweeps the global synaptic gain for both networks, records the resulting
activity level, and scores the same task at each setting. If the fly only wins
where it is quieter, the honest conclusion is "quieter is better", not "the
connectome is better".

    python flybench/probe6.py --task mackey
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena import (Reservoir, build_control, encoder_for, load_real,  # noqa: E402
                   make_task, run_reservoir, score_counts, spans)

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mackey")
    ap.add_argument("--gains", nargs="+", type=float, default=[2.0, 3.0, 5.0, 8.0, 12.0])
    args = ap.parse_args()

    W_real, superclass = load_real()
    Ws = {"fly": W_real, "random": build_control("random"), "shuffled": build_control("shuffled")}
    desc = np.flatnonzero(superclass == "descending_neuron")
    task = make_task(args.task)
    sp = spans(len(task["u"]))
    print(f"task {args.task}, {len(task['u'])} steps")
    print(f"{'model':>9} {'gain':>5} {'spikes/step':>12} {'NRMSE':>8} {'skill':>8} {'dir':>6} {'instant R2':>11}")
    rows = []
    for model, W in Ws.items():
        enc, w = encoder_for(W, desc)
        for gain in args.gains:
            res = Reservoir(W, superclass, gain=gain)
            res.reset(64)
            counts, spikes = run_reservoir(res, enc, w, task["u"], desc)
            out = score_counts(counts, task, sp)
            row = dict(model=model, gain=gain, spikes_per_step=float(spikes.mean()), **out)
            rows.append(row)
            print(f"{model:>9} {gain:5.1f} {row['spikes_per_step']:12.0f} {out['nrmse']:8.4f} "
                  f"{out['skill_vs_persistence']:+8.4f} {out['directional_acc']:6.3f} "
                  f"{out['instant_r2']:11.4f}")
    (HERE / f"probe6_{args.task}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
