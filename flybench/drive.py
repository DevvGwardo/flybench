"""FlyDrive: put the frozen fly connectome behind the wheel, sober vs drugged.

The question the earlier flybench run left open: a fly brain beats a random
network at *predicting* a series, but can it actually *control* something in a
closed loop? Control is the real-world test - the output changes the input.

How it works
------------
  * A point-mass car drives a winding road. State: lateral offset x and heading
    theta. The fly's job is to keep the car in its lane.
  * The car's errors are encoded into the cells that synapse DIRECTLY onto the
    fly's own steering and walking motor neurons (steer_L/R = DNa02,
    forward_L/R = DNg100). Three channels: lateral error, heading error, and a
    curvature lookahead.
  * A linear decoder reads the 1,314 descending neurons - the brain's output
    cables - and turns them into a yaw command.
  * The decoder is fitted ONCE, on sober runs, to imitate a competent PD driver
    (behaviour cloning). It is then FROZEN. Only the brain's global parameters
    change between conditions.

That design is the whole point: it is the same driver. Same decoder, same
weights, same wiring. The only thing that changes is the brain's chemistry.

Honest limits
-------------
  * fly.ai has NO neuromodulators. "Drugs" here are global parameter presets -
    excitability, synaptic gain, noise, firing threshold. They are not
    pharmacology. They are labelled by direction (stimulant / depressant / ...)
    because that is what the parameters do, and the table says so.
  * The decoder is trained; the fly is not. This measures how well a fixed
    connectome can be *read*, not what the fly has learned.
  * A gain-matched control run is included. Under a drug the decoder's output
    grows or shrinks along with the activity, so a score change might just be a
    mis-scaled decoder. The control divides the features by the drug's
    command-scale change (measured on the identical road and start), so the
    decoder's output distribution is the same as sober and only the information
    content of the activity can differ. Matching whole-brain spike rate instead
    is wrong: it halves the command amplitude, the car stops steering, and a
    mis-scaled decoder reads as a dramatic drug effect.

The decoder's overall gain is tuned on a held-out sober road, scored on the
closed loop rather than on open-loop R2. A regression fitted on teacher data
shrinks toward the mean once the car leaves the band it was trained on, so the
raw decoder under-steers. Left untuned, that under-steer dominates every
comparison: any drug that adds activity would "improve driving" purely by
propping up a mis-scaled decoder.

    python flybench/drive.py --steps 2400 --seeds 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_brain import _propagate  # noqa: E402
from arena import Reservoir, load_real, DT  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "drive_results.json"

# ------------------------------------------------------------------- car + road

V = 20.0                 # m/s
LANE = 1.5               # lane half-width, m
OFF_ROAD = 3.0           # |e| beyond this for CRASH_STEPS is a crash
CRASH_STEPS = 60
KP, KD = 1.5, 0.7        # the PD driver the fly is cloned from
OMEGA_MAX = 1.0          # rad/s
E_SCALE, H_SCALE, C_SCALE = 2.0, 0.5, 3.0
LOOKAHEAD = 60.0         # m


class Road:
    """Winding road: centreline is a sinusoid in distance travelled.

    amp 4 m over a 90 m wavelength at 20 m/s needs ~0.39 rad/s of yaw to track,
    against a 1.0 rad/s limit. Gentle enough to be drivable, sharp enough that a
    controller which has lost the plot ends up in the dirt.
    """

    def __init__(self, amp: float = 4.0, wave: float = 90.0, phase: float = 0.0):
        self.amp, self.wave, self.phase = amp, wave, phase
        self.k = 2 * np.pi / wave

    def centre(self, s):
        return self.amp * np.sin(self.k * (s + self.phase))

    def heading(self, s):
        return self.amp * self.k * np.cos(self.k * (s + self.phase))


class Car:
    """A point-mass car on a road. `offset` is the lateral error at t=0.

    The car always starts aligned with the road (heading error zero, lateral
    error exactly `offset`). Starting from a raw x would fold the road's own
    centreline value at s=0 - up to a full 4 m of amplitude - into the initial
    error, so a "phase" sweep would silently become a difficulty sweep.
    """

    def __init__(self, road: Road, offset: float = 0.0):
        self.road, self.s = road, 0.0
        self.x = road.centre(0.0) + offset
        self.theta = road.heading(0.0)

    def errors(self):
        e = self.x - self.road.centre(self.s)
        h = self.theta - self.road.heading(self.s)
        return e, h

    def raw_channels(self):
        """What the fly gets to sense: offset, heading error, curve ahead."""
        e, h = self.errors()
        curve = self.road.centre(self.s + LOOKAHEAD) - self.road.centre(self.s)
        return np.array([e / E_SCALE, h / H_SCALE, curve / C_SCALE], np.float32)

    def channels(self, mu=None, sd=None):
        u = self.raw_channels()
        if mu is not None:
            u = (u - mu) / sd
        return np.clip(u, -2.5, 2.5)

    def drive(self, omega: float):
        self.theta = float(np.clip(self.theta + omega * DT, -0.6, 0.6))
        self.s += V * DT
        self.x += V * self.theta * DT


def teacher(car: Car) -> float:
    e, h = car.errors()
    return float(np.clip(-KP * e - KD * h, -OMEGA_MAX, OMEGA_MAX))


def calibrate_channels(steps: int = 1200, seed: int = 4242):
    """Fixed per-channel normalisation from one teacher-driven run.

    The three channels have wildly different natural scales (metres of offset,
    radians of heading, metres of upcoming curve). Injected through a shared
    +/-1 encoder, unscaled channels would mean the curve channel alone decides
    whether the brain fires. Normalised once, on a teacher run, then frozen.
    """
    road = Road(phase=float(np.random.default_rng(seed).uniform(0, 400)))
    car = Car(road, offset=float(np.random.default_rng(seed + 1).uniform(-1.0, 1.0)))
    U = []
    for _ in range(steps):
        car.drive(teacher(car))
        U.append(car.raw_channels())
    U = np.array(U)
    mu, sd = U.mean(0), np.maximum(U.std(0), 1e-3)
    return mu.astype(np.float32), sd.astype(np.float32)


# ------------------------------------------------------------------- the brain

class DriveBrain(Reservoir):
    """Reservoir with a settable firing threshold, so a preset can move it."""

    def __init__(self, *a, thr: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.thr = float(thr)

    def step(self) -> np.ndarray:
        current = _propagate(self.indptr, self.indices, self.weights, self.fired, self.n) * self.gain
        self.v *= self.decay
        self.v += current + self.tonic
        self.v += (self.rng.random(self.n) < self.noise_hz * self.dt) * np.float32(self.noise_amp)
        fired = np.flatnonzero(self.v >= self.thr)
        self.v[fired] = 0.0
        self.fired = fired
        return fired


def motor_encoder(W, meta, k: int = 2400, seed: int = 7):
    """Cells that synapse directly onto the steering + walking motor neurons.

    Same lesson as probe3: drive a random 0.6% of the brain and the signal dies
    inside two hops, because one synapse is worth ~1/153 of threshold. Drive the
    last relay before the motor command and it lands.
    """
    motors = np.concatenate([meta["group_steer_L"], meta["group_steer_R"],
                             meta["group_forward_L"], meta["group_forward_R"]]).astype(np.int64)
    pre = np.unique(W.tocsc()[:, motors].indices)
    rng = np.random.default_rng(seed)
    if len(pre) > k:
        pre = rng.choice(pre, k, replace=False)
    pre = np.asarray(pre, np.int64)
    # split the encoder evenly across the 3 sensory channels
    chan = np.arange(len(pre)) % 3
    rng.shuffle(chan)
    w = rng.choice(np.array([-1.0, 1.0], np.float32), size=len(pre))
    return pre, chan.astype(np.int64), w


def run_episode(W, meta, desc, enc, chan, w, cond, steps, seed,
                road_phase, x0, readout=None, scale=0.4, trace_tau=0.1,
                zmu=None, zsd=None, gain_match: float = 1.0, record: bool = False,
                chmu=None, chsd=None, teacher_noise: float = 0.0,
                decoder_gain: float = 1.0):
    """One run. With readout=None the PD teacher drives and we record the brain.
    With a readout the car is under the fly's control."""
    res = DriveBrain(W, meta["superclass"].astype(str),
                     tonic=cond["tonic"], gain=cond["gain"], seed=seed, thr=cond["thr"])
    res.noise_hz, res.noise_amp = cond["noise_hz"], cond["noise_amp"]
    res.reset(seed)
    nrng = np.random.default_rng(seed + 31337)

    slot = np.full(res.n, -1, np.int64)
    slot[desc] = np.arange(len(desc))
    road = Road(phase=road_phase)
    car = Car(road, offset=x0)

    counts = np.zeros((steps, len(desc)), np.float32) if record else None
    feats = np.zeros((steps, len(desc)), np.float32) if record else None
    targets = np.zeros(steps, np.float32) if record else None
    omegas = np.zeros(steps, np.float32)
    cmds = np.zeros(steps, np.float32)
    errs = np.zeros(steps, np.float32)
    spikes = np.zeros(steps, np.int32)

    acc = np.zeros(len(desc), np.float32)
    decay = np.float32(np.exp(-DT / trace_tau))
    crash_at = -1
    off = 0

    for t in range(steps):
        u = car.channels(chmu, chsd)
        res.v[enc] += np.float32(scale) * u[chan] * w
        fired = res.step()
        spikes[t] = len(fired)

        s = slot[fired]
        s = s[s >= 0]
        acc = acc * decay
        if len(s):
            np.add.at(acc, s, 1.0)
        if record:
            if len(s):
                np.add.at(counts[t], s, 1.0)
            feats[t] = acc

        if readout is None:
            clean = teacher(car)
            omega = clean
            if teacher_noise:
                omega = float(np.clip(clean + nrng.normal(0, teacher_noise),
                                      -OMEGA_MAX, OMEGA_MAX))
            if record:
                # the car is driven by the jittered command so it visits a wider
                # band of the lane, but the target we clone is the clean PD
                # command for the state actually reached. Recording the jitter
                # instead makes the target mostly unpredictable noise and caps
                # the achievable R2 for no reason.
                targets[t] = clean
        else:
            f = (acc / gain_match - zmu) / zsd
            raw = decoder_gain * float(f @ readout)
            cmds[t] = raw
            omega = float(np.clip(raw, -OMEGA_MAX, OMEGA_MAX))

        omegas[t] = omega
        car.drive(omega)
        e, _ = car.errors()
        errs[t] = e
        if abs(e) > OFF_ROAD:
            off += 1
            if off >= CRASH_STEPS and crash_at < 0:
                crash_at = t
        else:
            off = 0

    alive = steps if crash_at < 0 else crash_at
    return dict(counts=counts, feats=feats, targets=targets, omegas=omegas,
                cmds=cmds, errs=errs, spikes=spikes, crash_at=crash_at, alive=alive,
                mean_abs_err=float(np.abs(errs[:alive]).mean()),
                in_lane=float((np.abs(errs[:alive]) < LANE).mean()),
                spikes_per_step=float(spikes.mean()),
                travelled=float(alive * V * DT))


# ------------------------------------------------------------------- conditions

SOBER = dict(tonic=0.14, gain=3.0, noise_hz=1.2, noise_amp=0.22, thr=1.0)
CONDITIONS = {
    "sober":        dict(SOBER),
    # more gain, more noise: the brain fires harder and jitters
    "stimulant":    dict(SOBER, gain=4.6, noise_hz=1.9, noise_amp=0.26),
    # resting potential pushed toward threshold, gain cut: sluggish and twitchy
    "depressant":   dict(SOBER, tonic=0.185, gain=2.2, noise_hz=1.6),
    # under-aroused: barely above rest
    "sedative":     dict(SOBER, tonic=0.05, gain=1.8, noise_hz=0.8),
    # pure noise load, wiring untouched
    "psychedelic":  dict(SOBER, noise_hz=5.0, noise_amp=0.30),
    # the stabiliser: gain held, noise cut
    "stabiliser":   dict(SOBER, gain=3.0, noise_hz=0.4, noise_amp=0.12),
}


# ------------------------------------------------------------------- readout

def fit_readout(X, y, tr, lam):
    A = X[tr].T @ X[tr] + lam * len(X[tr]) * np.eye(X.shape[1], dtype=np.float32)
    return np.linalg.solve(A, X[tr].T @ y[tr]).astype(np.float32)


def r2(pred, y):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def train(W, meta, desc, enc, chan, w, steps, n_eps, chmu, chsd, seed0=100):
    """Teacher-driven sober runs; the fly watches a competent driver.

    The teacher's command is jittered so the car visits a wider band of the lane
    than a perfect driver would. Without that, behaviour cloning only ever sees
    the middle of the road and the decoder has nothing to say when the car
    starts to drift - the classic failure mode of imitation learning.
    """
    Xs, ys = [], []
    for i in range(n_eps):
        rng = np.random.default_rng(seed0 + i)
        ep = run_episode(W, meta, desc, enc, chan, w, SOBER, steps, seed0 + i,
                         road_phase=float(rng.uniform(0, 400)),
                         x0=float(np.random.default_rng(seed0 + i + 900).uniform(-1.2, 1.2)),
                         record=True, chmu=chmu, chsd=chsd, teacher_noise=0.18)
        Xs.append(ep["feats"][200:])          # drop the settling transient
        ys.append(ep["targets"][200:])
        print(f"  teacher ep {i}: mean|e| {ep['mean_abs_err']:.3f} m  "
              f"in-lane {ep['in_lane']:.2f}  {ep['spikes_per_step']:.0f} spikes/step")
    X, y = np.concatenate(Xs), np.concatenate(ys)
    tr = slice(0, int(len(X) * 0.7))
    va = slice(int(len(X) * 0.7), len(X))
    mu, sd = X[tr].mean(0), X[tr].std(0)
    pos = sd[sd > 0]
    sd = np.maximum(sd, 0.05 * pos.mean() if len(pos) else 1.0)
    Xn = (X - mu) / sd
    best = max(((r2(Xn[va] @ fit_readout(Xn, y, tr, lam), y[va]), lam)
                for lam in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)), key=lambda c: c[0])
    print(f"  readout: val R2 {best[0]:.3f} at lambda {best[1]:g}  "
          f"({X.shape[1]} features, {len(X)} samples)")
    return fit_readout(Xn, y, tr, best[1]), mu, sd, best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--train-eps", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--dump-only", action="store_true",
                    help="fit the decoder and save readout.npz, then stop")
    args = ap.parse_args()

    t0 = time.time()
    W, sc = load_real()
    meta = np.load(Path.home() / "fly-data" / "brain.npz", allow_pickle=True)
    desc = np.flatnonzero(sc == "descending_neuron")
    enc, chan, w = motor_encoder(W, meta)
    chmu, chsd = calibrate_channels()
    print(f"connectome {W.shape[0]:,} neurons / {W.nnz:,} synapses")
    print(f"motor channel: {len(enc)} cells feed steer_L/R + forward_L/R; "
          f"{len(desc)} descending neurons read out")
    print(f"channel scales (sd): offset {chsd[0]*E_SCALE:.3f} m  "
          f"heading {chsd[1]*H_SCALE:.4f} rad  curve {chsd[2]*C_SCALE:.3f} m")

    readout, mu, sd, lam, val_r2 = train(W, meta, desc, enc, chan, w,
                                         args.steps, args.train_eps, chmu, chsd)
    np.savez(HERE / "readout.npz", readout=readout, mu=mu, sd=sd,
             chmu=chmu, chsd=chsd, lam=lam, val_r2=val_r2)
    print(f"  decoder saved -> {HERE / 'readout.npz'}")
    if args.dump_only:
        print("dump-only: stopping here")
        return

    # PD teacher with no brain at all - the ceiling a perfect controller hits
    road = Road(phase=1.0)
    car = Car(road, offset=1.0)
    terr = []
    for _ in range(args.steps):
        car.drive(teacher(car))
        terr.append(car.errors()[0])
    terr = np.abs(np.array(terr))
    print(f"\nPD teacher alone: mean|e| {terr.mean():.3f} m  "
          f"in-lane {(terr < LANE).mean():.2f}  worst {terr.max():.3f} m")

    # one road + start per seed, reused by every condition, so the only thing
    # that differs between two rows is the brain's parameters.
    seeds = []
    for i in range(args.seeds):
        seeds.append((i,
                      float(np.random.default_rng(200 + i).uniform(0, 400)),
                      float(np.random.default_rng(700 + i).uniform(-1.0, 1.0))))

    def run(cond, i, phase, x0, gm=1.0, g=1.0, seed_off=300):
        return run_episode(W, meta, desc, enc, chan, w, cond, args.steps, seed_off + i,
                           phase, x0, readout=readout, trace_tau=args.tau,
                           zmu=mu, zsd=sd, gain_match=gm, chmu=chmu, chsd=chsd,
                           decoder_gain=g)

    # Tune the decoder's overall gain on a held-out sober road, scored on the
    # closed loop rather than on open-loop R2. A regression fitted on teacher
    # data shrinks toward the mean once the car is outside the band it was
    # trained on, so the raw decoder under-steers and the car drifts wide. Left
    # untuned, that under-steer dominates every comparison: any drug that adds
    # activity would "improve driving" purely by propping up a mis-scaled
    # decoder. Tuned, sober sits at its best and a drug can only move it off.
    val = [(i, float(np.random.default_rng(800 + i).uniform(0, 400)),
            float(np.random.default_rng(900 + i).uniform(-1.0, 1.0))) for i in range(3)]
    g_scores = []
    for g in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3):
        m = float(np.mean([run(SOBER, i, ph, x0, g=g, seed_off=800)["mean_abs_err"]
                           for i, ph, x0 in val]))
        g_scores.append((m, g))
    best_err, best_g = min(g_scores)
    print("decoder gain sweep (held-out sober road): "
          + "  ".join(f"x{g:g}->{m:.3f}m" for m, g in g_scores))
    print(f"  -> decoder gain x{best_g:g} (mean|e| {best_err:.3f} m)\n")
    G = best_g
    np.savez(HERE / "readout.npz", readout=readout, mu=mu, sd=sd, chmu=chmu,
             chsd=chsd, lam=lam, val_r2=val_r2, decoder_gain=G)

    # Sober command spread per seed. The gain-matched control divides the
    # features by the drug's command-scale change, so the decoder's OUTPUT
    # distribution is the same as sober and only the information content can
    # differ. Matching whole-brain spike rate instead is wrong: it halves the
    # command amplitude and the car simply stops steering, which reads as a
    # dramatic drug effect and is really a mis-scaled decoder.
    sober_sd = {i: float(run(SOBER, i, ph, x0, g=G)["cmds"].std()) for i, ph, x0 in seeds}

    rows = []
    for cond_name, cond in CONDITIONS.items():
        modes = ["frozen"] if cond_name == "sober" else ["frozen", "gain-matched"]
        ratios = {}
        for i, ph, x0 in seeds:
            ratios[i] = 1.0 if cond_name == "sober" else (
                float(run(cond, i, ph, x0, g=G)["cmds"].std()) / sober_sd[i]
                if sober_sd[i] > 0 else 1.0)
        for gm_name in modes:
            accs = [run(cond, i, ph, x0,
                        1.0 if gm_name == "frozen" else ratios[i], g=G)
                    for i, ph, x0 in seeds]
            per = [a["mean_abs_err"] for a in accs]
            lost = sum(1 for v in per if v > 1.0)
            rows.append(dict(
                condition=cond_name, mode=gm_name, seeds=len(accs),
                cmd_scale_ratio=round(float(np.mean(list(ratios.values()))), 3),
                mean_abs_err=round(float(np.mean(per)), 3),
                median_abs_err=round(float(np.median(per)), 3),
                lost_road=lost,
                in_lane=round(float(np.mean([a["in_lane"] for a in accs])), 3),
                crashes=sum(1 for a in accs if a["crash_at"] >= 0),
                spikes_per_step=round(float(np.mean([a["spikes_per_step"] for a in accs])), 0),
                travelled=round(float(np.mean([a["travelled"] for a in accs])), 0),
                per_seed=[round(v, 2) for v in per]))
            print(f"  {cond_name:>12} {gm_name:>13}: "
                  f"median|e| {rows[-1]['median_abs_err']:6.3f} m  "
                  f"mean {rows[-1]['mean_abs_err']:6.3f}  "
                  f"lost {lost:2d}/{len(accs)}  "
                  f"in-lane {rows[-1]['in_lane']:.2f}  "
                  f"{rows[-1]['spikes_per_step']:8.0f} spikes/step  "
                  f"cmd x{rows[-1]['cmd_scale_ratio']:.2f}")

    base = [r for r in rows if r["condition"] == "sober" and r["mode"] == "frozen"][0]
    print(f"\nvs sober (median|e| {base['median_abs_err']:.3f} m, "
          f"lost {base['lost_road']}/{base['seeds']})")
    for r in rows:
        if r["mode"] != "frozen" or r["condition"] == "sober":
            continue
        d = r["median_abs_err"] - base["median_abs_err"]
        print(f"  {r['condition']:>12}: median {d:+.3f} m  "
              f"lost {r['lost_road']}/{r['seeds']} vs {base['lost_road']}/{base['seeds']}  "
              f"({'worse' if d > 0.05 else 'better' if d < -0.05 else 'same'})")

    OUT.write_text(json.dumps(dict(steps=args.steps, seeds=args.seeds,
                                   readout_val_r2=val_r2, lambda_=lam,
                                   encoder_n=int(len(enc)), rows=rows), indent=2))
    print(f"\n{time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
