"""FlyArena: the frozen fly connectome as a computer, scored against controls.

    Is a real fruit-fly nervous system, used as a fixed reservoir, actually good
    at anything? Better than a random network with the same size and the same
    synapse statistics? Better than feeding the input straight to a linear model?

How a task gets in and out. The input is a scalar stream u(t). It is injected as
current into the cells that synapse DIRECTLY onto the descending neurons - the
last relay before the brain's output cables - with fixed random +/-1 weights
(the encoder, W_in). Everything between the encoder and the 1,314 descending
neurons is the connectome. Only a linear readout on those 1,314 output neurons
is trained; the brain never changes.

What the probes established, and why the setup looks like this:
  * Input into a random 0.6% of the brain attenuates to nothing in two hops:
    weights are normalised so each neuron's incoming synapses sum to 1, so one
    presynaptic spike moves a neuron by ~1/153 of threshold. Driving the cells
    that feed the output neurons is what gets a signal through.
  * The readout penalty must be validated over a wide range. At lambda=1e-2 the
    same recorded activity scores R^2 -2; at lambda=1 it scores 0.79.
  * A constant background drive (the repo's eye_drive) does NOT help a scalar
    reservoir - it only adds noise. Dropped.
  * Raw spike counts are recorded and the readout's time constant is applied
    offline, so tau is tuned on validation like any other hyperparameter.

Models raced (all n = 166,700, all 25,582,938 synapses, same LIF dynamics):
    fly        MaleCNS v1.0 connectome, weights as electron microscopy found them
    random     same weight magnitudes and count, placed at random
    shuffled   the real synapses, retargeted by a random permutation
               (keeps every neuron's in-degree exactly; destroys topology)
    linear     no reservoir at all - ridge on the raw input window

Tasks, all scored on a held-out tail, never shuffled:
    memory     memory capacity: how many past inputs survive at the output
    mackey     Mackey-Glass, tau = 17 time units, subsampled to the standard rate
    narma      NARMA-10
    sunspots   real monthly sunspot counts, 1749-2026 (SILSO)
    btc        real BTC/USD daily closes, 10 years (Yahoo)
    brent      real Brent crude daily, 1987-2026 (FRED)

    python flybench/arena.py run --models fly random shuffled linear
    python flybench/arena.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fly_brain import DATA, _propagate  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = DATA / "controls"
RESULTS = HERE / "results.json"

DT = 0.02
T_STEPS = 7000                     # for synthetic tasks; real series use their own length
FRACTIONS = (0.07, 0.57, 0.14, 0.22)   # washout, train, validate, test
DELAYS = 20
K_ENC = 30000
SCALE = 0.4
TONIC, GAIN = 0.14, 3.0
LAMBDAS = np.logspace(-2, 4, 7)
TAUS = (0.05, 0.1, 0.2, 0.5)       # readout time constant, seconds


def spans(n: int):
    """Temporal splits: washout, then train / validate / test on the tail."""
    w, tr, va = (int(n * f) for f in FRACTIONS[:3])
    return slice(w, w + tr), slice(w + tr, w + tr + va), slice(w + tr + va, n)


# ---------------------------------------------------------------- reservoirs

class Reservoir:
    """LIF network over an arbitrary sparse weight matrix.

    Same dynamics as fly_brain.FlyBrain (dt, tau, noise, threshold); only the
    weights differ, so any difference in score is a difference in wiring.
    """

    def __init__(self, W: sparse.csc_matrix, superclass: np.ndarray,
                 tonic: float = TONIC, gain: float = GAIN, seed: int = 64):
        W = W.tocsc()
        self.n = W.shape[0]
        self.indptr, self.indices = W.indptr, W.indices
        self.weights = W.data.astype(np.float32)
        self.superclass = superclass
        self.dt, self.tau = DT, 0.100
        self.tonic, self.gain = tonic, gain
        self.noise_hz, self.noise_amp = 1.2, 0.22
        self.decay = np.float32(np.exp(-self.dt / self.tau))
        self.reset(seed)

    def reset(self, seed: int = 64) -> None:
        self.rng = np.random.default_rng(seed)
        self.v = np.zeros(self.n, np.float32)
        self.fired = np.empty(0, np.int64)

    def step(self) -> np.ndarray:
        current = _propagate(self.indptr, self.indices, self.weights, self.fired, self.n) * self.gain
        self.v *= self.decay
        self.v += current + self.tonic
        self.v += (self.rng.random(self.n) < self.noise_hz * self.dt) * np.float32(self.noise_amp)
        fired = np.flatnonzero(self.v >= 1.0)
        self.v[fired] = 0.0
        self.fired = fired
        return fired


def encoder_for(W: sparse.csc_matrix, desc: np.ndarray, k: int = K_ENC, seed: int = 7):
    """The cells that synapse directly onto the output neurons, with fixed random +/-1 weights.

    Computed from each network's OWN wiring, so every model is driven the same
    way: 'the last relay before your output cables'.
    """
    pre = np.unique(W.tocsc()[:, desc].indices)
    if len(pre) > k:
        pre = np.random.default_rng(seed).choice(pre, k, replace=False)
    enc = np.asarray(pre, np.int64)
    w = np.random.default_rng(seed).choice(np.array([-1.0, 1.0], np.float32), size=len(enc))
    return enc, w


def run_reservoir(res: Reservoir, enc, w, u: np.ndarray, desc: np.ndarray, scale: float = SCALE):
    """Drive the network with u. Returns raw per-step spike counts of the 1,314
    descending neurons (so the readout's time constant can be applied offline)
    and the total spikes per step across the whole brain."""
    slot = np.full(res.n, -1, np.int64)
    slot[desc] = np.arange(len(desc))
    counts = np.zeros((len(u), len(desc)), np.float32)
    spikes = np.empty(len(u), np.int32)
    for t, ut in enumerate(u):
        res.v[enc] += np.float32(ut * scale) * w
        fired = res.step()
        spikes[t] = len(fired)
        s = slot[fired]
        s = s[s >= 0]
        if len(s):
            np.add.at(counts[t], s, 1.0)
    return counts, spikes


def filter_traces(counts: np.ndarray, tau: float) -> np.ndarray:
    """Exponential spike trace: X[t] = decay * X[t-1] + counts[t]."""
    decay = np.float32(np.exp(-DT / tau))
    X = np.empty_like(counts)
    acc = np.zeros(counts.shape[1], np.float32)
    for t in range(len(counts)):
        acc = acc * decay + counts[t]
        X[t] = acc
    return X


# ---------------------------------------------------------------- controls

def load_real():
    W = sparse.load_npz(DATA / "weights.npz").tocsr()
    meta = np.load(DATA / "brain.npz", allow_pickle=True)
    return W, meta["superclass"].astype(str)


def _normalise(W: sparse.coo_matrix) -> sparse.csr_matrix:
    """Same recipe as build_brain.py: every neuron's incoming weights sum to 1."""
    W = W.tocsr().astype(np.float32)
    incoming = np.asarray(abs(W).sum(axis=0)).ravel()
    coo = W.tocoo()
    coo.data = coo.data / np.maximum(incoming[coo.col], 1.0)
    return coo.tocsr().astype(np.float32)


def build_control(kind: str, seed: int = 11) -> sparse.csr_matrix:
    path = CACHE / f"{kind}_{seed}.npz"
    if path.exists():
        return sparse.load_npz(path).tocsr()
    W, _ = load_real()
    W = W.tocoo()
    rng = np.random.default_rng(seed)
    n, nnz = W.shape[0], W.nnz
    if kind == "random":
        post = rng.integers(0, n, nnz, dtype=np.int32)
        pre = rng.integers(0, n, nnz, dtype=np.int32)
        signs = np.where(rng.random(nnz) < 0.35, -1.0, 1.0)
        data = np.abs(W.data) * signs
    elif kind == "shuffled":
        post = rng.permutation(n)[W.row].astype(np.int32)
        pre = W.col.astype(np.int32)
        data = W.data
    else:
        raise ValueError(kind)
    out = _normalise(sparse.coo_matrix((data.astype(np.float32), (post, pre)), shape=(n, n)))
    CACHE.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, out)
    return out


# ---------------------------------------------------------------- readout

def zsplit(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    pos = sd[sd > 0]
    return mu, np.maximum(sd, 0.05 * pos.mean() if len(pos) else 1.0)


def fit_ridge(Xn, y, tr, lam):
    A = Xn[tr].T @ Xn[tr] + lam * len(Xn[tr]) * np.eye(Xn.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xn[tr].T @ y[tr])


def r2_of(pred, y):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def _normalise_y(y, tr):
    return (y - y[tr].mean()) / (y[tr].std() + 1e-12)


def score_counts(counts, task, sp):
    """Tune (tau, lambda) on validation, then report the held-out test score.

    For memory, validation score is the sum of val R^2 over all delays, matching
    how the test score is computed.
    """
    tr, va, te = sp
    u, y, kind = task["u"], task["y"], task["kind"]
    best = None
    for tau in TAUS:
        X = filter_traces(counts, tau)
        mu, sd = zsplit(X, tr)
        Xn = (X - mu) / sd
        if kind == "memory":
            val = 0.0
            for k in (1, 2, 4, 8, 16):      # coarse delay set for tuning tau only
                yk = np.roll(u, k)
                scored = [(r2_of(Xn[va] @ fit_ridge(Xn, yk, tr, lam), yk[va]), lam)
                          for lam in (0.1, 1.0, 10.0)]
                val += max(0.0, max(scored, key=lambda c: c[0])[0])
            cand = (val, tau, None)
        else:
            yn = _normalise_y(y, tr)
            cand = max(((r2_of(Xn[va] @ fit_ridge(Xn, yn, tr, lam), yn[va]), tau, lam)
                        for lam in LAMBDAS), key=lambda c: c[0])
        if best is None or cand[0] > best[0]:
            best = cand
    _, tau, lam = best
    X = filter_traces(counts, tau)
    mu, sd = zsplit(X, tr)
    Xn = (X - mu) / sd
    out = dict(tau=tau)
    if kind == "memory":
        total, per = 0.0, []
        for k in range(1, DELAYS + 1):
            yk = np.roll(u, k)
            bl = max(((r2_of(Xn[va] @ fit_ridge(Xn, yk, tr, lm), yk[va]), lm) for lm in LAMBDAS),
                     key=lambda c: c[0])
            r = r2_of(Xn[te] @ fit_ridge(Xn, yk, tr, bl[1]), yk[te])
            per.append(round(float(r), 3))
            total += max(0.0, float(r))
        out.update(memory_capacity=round(total, 3), per_delay=per)
    else:
        yn = _normalise_y(y, tr)
        bl = max(((r2_of(Xn[va] @ fit_ridge(Xn, yn, tr, lm), yn[va]), lm) for lm in LAMBDAS),
                 key=lambda c: c[0])
        w = fit_ridge(Xn, yn, tr, bl[1])
        pred, truth = Xn[te] @ w, yn[te]
        persist = yn[te.start - 1: te.stop - 1]
        nrmse = float(np.sqrt(((truth - pred) ** 2).mean()))
        p_nrmse = float(np.sqrt(((truth - persist) ** 2).mean()))
        out.update(nrmse=round(nrmse, 4), persistence_nrmse=round(p_nrmse, 4),
                   skill_vs_persistence=round(1 - nrmse / p_nrmse, 4) if p_nrmse else None,
                   directional_acc=round(float((np.sign(np.diff(pred)) == np.sign(np.diff(truth))).mean()), 4),
                   lambda_=float(bl[1]), val_r2=round(float(bl[0]), 4))
    # how much of the input is readable at the output right now, same readout
    yn = _normalise_y(u, tr)
    bl = max(((r2_of(Xn[va] @ fit_ridge(Xn, yn, tr, lm), yn[va]), lm) for lm in LAMBDAS),
             key=lambda c: c[0])
    out["instant_r2"] = round(r2_of(Xn[te] @ fit_ridge(Xn, yn, tr, bl[1]), yn[te]), 4)
    out["instant_val_r2"] = round(float(bl[0]), 4)
    return out


def score_linear(task, sp, win: int = 20):
    """No reservoir: ridge on the raw last-`win` inputs. The floor to beat."""
    tr, va, te = sp
    u, y, kind = task["u"], task["y"], task["kind"]
    X = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.zeros(win, np.float32), u]), win)[: len(u)]
    mu, sd = zsplit(X, tr)
    Xn = (X - mu) / sd
    out: dict = {}
    if kind == "memory":
        total, per = 0.0, []
        for k in range(1, DELAYS + 1):
            yk = np.roll(u, k)
            bl = max(((r2_of(Xn[va] @ fit_ridge(Xn, yk, tr, lm), yk[va]), lm) for lm in LAMBDAS),
                     key=lambda c: c[0])
            r = r2_of(Xn[te] @ fit_ridge(Xn, yk, tr, bl[1]), yk[te])
            per.append(round(float(r), 3))
            total += max(0.0, float(r))
        out.update(memory_capacity=round(total, 3), per_delay=per)
    else:
        yn = _normalise_y(y, tr)
        bl = max(((r2_of(Xn[va] @ fit_ridge(Xn, yn, tr, lm), yn[va]), lm) for lm in LAMBDAS),
                 key=lambda c: c[0])
        pred, truth = Xn[te] @ fit_ridge(Xn, yn, tr, bl[1]), yn[te]
        persist = yn[te.start - 1: te.stop - 1]
        nrmse = float(np.sqrt(((truth - pred) ** 2).mean()))
        p_nrmse = float(np.sqrt(((truth - persist) ** 2).mean()))
        out.update(nrmse=round(nrmse, 4), persistence_nrmse=round(p_nrmse, 4),
                   skill_vs_persistence=round(1 - nrmse / p_nrmse, 4) if p_nrmse else None,
                   directional_acc=round(float((np.sign(np.diff(pred)) == np.sign(np.diff(truth))).mean()), 4))
    yn = _normalise_y(u, tr)
    bl = max(((r2_of(Xn[va] @ fit_ridge(Xn, yn, tr, lm), yn[va]), lm) for lm in LAMBDAS),
             key=lambda c: c[0])
    out["instant_r2"] = round(r2_of(Xn[te] @ fit_ridge(Xn, yn, tr, bl[1]), yn[te]), 4)
    return out


# ---------------------------------------------------------------- tasks

def mackey_glass(n: int, tau_steps: int = 170, subsample: int = 10) -> np.ndarray:
    """Classic Mackey-Glass: tau = 17 time units, integrated at dt = 0.1, so the
    delay is 170 steps. Subsampled by 10 to the standard rate - at the full
    integration rate consecutive samples are near-identical and persistence is
    a near-perfect predictor, which makes the task meaningless.

    A delay of 17 *steps* is only 1.7 time units; the fixed point at x = 1 is
    stable there and the series flatlines. make_task refuses to score a constant.
    """
    x = np.full(n * subsample + 5000, 1.2)
    for t in range(tau_steps, len(x) - 1):
        x[t + 1] = x[t] + 0.1 * (0.2 * x[t - tau_steps] / (1 + x[t - tau_steps] ** 10)
                                 - 0.1 * x[t])
    return x[5000::subsample]


def narma10(n: int, seed: int = 5):
    rng = np.random.default_rng(seed)
    u = rng.random(n) * 0.5
    y = np.zeros(n)
    for t in range(10, n):
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * y[t - 10: t].sum() + 1.5 * u[t - 10] * u[t] + 0.1
    return u, y


def load_series(name: str) -> np.ndarray:
    d = HERE / "data"
    if name == "sunspots":
        raw = np.genfromtxt(d / "sn_monthly.csv", delimiter=";", usecols=(3,))
    elif name == "btc":
        r = json.loads((d / "yahoo_btc.json").read_text())["chart"]["result"][0]
        raw = np.array([c for c in r["indicators"]["quote"][0]["close"] if c is not None])
    elif name == "brent":
        raw = np.genfromtxt(d / "fred_DCOILBRENTEU.csv", delimiter=",", skip_header=1, usecols=(1,))
        raw = raw[np.isfinite(raw)]
    else:
        raise ValueError(name)
    return np.asarray(raw, float)


def make_task(name: str) -> dict:
    """Return u (drive), y (target) and the kind of score to report.

    Series tasks are centred on the training slice only, so no test information
    leaks into the drive. A DC offset would otherwise just park the network
    against its firing threshold and carry no information.
    """
    if name == "memory":
        u = (np.random.default_rng(1234).random(T_STEPS).astype(np.float32) - 0.5) * 2.0
        return dict(u=u, y=None, kind="memory")
    if name == "mackey":
        s = mackey_glass(T_STEPS + 1)
        u, y = s[:-1], s[1:]
    elif name == "narma":
        u, y = narma10(T_STEPS + 1)
        u, y = u[:-1], y[1:]
    else:
        s = load_series(name)
        if len(s) > T_STEPS + 1:
            s = s[-(T_STEPS + 1):]
        u, y = s[:-1], s[1:]
    tr = spans(len(u))[0]
    mu, sd = u[tr].mean(), u[tr].std() + 1e-12
    if sd < 0.05:
        raise SystemExit(f"task {name!r} is (near) constant - refusing to score it")
    return dict(u=((u - mu) / sd).astype(np.float32), y=(y - mu) / sd, kind="series")


# ---------------------------------------------------------------- driver

def evaluate(model: str, task_name: str, W, superclass) -> dict:
    task = make_task(task_name)
    sp = spans(len(task["u"]))
    t0 = time.time()
    base = dict(model=model, task=task_name, steps=len(task["u"]))
    if model == "linear":
        return dict(base, **score_linear(task, sp), wall_s=round(time.time() - t0, 1))
    desc = np.flatnonzero(superclass == "descending_neuron")
    enc, w = encoder_for(W, desc)
    res = Reservoir(W, superclass)
    res.reset(64)
    counts, spikes = run_reservoir(res, enc, w, task["u"], desc)
    return dict(base, **score_counts(counts, task, sp), encoder_n=int(len(enc)),
                spikes_per_step=round(float(spikes.mean()), 1),
                wall_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--models", nargs="+", default=["fly", "random", "shuffled", "linear"])
    ap.add_argument("--tasks", nargs="+",
                    default=["memory", "mackey", "narma", "sunspots", "btc", "brent"])
    args = ap.parse_args()

    if args.cmd == "report":
        rows = json.loads(RESULTS.read_text())
        for task in dict.fromkeys(r["task"] for r in rows):
            print(f"\n=== {task} ({rows[0].get('steps', '?')} steps) ===")
            for r in [x for x in rows if x["task"] == task]:
                inst = r.get("instant_r2", float("nan"))
                if "memory_capacity" in r:
                    print(f"  {r['model']:>9}  MC {r['memory_capacity']:6.2f}   "
                          f"instant R2 {inst:6.3f}   {r['wall_s']:5.0f}s")
                else:
                    print(f"  {r['model']:>9}  NRMSE {r['nrmse']:.4f}  persistence {r['persistence_nrmse']:.4f}  "
                          f"skill {r['skill_vs_persistence']:+.4f}  dir {r['directional_acc']:.3f}  "
                          f"instant R2 {inst:6.3f}")
        return

    W_real, superclass = load_real()
    print(f"connectome {W_real.shape[0]:,} neurons, {W_real.nnz:,} synapses")
    Ws = {"fly": W_real, "random": build_control("random"), "shuffled": build_control("shuffled")}
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {(r["model"], r["task"]) for r in results}
    for model in args.models:
        for task in args.tasks:
            if (model, task) in done:
                continue
            r = evaluate(model, task, Ws.get(model), superclass)
            results.append(r)
            RESULTS.write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in r.items() if k != "per_delay"}))
    print(f"\n{len(results)} runs -> {RESULTS}")


if __name__ == "__main__":
    main()
