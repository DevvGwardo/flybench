"""Live console: watch the fly's nervous system predict a real series in real time.

Runs the real connectome, streams every spike over SSE, and plots the prediction
against the truth as it happens. The readout is fitted once on the first part of
the series; everything streamed is data the readout has never seen.

    python flybench/live.py --task btc --port 8931
    open http://127.0.0.1:8931

Endpoints
    /           the console
    /positions  real neuron coordinates (from the EM volume) + region ids
    /stream     SSE, one frame per brain step (20 ms of biological time)
    /metrics    the fitted readout's held-out score, as JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena import (DT, LAMBDAS, Reservoir, TAUS, _normalise_y, encoder_for,  # noqa: E402
                   filter_traces, fit_ridge, load_real, make_task, r2_of,
                   run_reservoir, spans, zsplit)
from fly_brain import DATA  # noqa: E402

HERE = Path(__file__).resolve().parent
REGIONS = ["ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "visual_projection",
           "vnc_sensory", "ol_sensory", "cb_sensory", "descending_neuron"]
N_POINTS = 12000
HOT_PER_FRAME = 400


def region_map(superclass: np.ndarray) -> np.ndarray:
    out = np.full(len(superclass), 255, np.uint8)
    for i, name in enumerate(REGIONS):
        out[superclass == name] = i
    return out


class Runner:
    """Owns the brain; produces one frame per step for the SSE clients."""

    def __init__(self, task_name: str):
        self._frame: dict | None = None
        self._stop = threading.Event()
        W, superclass = load_real()
        self.W, self.superclass = W, superclass
        self.desc = np.flatnonzero(superclass == "descending_neuron")
        self.enc, self.w = encoder_for(W, self.desc)
        self.task_name = task_name
        task = make_task(task_name)
        self.u, self.y = task["u"], task["y"]
        self.res = Reservoir(W, superclass)
        self.res.reset(64)
        self.regions = region_map(superclass)
        self.region_of_desc = self.regions[self.desc]
        # positions: real coordinates where the EM volume gives them
        meta = np.load(DATA / "brain.npz", allow_pickle=True)
        pos = meta["positions"]
        ok = np.isfinite(pos).all(axis=1)
        idx = np.flatnonzero(ok)
        rng = np.random.default_rng(3)
        self.pts_idx = np.sort(rng.choice(idx, min(N_POINTS, len(idx)), replace=False))
        self.pts = pos[self.pts_idx].astype(np.float32)
        self.pts = (self.pts - self.pts.mean(0)) / (np.abs(self.pts - self.pts.mean(0)).max() + 1e-6)
        self.pts_region = self.regions[self.pts_idx]
        self.slot_of_point = {int(v): k for k, v in enumerate(self.pts_idx)}
        self.frame = 0
        self.history: list[dict] = []
        self.metrics: dict = {}
        self.lock = threading.Lock()
        self.fit_readout()

    def start_clock(self, fps: float = 50.0) -> None:
        """Start the thread that owns step().

        Called explicitly rather than from __init__ so a combined console can
        run several Runners from ONE thread. Two clocks would step the same
        numba kernels concurrently, which is not safe.
        """
        self._clock = threading.Thread(target=self._run_clock, args=(fps,), daemon=True)
        self._clock.start()

    def _run_clock(self, fps: float = 50.0) -> None:
        period = 1.0 / fps
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._frame = self.step()
            except Exception as e:
                print(f"clock error: {type(e).__name__}: {e}")
                time.sleep(0.5)
                continue
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def fit_readout(self):
        """Record the training slice, tune the readout on validation, report the held-out score."""
        t0 = time.time()
        self.slot = np.full(self.res.n, -1, np.int64)
        self.slot[self.desc] = np.arange(len(self.desc))
        counts, spikes = run_reservoir(self.res, self.enc, self.w, self.u, self.desc)
        self.sp = spans(len(self.u))
        tr, va, te = self.sp
        self.ym, self.ys = self.y[tr].mean(), self.y[tr].std() + 1e-12
        yn = (self.y - self.ym) / self.ys
        best = None
        for tau in TAUS:
            X = filter_traces(counts, tau)
            mu, sd = zsplit(X, tr)
            Xn = (X - mu) / sd
            for lam in LAMBDAS:
                w = fit_ridge(Xn, yn, tr, lam)
                v = r2_of(Xn[va] @ w, yn[va])
                if best is None or v > best[0]:
                    best = (float(v), tau, float(lam), w, mu, sd, X)
        self.val_r2, self.tau, self.lam, self.wout, self.mu, self.sd, X = best
        self.decay = np.float32(np.exp(-DT / self.tau))
        Xn = (X - self.mu) / self.sd
        pred, truth = Xn[te] @ self.wout, yn[te]
        persist = yn[te.start - 1: te.stop - 1]
        nrmse = float(np.sqrt(((truth - pred) ** 2).mean()))
        p_nrmse = float(np.sqrt(((truth - persist) ** 2).mean()))
        self.metrics = dict(task=self.task_name, steps=len(self.u), encoder_cells=int(len(self.enc)),
                            tau=self.tau, lambda_=self.lam, val_r2=round(self.val_r2, 4),
                            held_out_nrmse=round(nrmse, 4), persistence_nrmse=round(p_nrmse, 4),
                            skill_vs_persistence=round(1 - nrmse / p_nrmse, 4) if p_nrmse else None,
                            spikes_per_step=round(float(spikes.mean()), 1),
                            train_seconds=round(time.time() - t0, 1))
        print(json.dumps(self.metrics, indent=2))
        # replay the held-out tail from a fresh state, so the stream is honest
        self.res.reset(64)
        self.stream_start = te.start
        self.trace = np.zeros(len(self.desc), np.float32)
        for t in range(self.stream_start):
            self.res.v[self.enc] += np.float32(self.u[t] * 0.4) * self.w
            fired = self.res.step()
            self.trace *= self.decay
            s = self.slot[fired]
            self.trace[s[s >= 0]] += 1.0
        self.preds: list[float] = []
        self.truths: list[float] = []

    def step(self) -> dict:
        """One brain step: advance, predict, emit."""
        t = self.stream_start + self.frame
        if t >= len(self.u):
            self.frame = 0
            t = self.stream_start
            self.preds, self.truths = [], []
        self.res.v[self.enc] += np.float32(self.u[t] * 0.4) * self.w
        fired = self.res.step()
        self.trace *= self.decay
        s = self.slot[fired]
        self.trace[s[s >= 0]] += 1.0
        pred = float((self.trace - self.mu) / self.sd @ self.wout)
        truth = float((self.y[t] - self.ym) / self.ys)
        self.preds.append(pred)
        self.truths.append(truth)
        self.frame += 1
        hot = fired if len(fired) <= HOT_PER_FRAME else np.random.choice(fired, HOT_PER_FRAME, replace=False)
        hot_slots = [self.slot_of_point[int(i)] for i in hot if int(i) in self.slot_of_point]
        counts = np.bincount(self.regions[fired], minlength=len(REGIONS) + 1)[: len(REGIONS)]
        return dict(t=t, u=round(float(self.u[t]), 4), y=round(truth, 4), pred=round(pred, 4),
                    hot=hot_slots, regions=[int(c) for c in counts], series=self.task_name,
                    fired=int(len(fired)), score=round(self.running_r2(), 4))

    def running_r2(self) -> float:
        if len(self.preds) < 30:
            return 0.0
        p, y = np.array(self.preds), np.array(self.truths)
        return float(1 - ((y - p) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>flybrain live</title>
<style>
:root{--bg:#0a0b09;--fg:#c8f542;--dim:#3a4423;--hot:#ff5df2}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
header{display:flex;gap:18px;align-items:baseline;padding:10px 14px;border-bottom:1px solid var(--dim)}
h1{font-size:14px;margin:0;letter-spacing:.14em;text-transform:uppercase}
.big{font-size:22px;font-variant-numeric:tabular-nums}
.wrap{display:grid;grid-template-columns:minmax(360px,1fr) minmax(420px,1.1fr);gap:14px;padding:14px}
.panel{border:1px solid var(--dim);padding:10px}
.sub{opacity:.6;font-size:11px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}
canvas{display:block;width:100%;background:#000}
.bars{display:grid;gap:3px;margin-top:8px}
.bar{display:grid;grid-template-columns:150px 1fr 52px;gap:8px;align-items:center;font-size:11px}
.bar i{display:block;height:7px;background:var(--fg)}
.bar span:last-child{text-align:right;opacity:.7}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;opacity:.75;margin-top:6px}
.legend b{display:inline-block;width:8px;height:8px;margin-right:4px}
</style>
<header>
  <h1>flybrain live</h1>
  <div><span class="sub">live score</span> <span class="big" id="score">--</span></div>
  <div><span class="sub">spikes/step</span> <span class="big" id="fired">--</span></div>
  <div><span class="sub">task</span> <span class="big" id="task">--</span></div>
</header>
<div class="wrap">
  <div class="panel">
    <div class="sub">real series vs what the brain predicts (held-out)</div>
    <canvas id="sig" height="240"></canvas>
    <div class="legend"><span><b style="background:#4a5a2a"></b>truth</span>
      <span><b style="background:#ff5df2"></b>prediction</span>
      <span id="hz"></span></div>
    <div class="sub" style="margin-top:12px">region firing, spikes this step</div>
    <div class="bars" id="bars"></div>
  </div>
  <div class="panel">
    <div class="sub">all 166,700 neurons &middot; real coordinates from the EM volume</div>
    <canvas id="brain" height="420"></canvas>
    <div class="legend" id="rlegend"></div>
    <div class="sub" style="margin-top:10px">the 1,314 output cables, spike traces</div>
    <canvas id="desc" height="90"></canvas>
  </div>
</div>
<script>
const REGIONS=["ol_intrinsic","cb_intrinsic","vnc_intrinsic","visual_projection","vnc_sensory","ol_sensory","cb_sensory","descending"];
const RC=["#2f3a1e","#3f4f26","#55682f","#7a9430","#a07a2e","#c8f542","#8fd0ff","#ff5df2"];
let P=[],PR=[],slots=new Map(),hot=new Set(),frame=null;
const sig=document.getElementById('sig'),bx=sig.getContext('2d');
const br=document.getElementById('brain'),bc=br.getContext('2d');
const dc=document.getElementById('desc'),dx=dc.getContext('2d');
let ang=0,hist=[],preds=[],truths=[];
function size(c){const r=c.getBoundingClientRect();c.width=r.width*devicePixelRatio;c.height=c.height*devicePixelRatio;}
function resize(){[sig,br,dc].forEach(size);}
addEventListener('resize',resize);
fetch('/positions').then(r=>r.json()).then(d=>{
  P=d.pts;PR=d.region;P.forEach((_,i)=>slots.set(i,i));
  document.getElementById('rlegend').innerHTML=REGIONS.map((n,i)=>
    `<span><b style="background:${RC[i]}"></b>${n}</span>`).join('');
  resize();
});
const es=new EventSource('/stream');
es.onmessage=e=>{
  frame=JSON.parse(e.data);
  hot=new Set(frame.hot);
  hist.push(frame.y);preds.push(frame.pred);truths.push(frame.y);
  if(hist.length>420){hist.shift();preds.shift();truths.shift();}
  document.getElementById('score').textContent=(frame.score*100).toFixed(1)+'%';
  document.getElementById('fired').textContent=frame.fired;
  document.getElementById('bars').innerHTML=REGIONS.map((n,i)=>{
    const v=frame.regions[i]||0,mx=Math.max(1,...frame.regions);
    return `<div class="bar"><span>${n}</span><i style="width:${100*v/mx}%"></i><span>${v}</span></div>`;}).join('');
};
function drawSig(){
  const w=sig.width,h=sig.height;bx.clearRect(0,0,w,h);
  if(!hist.length)return;
  const lo=-3,hi=3,Y=v=>h-(v-lo)/(hi-lo)*h;
  bx.strokeStyle='#3a4423';bx.lineWidth=1;
  for(let g=-3;g<=3;g++){bx.beginPath();bx.moveTo(0,Y(g));bx.lineTo(w,Y(g));bx.stroke();}
  bx.lineWidth=1.6*devicePixelRatio;
  bx.strokeStyle='#4a5a2a';bx.beginPath();truths.forEach((v,i)=>{const x=i/w*hist.length*w; i?bx.lineTo(x,Y(v)):bx.moveTo(x,Y(v));});bx.stroke();
  bx.strokeStyle='#ff5df2';bx.beginPath();preds.forEach((v,i)=>{const x=i/w*hist.length*w; i?bx.lineTo(x,Y(v)):bx.moveTo(x,Y(v));});bx.stroke();
}
function drawBrain(){
  const w=br.width,h=br.height;bc.clearRect(0,0,w,h);
  if(!P.length)return;
  ang+=0.004;const c=Math.cos(ang),s=Math.sin(ang),sc=Math.min(w,h)*0.36;
  for(let i=0;i<P.length;i++){
    const p=P[i],X=p[0]*c-p[2]*s,Z=p[0]*s+p[2]*c,Y=p[1];
    const px=w/2+X*sc/(2.2+Z),py=h/2+Y*sc/(2.2+Z);
    if(hot.has(i)){bc.fillStyle='#ff5df2';bc.fillRect(px-1.6,py-1.6,3.2,3.2);}
    else{bc.fillStyle=RC[PR[i]]||'#2a3216';bc.fillRect(px,py,1.5,1.5);}
  }
}
function drawDesc(){
  const w=dc.width,h=dc.height;dx.clearRect(0,0,w,h);
  if(!frame)return;
  dx.fillStyle='#c8f542';
  const n=frame.regions[7]||0,mx=Math.max(1,n);
  for(let i=0;i<48;i++){const v=(i*37%n)/mx;dx.fillRect(i*(w/48),h-h*v, w/48-1, h*v);}
}
function loop(){drawSig();drawBrain();drawDesc();requestAnimationFrame(loop);}
loop();
</script>"""


class Handler(BaseHTTPRequestHandler):
    runner: Runner = None

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/positions"):
            r = self.runner
            self._send(json.dumps({"pts": r.pts.tolist(), "region": r.pts_region.tolist()}).encode(),
                       "application/json")
        elif self.path.startswith("/metrics"):
            self._send(json.dumps(self.runner.metrics).encode(), "application/json")
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    while self.runner._frame is None:
                        time.sleep(0.01)
                    frame = self.runner._frame
                    self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1 / 50.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="btc")
    ap.add_argument("--port", type=int, default=8931)
    args = ap.parse_args()
    print(f"fitting readout on {args.task} ...")
    Handler.runner = Runner(args.task)
    Handler.runner.start_clock(50.0)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"flybrain live on http://127.0.0.1:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
