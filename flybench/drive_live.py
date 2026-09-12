"""FlyDrive live: two flies race the same road, one sober, one on a drug.

Same connectome, same decoder, same road, same starting position. The only
difference is the global parameter preset applied to the brain. Top panel is
sober; bottom panel is the drugged fly. Watch where they end up.

    python flybench/drive_live.py --drug stimulant --port 8932

Needs flybench/readout.npz (fit one with: python flybench/drive.py --dump-only).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from arena import load_real, DT  # noqa: E402
from drive import (DriveBrain, Road, Car, teacher, motor_encoder,  # noqa: E402
                   CONDITIONS, SOBER, calibrate_channels, V, LANE, OFF_ROAD, CRASH_STEPS)

TRACK = 1200.0        # metres
K_STEPS = 6           # physics steps per emitted frame
# Timeout fallback: a car at full speed covers TRACK in TRACK/(V*DT) = 3000
# physics steps, so 4x that guarantees the race always terminates even if a
# car stalls without crashing.
RACE_TIMEOUT_STEPS = 12000
DRUGS = [k for k in CONDITIONS if k != "sober"]


class Runner:
    """Two brains, one road. Stepped sequentially: numba's default threading
    layer is not threadsafe, so the two networks never run concurrently."""

    def __init__(self, drug: str, seed: int = 64):
        self.W, sc = load_real()
        self.meta = np.load(Path.home() / "fly-data" / "brain.npz", allow_pickle=True)
        self.desc = np.flatnonzero(sc == "descending_neuron")
        self.enc, self.chan, self.w = motor_encoder(self.W, self.meta)
        z = np.load(HERE / "readout.npz")
        self.readout, self.zmu, self.zsd = z["readout"], z["mu"], z["sd"]
        self.chmu, self.chsd = z["chmu"], z["chsd"]
        self.g = float(z["decoder_gain"]) if "decoder_gain" in z else 1.0
        self.drug = drug
        self.seed = seed
        self.lock = threading.RLock()
        self.race = 0
        self._frame: dict | None = None
        self._pending_drug: str | None = None
        self._stop = threading.Event()
        self.reset(drug, seed)

    def start_clock(self, fps: float = 30.0) -> None:
        """Start the thread that owns the simulation.

        Every SSE client reads the same latest frame. Stepping per-connection
        means two viewers advance the race twice as fast, and numba's workqueue
        layer is not thread-safe so concurrent steppers are a wedge risk.
        Called explicitly so a combined console can own one clock for all
        Runners.
        """
        self._clock = threading.Thread(target=self._run_clock, args=(fps,), daemon=True)
        self._clock.start()

    def request_reset(self, drug: str | None = None) -> None:
        """Post a reset for the clock thread to perform. Safe from any thread:
        it touches nothing but a string."""
        self._pending_drug = drug or self.drug

    def _run_clock(self, fps: float = 30.0) -> None:
        period = 1.0 / fps
        while not self._stop.is_set():
            if self._pending_drug is not None:
                drug, self._pending_drug = self._pending_drug, None
                self.reset(drug)
                continue
            t0 = time.monotonic()
            try:
                frame = self.step()
                self._frame = frame
                if frame["done"]:
                    # hold the result long enough to read, then run the next
                    # road. Otherwise the car drives off past the finish line
                    # and the console never shows a fresh comparison.
                    time.sleep(5.0)
                    self.reset(self.drug)   # takes the lock itself
                    continue
            except Exception as e:                      # keep the console alive
                print(f"clock error: {type(e).__name__}: {e}")
                time.sleep(0.5)
                continue
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def latest(self) -> dict:
        """Block until a frame exists, then return a snapshot (no stepping).

        Deliberately lock-free: attribute reads are atomic in CPython, and the
        clock thread replaces the whole dict rather than mutating it in place.
        Taking the lock here would put an HTTP thread in the same mutual
        exclusion as the brain."""
        while self._frame is None:
            time.sleep(0.01)
        return self._frame

    def reset(self, drug: str, seed: int | None = None):
        with self.lock:          # the clock thread steps concurrently
            self._reset(drug, seed)

    def _reset(self, drug: str, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self.drug = drug
        self.race += 1
        rng = np.random.default_rng(self.seed + self.race * 977)
        self.phase = float(rng.uniform(0, 400))
        self.road = Road(phase=self.phase)
        self.x0 = float(rng.uniform(-0.9, 0.9))
        self.brains, self.cars, self.acc, self.stats = {}, {}, {}, {}
        for name, cond in (("sober", SOBER), ("drugged", CONDITIONS[drug])):
            b = DriveBrain(self.W, self.meta["superclass"].astype(str),
                           tonic=cond["tonic"], gain=cond["gain"],
                           seed=self.seed + 11, thr=cond["thr"])
            b.noise_hz, b.noise_amp = cond["noise_hz"], cond["noise_amp"]
            b.reset(self.seed + 11)
            self.brains[name] = b
            self.cars[name] = Car(self.road, offset=self.x0)
            self.acc[name] = np.zeros(len(self.desc), np.float32)
            self.stats[name] = dict(errs=[], spikes=[], crash_at=-1, off=0, alive=True)
        self.slot = np.full(self.brains["sober"].n, -1, np.int64)
        self.slot[self.desc] = np.arange(len(self.desc))
        self.t = 0
        self.decay = np.float32(np.exp(-DT / 0.1))

    def _one(self, name: str) -> None:
        b, car, st = self.brains[name], self.cars[name], self.stats[name]
        if not st["alive"]:
            st["spikes"].append(0)
            return
        u = car.channels(self.chmu, self.chsd)
        b.v[self.enc] += np.float32(0.4) * u[self.chan] * self.w
        fired = b.step()
        st["spikes"].append(len(fired))
        s = self.slot[fired]
        s = s[s >= 0]
        self.acc[name] *= self.decay
        if len(s):
            np.add.at(self.acc[name], s, 1.0)
        f = (self.acc[name] - self.zmu) / self.zsd
        omega = float(np.clip(self.g * (f @ self.readout), -1.0, 1.0))
        car.drive(omega)
        e = car.errors()[0]
        st["errs"].append(e)
        if abs(e) > OFF_ROAD:
            st["off"] += 1
            if st["off"] >= CRASH_STEPS and st["crash_at"] < 0:
                st["crash_at"] = self.t
                st["alive"] = False
        else:
            st["off"] = 0

    def step(self) -> dict:
        with self.lock:
            for _ in range(K_STEPS):
                self._one("sober")
                self._one("drugged")
                self.t += 1
            cars = {}
            for name in ("sober", "drugged"):
                car, st = self.cars[name], self.stats[name]
                errs = st["errs"] or [0.0]
                spikes = st["spikes"][-50:] or [0]
                cars[name] = dict(
                    x=round(car.x, 3), theta=round(car.theta, 4),
                    e=round(errs[-1], 3),
                    dist=round(car.s, 1),
                    mean_abs_err=round(float(np.abs(errs).mean()), 3),
                    in_lane=round(float((np.abs(errs) < LANE).mean()), 3),
                    spikes=round(float(np.mean(spikes)), 0),
                    crashed=not st["alive"],
                    crash_at=st["crash_at"])
            both_dead = all(not self.stats[n]["alive"] for n in ("sober", "drugged"))
            either_finished = self.cars["sober"].s >= TRACK \
                or self.cars["drugged"].s >= TRACK
            done = both_dead or either_finished or self.t >= RACE_TIMEOUT_STEPS
            return dict(t=self.t, drug=self.drug, race=self.race, done=bool(done),
                        track=TRACK, x0=round(self.x0, 3), phase=round(self.phase, 1),
                        cars=cars)

    def road_points(self, every: float = 10.0):
        s = np.arange(0, TRACK + every, every)
        return [[round(float(v), 3), round(float(self.road.centre(v)), 3)] for v in s]


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>FlyDrive</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0b09;color:#e8e4d9;font:13px/1.4 "IBM Plex Mono",ui-monospace,Menlo,monospace}
header{display:flex;align-items:baseline;gap:18px;padding:14px 18px;border-bottom:1px solid #1e201a}
h1{font:600 15px/1 Syne,system-ui;letter-spacing:.14em;color:#c8f542}
select,button{background:#141610;color:#e8e4d9;border:1px solid #2c3024;padding:5px 9px;font:inherit;cursor:pointer}
button:hover,select:hover{border-color:#c8f542}
#score{margin-left:auto;display:flex;gap:22px;color:#8f8b7e}
#score b{color:#e8e4d9;font-weight:500}
.panel{padding:12px 18px 4px}
.head{display:flex;gap:16px;align-items:baseline;margin-bottom:6px}
.tag{font:600 11px/1 Syne,system-ui;letter-spacing:.16em;padding:3px 7px;border:1px solid #2c3024}
.sober .tag{color:#c8f542;border-color:#3f4a22}
.drugged .tag{color:#ff7a59;border-color:#4a2a22}
.m{color:#8f8b7e}.m b{color:#e8e4d9;font-weight:500}
canvas{display:block;width:100%;height:132px;background:#0d0f0b;border:1px solid #1a1c16}
.drug-name{color:#ff7a59}
footer{padding:10px 18px 22px;color:#6b675c;font-size:11px;line-height:1.6}
</style></head><body>
<header><h1>FLYDRIVE</h1>
<span class="m">drug</span><select id="drug"></select>
<button id="restart">restart race</button>
<div id="score"><span>track <b id="track">0</b> m</span><span>race <b id="race">1</b></span></div>
</header>
<div class="panel sober"><div class="head"><span class="tag">SOBER</span>
<div class="m">offset <b id="s_e">0</b> m</div><div class="m">mean|e| <b id="s_mae">0</b></div>
<div class="m">in-lane <b id="s_lane">0</b>%</div><div class="m">dist <b id="s_d">0</b> m</div>
<div class="m">brain <b id="s_sp">0</b> Hz</div></div>
<canvas id="cs"></canvas></div>
<div class="panel drugged"><div class="head"><span class="tag">ON <span class="drug-name" id="dname">-</span></span>
<div class="m">offset <b id="d_e">0</b> m</div><div class="m">mean|e| <b id="d_mae">0</b></div>
<div class="m">in-lane <b id="d_lane">0</b>%</div><div class="m">dist <b id="d_d">0</b> m</div>
<div class="m">brain <b id="d_sp">0</b> Hz</div></div>
<canvas id="cd"></canvas></div>
<footer id="foot">same connectome. same decoder. same road. only the brain's global parameters differ.</footer>
<script>
const $=id=>document.getElementById(id);
const K=6;                       // physics steps per emitted frame
const LANE=__LANE__;             // lane half-width, injected from drive.py
let frame=null,road=null,tr={sober:[],drugged:[]},drugs=[],lastRace=-1;
const d0=new URLSearchParams(location.search).get('drug');
fetch('/road').then(r=>r.json()).then(r=>{road=r.pts;});
fetch('/drugs').then(r=>r.json()).then(d=>{drugs=d;
  $('drug').innerHTML=d.map(x=>`<option${x==d0?' selected':''}>${x}</option>`).join('');});
$('drug').addEventListener('change',e=>fetch('/reset?drug='+e.target.value));
$('restart').addEventListener('click',()=>fetch('/reset?drug='+$('drug').value));

function draw(cv,key,color){
  const c=cv.getContext('2d'),w=cv.width=cv.clientWidth*2,h=cv.height=264;
  c.clearRect(0,0,w,h);
  const car=frame.cars[key], s=car.dist, span=90, sc=w/span;
  const toY=v=>h/2+v*sc*0.55;
  const toX=v=>(v-(s-span*0.15))*sc;
  // road band
  c.beginPath();
  for(let i=0;i<=span;i+=2){const d=s-span*0.15+i;c.lineTo(toX(d),toY(-LANE));}
  for(let i=span;i>=0;i-=2){const d=s-span*0.15+i;c.lineTo(toX(d),toY(LANE));}
  c.closePath();c.fillStyle='#141610';c.fill();
  c.strokeStyle='#2c3024';c.lineWidth=2;c.stroke();
  // centreline
  c.beginPath();c.setLineDash([14,14]);c.strokeStyle='#3a3f2c';c.lineWidth=2;
  for(let i=0;i<=span;i+=2){const d=s-span*0.15+i;c.lineTo(toX(d),toY(0));}
  c.stroke();c.setLineDash([]);
  // travelled trail
  const t=tr[key];c.beginPath();
  t.forEach((e,i)=>{const d=s-(t.length-i)*0.12*K;c.lineTo(toX(d),toY(e));});
  c.strokeStyle=color;c.globalAlpha=.35;c.lineWidth=3;c.stroke();c.globalAlpha=1;
  // car
  const cx=toX(s),cy=toY(car.e);
  c.save();c.translate(cx,cy);c.rotate(-car.theta*2.2);
  c.fillStyle=color;c.fillRect(-9,-6,18,12);
  c.fillStyle='#0a0b09';c.fillRect(2,-4,5,8);c.restore();
  // crash flash
  if(car.crashed){c.fillStyle='rgba(255,60,40,.10)';c.fillRect(0,0,w,h);}
}
let errs=0;
function loop(){requestAnimationFrame(loop);if(!frame)return;
  try{drawFrame();}catch(e){if(errs++<3)$('foot').textContent='page error: '+e.message;}
}
function drawFrame(){
  draw($('cs'),'sober','#c8f542');draw($('cd'),'drugged','#ff7a59');
  const S=frame.cars.sober,D=frame.cars.drugged;
  tr.sober.push(S.e);tr.drugged.push(D.e);
  if(tr.sober.length>240){tr.sober.shift();tr.drugged.shift();}
  $('s_e').textContent=S.e.toFixed(2);$('d_e').textContent=D.e.toFixed(2);
  $('s_mae').textContent=S.mean_abs_err.toFixed(2);$('d_mae').textContent=D.mean_abs_err.toFixed(2);
  $('s_lane').textContent=(S.in_lane*100).toFixed(0);$('d_lane').textContent=(D.in_lane*100).toFixed(0);
  $('s_d').textContent=S.dist.toFixed(0);$('d_d').textContent=D.dist.toFixed(0);
  $('s_sp').textContent=S.spikes.toFixed(0);$('d_sp').textContent=D.spikes.toFixed(0);
  $('dname').textContent=frame.drug;$('track').textContent=frame.track;$('race').textContent=frame.race;
  if(frame.done)$('foot').textContent=`race ${frame.race} over - sober ${S.mean_abs_err.toFixed(2)} m vs ${frame.drug} ${D.mean_abs_err.toFixed(2)} m mean lane error`;
  else $('foot').textContent="same connectome. same decoder. same road. only the brain's global parameters differ.";
}
loop();
const es=new EventSource('/stream');
es.onmessage=e=>{frame=JSON.parse(e.data);
  if(frame.race!==lastRace){lastRace=frame.race;tr={sober:[],drugged:[]};}};
</script></body></html>"""

# The page template is a plain string, so the lane width is substituted in from
# the one definition in drive.py. A bare `LANE` in the JS throws a
# ReferenceError every animation frame and the canvases never draw.
PAGE = PAGE.replace("__LANE__", repr(float(LANE)))


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
        p = self.path
        if p.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    frame = self.runner.latest()
                    self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1 / 30.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif p.startswith("/road"):
            self._send(json.dumps({"pts": self.runner.road_points()}).encode(), "application/json")
        elif p.startswith("/drugs"):
            self._send(json.dumps(DRUGS).encode(), "application/json")
        elif p.startswith("/reset"):
            drug = self.runner.drug
            if "drug=" in p:
                from urllib.parse import unquote
                q = unquote(p.split("drug=", 1)[1].split("&")[0])
                if q in DRUGS:
                    drug = q
            self.runner.request_reset(drug)   # clock thread does the work
            self._send(json.dumps({"drug": drug, "queued": True}).encode(),
                       "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", default=DRUGS[0], choices=DRUGS)
    ap.add_argument("--port", type=int, default=8932)
    ap.add_argument("--seed", type=int, default=64)
    args = ap.parse_args()
    if not (HERE / "readout.npz").exists():
        raise SystemExit("no flybench/readout.npz - run: python flybench/drive.py --dump-only")
    print("loading connectome + decoder ...")
    Handler.runner = Runner(args.drug, args.seed)
    Handler.runner.start_clock(30.0)
    print(f"flydrive live on http://127.0.0.1:{args.port}  (drug: {args.drug})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
