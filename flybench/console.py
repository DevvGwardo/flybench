"""FlyConsole: one page, everything. For streaming.

Combines the two consoles into a single responsive full-page view:

    left   the race       sober vs drugged, same connectome, same decoder
    right  the brain      166,700 neurons at real EM coordinates, firing live
    below  the readout    real series vs what the brain predicts (held out)
    footer the scoreboard the measured results, read from the JSON on disk

One process, ONE clock thread. numba's workqueue layer is not thread-safe, so
all three brains (the readout reservoir plus the two racers) are stepped
sequentially from a single thread; every SSE client just reads the latest frame.

    python flybench/console.py --port 8933

    /            the page
    /stream      SSE, one combined frame
    /positions   real EM coordinates + region per point
    /road        the road centreline + the track constants
    /drugs       the available presets
    /results     the measured benchmark tables
    /reset?drug= switch the drug the race is run on
"""
from __future__ import annotations

import argparse
import collections
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

import drive_live  # noqa: E402
from drive_live import DRUGS, Runner as DriveRunner  # noqa: E402
from live import REGIONS, Runner as BrainRunner  # noqa: E402

# Each emitted frame advances the race by this many physics steps. 3 keeps the
# whole combined tick (1 readout brain + 2 racer brains) inside a 30 fps budget.
drive_live.K_STEPS = 3
HOLD_AFTER_RACE = 4.0          # seconds to hold a finished race before restarting


class Combined:
    def __init__(self, task: str, drug: str, seed: int = 64):
        print("fitting the readout ...")
        self.brain = BrainRunner(task)
        print("building the race ...")
        self.drive = DriveRunner(drug, seed)
        self._frame: dict | None = None
        self._stop = threading.Event()
        self._pending_drug: str | None = None
        self._done_at: float | None = None
        self.ticks = 0
        self.fps = 0.0
        self._recent: collections.deque = collections.deque(maxlen=240)

    # ---------------------------------------------------------------- clock

    def start_clock(self, fps: float = 30.0) -> None:
        self._clock = threading.Thread(target=self._run, args=(fps,), daemon=True)
        self._clock.start()

    def request_reset(self, drug: str | None = None) -> None:
        """Safe from any thread: touches a string and nothing else.

        An HTTP thread must never call into the brains. Doing so races the clock
        thread inside numba's workqueue and wedges the process.
        """
        self._pending_drug = drug or self.drive.drug

    def _run(self, fps: float) -> None:
        """Step everything, forever. Wrapped so no bug can kill the clock: a
        dead clock thread leaves the page frozen with no visible error."""
        period = 1.0 / fps
        while not self._stop.is_set():
            try:
                self._tick(period)
            except Exception as e:
                print(f"clock error: {type(e).__name__}: {e}", flush=True)
                time.sleep(0.5)

    def _tick(self, period: float) -> None:
        # a drug switch requested by a browser is performed HERE, on the clock
        # thread, and nowhere else
        if self._pending_drug is not None:
            drug, self._pending_drug = self._pending_drug, None
            self.drive.reset(drug)
            self._done_at = None

        t0 = time.monotonic()
        b = self.brain.step()
        r = self.drive.step()

        # hold a finished race long enough to read, then run the next road
        if r["done"]:
            if self._done_at is None:
                self._done_at = time.monotonic()
            elif time.monotonic() - self._done_at > HOLD_AFTER_RACE:
                self.drive.reset(self.drive.drug)
                self._done_at = None
                return
        else:
            self._done_at = None

        self.ticks += 1
        now = time.monotonic()
        self._recent.append(now)
        while self._recent and self._recent[0] < now - 2.0:      # rolling 2 s
            self._recent.popleft()
        if len(self._recent) > 1:
            self.fps = (len(self._recent) - 1) / (now - self._recent[0])
        self._frame = {"brain": b, "race": r, "tick": self.ticks,
                       "fps": round(self.fps, 1)}
        time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def latest(self) -> dict:
        """Lock-free: the clock replaces the whole dict, never mutates it."""
        while self._frame is None:
            time.sleep(0.01)
        return self._frame

    # ---------------------------------------------------------------- payloads

    def positions(self) -> dict:
        pts = self.brain.pts
        # worst-case extents, so the projection can be fitted to the panel
        # instead of guessing a scale and leaving the cloud as a blob
        radius_xz = float(np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2).max())
        radius_y = float(np.abs(pts[:, 1]).max())
        return {"pts": pts.tolist(),
                "region": self.brain.pts_region.tolist(),
                "regions": REGIONS,
                "radius_xz": radius_xz,
                "radius_y": radius_y}

    def road(self) -> dict:
        return {"pts": self.drive.road_points(),
                "lane": float(drive_live.LANE),
                "off_road": float(drive_live.OFF_ROAD),
                "track": float(drive_live.TRACK)}


def load_results() -> dict:
    out: dict = {}
    for key, name in (("race", "drive_race.json"), ("drugs", "drive_results.json"),
                      ("arena", "results.json"),
                      ("gain_mackey", "probe6_mackey.json"),
                      ("gain_brent", "probe6_brent.json")):
        try:
            out[key] = json.loads((HERE / name).read_text())
        except (OSError, json.JSONDecodeError):
            out[key] = {}
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fly console</title>
<style>
:root{
  --bg:#0a0b09; --panel:#0e100b; --line:#23271c; --line2:#2f3427;
  --fg:#e8e4d9; --dim:#8f8b7e; --sober:#c8f542; --drugged:#ff7a59;
  --truth:#6b7a3a; --hot:#ff5df2; --gold:#d9b96a;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  --display:Syne,system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--bg); color:var(--fg); font:14px/1.5 var(--mono);
  display:flex; flex-direction:column; overflow-x:hidden;
}
/* ---------- header ---------- */
header{
  display:flex; align-items:center; gap:clamp(12px,2vw,34px); flex-wrap:wrap;
  padding:clamp(10px,1.4vh,18px) clamp(14px,1.6vw,26px);
  border-bottom:1px solid var(--line); background:linear-gradient(180deg,#0d0f0a,#0a0b09);
}
h1{
  font:700 clamp(15px,1.5vw,22px)/1 var(--display); letter-spacing:.18em;
  color:var(--sober); white-space:nowrap;
}
h1 small{display:block;font:400 clamp(9px,.72vw,11px)/1.6 var(--mono);
  letter-spacing:.16em;color:var(--dim);margin-top:3px}
.stats{display:flex;gap:clamp(12px,1.6vw,30px);flex-wrap:wrap;align-items:flex-end}
.stat{display:flex;flex-direction:column;gap:2px}
.stat .k{font-size:clamp(12px,.7vw,12px);letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
.stat .v{font-size:clamp(15px,1.5vw,24px);font-variant-numeric:tabular-nums;font-weight:500}
.verdict{margin-left:auto;text-align:right;display:flex;flex-direction:column;gap:4px}
.verdict .who{font-size:clamp(11px,.95vw,15px);letter-spacing:.1em}
.verdict .win{font:700 clamp(14px,1.6vw,26px)/1 var(--display);letter-spacing:.06em}
.ctl{display:flex;gap:8px;align-items:center}
select,button{
  background:#141610;color:var(--fg);border:1px solid var(--line2);
  padding:7px 11px;font:inherit;font-size:clamp(11px,.85vw,13px);cursor:pointer;
  border-radius:2px;
}
select:hover,button:hover{border-color:var(--sober)}
button:active{background:#1c2015}
/* ---------- layout ---------- */
main{
  flex:1; display:grid; gap:clamp(10px,1vw,16px);
  padding:clamp(10px,1.2vh,18px) clamp(14px,1.6vw,26px);
  grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);
  grid-template-areas:"race brain" "read read";
  align-content:start;
}
@media (max-width:1080px){
  main{grid-template-columns:minmax(0,1fr);grid-template-areas:"race" "brain" "read"}
}
.panel{
  border:1px solid var(--line); background:var(--panel); border-radius:3px;
  padding:clamp(9px,1vw,14px); min-width:0; display:flex; flex-direction:column;
}
#pRace{grid-area:race} #pBrain{grid-area:brain} #pRead{grid-area:read}
.h{
  display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
  font:600 clamp(10px,.85vw,13px)/1 var(--display); letter-spacing:.16em;
  text-transform:uppercase; color:var(--dim); margin-bottom:9px;
}
.h b{color:var(--fg);font-weight:600}
.h .sp{margin-left:auto;font:400 clamp(9px,.72vw,11px)/1 var(--mono);
  letter-spacing:.08em;text-transform:none;color:var(--dim)}
/* ---------- race ---------- */
.lane{position:relative;border:1px solid var(--line);border-radius:2px;overflow:hidden}
.lane + .lane{margin-top:8px}
.lane canvas{display:block;width:100%;height:clamp(110px,20vh,230px);background:#0a0c07}
.lane .tag{
  position:absolute;top:clamp(16px,11%,26px);left:10px;display:flex;gap:9px;align-items:center;
  font:700 clamp(12px,.85vw,13px)/1 var(--display);letter-spacing:.16em;
  padding:5px 9px;border-radius:2px;background:rgba(10,11,9,.72);
  border:1px solid var(--line2);backdrop-filter:blur(2px);
  text-shadow:0 1px 2px #000;
}
.lane .tag .dot{width:9px;height:9px;border-radius:50%}
#laneS .tag{color:var(--sober)} #laneS .tag .dot{background:var(--sober)}
#laneD .tag{color:var(--drugged)} #laneD .tag .dot{background:var(--drugged)}
.lane .nums{
  position:absolute;top:clamp(16px,11%,26px);right:10px;display:flex;gap:14px;
  font-size:clamp(12px,.85vw,13px);font-variant-numeric:tabular-nums;
  background:rgba(10,11,9,.72);padding:5px 9px;border-radius:2px;
  border:1px solid var(--line2);backdrop-filter:blur(2px);
  text-shadow:0 1px 2px #000;
}
.lane .nums span{color:var(--dim)} .lane .nums b{color:var(--fg);font-weight:500}
.badge{
  position:absolute;bottom:clamp(16px,11%,26px);left:10px;font:700 clamp(12px,.85vw,13px)/1 var(--display);
  letter-spacing:.18em;padding:5px 9px;border-radius:2px;display:none;
  background:rgba(120,20,10,.88);color:#ffd9d0;border:1px solid #b4402c;
  text-shadow:0 1px 2px #000;
}
.badge.on{display:block}
/* ---------- scoreboard ---------- */
.board{display:grid;gap:4px;margin-top:10px}
.row{
  display:grid;grid-template-columns:minmax(74px,1fr) repeat(3,minmax(52px,auto));
  gap:8px;align-items:center;font-size:clamp(12px,.82vw,13px);
  padding:4px 6px;border-radius:2px;font-variant-numeric:tabular-nums;
}
.row.head{color:var(--dim);font-size:clamp(12px,.7vw,12px);letter-spacing:.1em;text-transform:uppercase}
.row.cur{background:#14180f}
.row b{font-weight:500}
.row .m{text-align:right;color:var(--dim)} .row .m b{color:var(--fg)}
/* ---------- brain ---------- */
#brain{display:block;width:100%;height:clamp(200px,34vh,420px);background:#08090600;border-radius:2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:9px;
  font-size:clamp(12px,.85vw,13px);color:var(--dim);text-shadow:0 1px 2px #000}
.legend i{display:inline-block;width:8px;height:8px;border-radius:1px;margin-right:5px;vertical-align:middle}
.bars{display:grid;gap:3px;margin-top:11px}
.bar{display:grid;grid-template-columns:minmax(96px,132px) 1fr minmax(46px,58px);
  gap:9px;align-items:center;font-size:clamp(12px,.85vw,13px);text-shadow:0 1px 2px #000}
.bar .t{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar .track{height:7px;background:#171a12;border-radius:2px;overflow:hidden}
.bar .fill{display:block;height:100%;background:var(--sober);transition:width .08s linear}
.bar .n{text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
/* ---------- readout ---------- */
#sig{display:block;width:100%;height:clamp(110px,17vh,190px);background:#0b0d08;border-radius:2px}
/* ---------- footer ---------- */
footer{
  border-top:1px solid var(--line);padding:clamp(8px,1vh,13px) clamp(14px,1.6vw,26px);
  color:var(--dim);font-size:clamp(12px,.85vw,13px);line-height:1.65;
  display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;
}
footer b{color:var(--fg);font-weight:500}
footer .err{color:var(--drugged)}
.sprite{position:absolute;width:0;height:0;overflow:hidden}
</style></head>
<body>

<svg class="sprite" aria-hidden="true"><defs>
  <!-- top-down open-wheel racer, nose at +x. currentColor is the team colour. -->
  <symbol id="car" viewBox="-25 -15 50 30">
    <!-- contact shadow -->
    <ellipse cx="-1" cy="1" rx="21" ry="11" fill="#000" opacity=".35"/>
    <!-- rear wing -->
    <rect x="-24" y="-12.5" width="5" height="25" rx="1.4" fill="#111309"/>
    <rect x="-24" y="-12.5" width="5" height="2.4" rx="1.1" fill="currentColor" opacity=".55"/>
    <rect x="-24" y="10.1" width="5" height="2.4" rx="1.1" fill="currentColor" opacity=".55"/>
    <!-- wheels: rear then front, with rims -->
    <rect x="-18" y="-14" width="10.5" height="5.6" rx="2.4" fill="#0a0b08"/>
    <rect x="-18" y="8.4" width="10.5" height="5.6" rx="2.4" fill="#0a0b08"/>
    <rect x="9.5" y="-14" width="9.5" height="5.6" rx="2.4" fill="#0a0b08"/>
    <rect x="9.5" y="8.4" width="9.5" height="5.6" rx="2.4" fill="#0a0b08"/>
    <rect x="-16.5" y="-12.6" width="7.5" height="2.6" rx="1.2" fill="#3a3f2c" opacity=".8"/>
    <rect x="-16.5" y="10" width="7.5" height="2.6" rx="1.2" fill="#3a3f2c" opacity=".8"/>
    <rect x="11" y="-12.6" width="6.6" height="2.6" rx="1.2" fill="#3a3f2c" opacity=".8"/>
    <rect x="11" y="10" width="6.6" height="2.6" rx="1.2" fill="#3a3f2c" opacity=".8"/>
    <!-- front wing -->
    <rect x="18" y="-13" width="6" height="26" rx="1.4" fill="#111309"/>
    <rect x="18" y="-13" width="6" height="2.6" rx="1.2" fill="currentColor" opacity=".6"/>
    <rect x="18" y="10.4" width="6" height="2.6" rx="1.2" fill="currentColor" opacity=".6"/>
    <!-- sidepods + floor -->
    <path d="M-14,-9.6 L6,-8.4 L8,-3 L8,3 L6,8.4 L-14,9.6 Q-16,0 -14,-9.6 Z"
          fill="currentColor" opacity=".55"/>
    <!-- main hull -->
    <path d="M-19,-6.2 L4,-6.6 L15,-4 Q19.6,0 15,4 L4,6.6 L-19,6.2 Q-21.4,0 -19,-6.2 Z"
          fill="currentColor" stroke="#070805" stroke-width="1.2"/>
    <!-- livery stripe -->
    <path d="M-18.4,-1.5 L15,-1.5 L17.4,0 L15,1.5 L-18.4,1.5 Q-19.4,0 -18.4,-1.5 Z"
          fill="#fff" opacity=".26"/>
    <!-- engine cover intake -->
    <rect x="-9" y="-3.4" width="4.5" height="6.8" rx="1.6" fill="#070805" opacity=".9"/>
    <!-- cockpit + halo -->
    <path d="M-4,-4.4 L7,-3.9 Q10.4,0 7,3.9 L-4,4.4 Q-7.6,0 -4,-4.4 Z" fill="#070805"/>
    <path d="M-4,-4.4 L7,-3.9 Q10.4,0 7,3.9 L-4,4.4 Q-7.6,0 -4,-4.4 Z"
          fill="none" stroke="#e8e4d9" stroke-width="1.1" opacity=".5"/>
    <ellipse cx="1.6" cy="0" rx="2.6" ry="2.1" fill="#c9c4b4" opacity=".55"/>
    <!-- nose tip -->
    <path d="M15,-4 Q19.6,0 15,4 L17.6,0 Z" fill="#fff" opacity=".18"/>
    <circle cx="19.2" cy="-2.6" r="1.15" fill="#fff8d0"/>
    <circle cx="19.2" cy="2.6" r="1.15" fill="#fff8d0"/>
    <!-- taillight bar -->
    <rect x="-21.4" y="-4.2" width="2.6" height="8.4" rx="1.2" fill="#ff3b1f" opacity=".95"/>
  </symbol>
</defs></svg>

<header>
  <h1>FLY CONSOLE<small>166,700 NEURONS &middot; 25.6M SYNAPSES</small></h1>
  <div class="stats">
    <div class="stat"><span class="k">decode score</span><span class="v" id="score">--</span></div>
    <div class="stat"><span class="k">spikes / step</span><span class="v" id="fired">--</span></div>
    <div class="stat"><span class="k">tick rate</span><span class="v" id="fps">--</span></div>
    <div class="stat"><span class="k">task</span><span class="v" id="task">--</span></div>
  </div>
  <div class="ctl">
    <select id="drug" aria-label="drug for the drugged car"></select>
    <button id="restart">new race</button>
  </div>
  <div class="verdict">
    <span class="who" id="who">race starting</span>
    <span class="win" id="win">&nbsp;</span>
  </div>
</header>

<main>
  <section class="panel" id="pRace">
    <div class="h">the race <b>sober vs drugged</b>
      <span class="sp" id="raceinfo">same connectome &middot; same decoder &middot; same road</span></div>

    <div class="lane" id="laneS">
      <canvas id="cs" role="img" aria-label="sober car position on track"></canvas>
      <div class="tag"><i class="dot"></i>SOBER</div>
      <div class="nums">
        <span>offset <b id="s_e">0</b> m</span>
        <span>mean <b id="s_mae">0</b></span>
        <span>in-lane <b id="s_lane">0</b>%</span>
        <span>dist <b id="s_d">0</b> m</span>
      </div>
      <div class="badge" id="s_bad">OFF ROAD</div>
    </div>

    <div class="lane" id="laneD">
      <canvas id="cd" role="img" aria-label="drugged car position on track"></canvas>
      <div class="tag"><i class="dot"></i>ON <span id="dname">-</span></div>
      <div class="nums">
        <span>offset <b id="d_e">0</b> m</span>
        <span>mean <b id="d_mae">0</b></span>
        <span>in-lane <b id="d_lane">0</b>%</span>
        <span>dist <b id="d_d">0</b> m</span>
      </div>
      <div class="badge" id="d_bad">OFF ROAD</div>
    </div>

    <div class="board" id="board"></div>
  </section>

  <section class="panel" id="pBrain">
    <div class="h">the brain <b>live</b>
      <span class="sp" id="brainsp">real EM coordinates</span></div>
    <canvas id="brain" role="img" aria-label="live brain activity, 166700 neurons"></canvas>
    <div class="legend" id="rlegend"></div>
    <div class="bars" id="bars"></div>
  </section>

  <section class="panel" id="pRead">
    <div class="h">the readout <b>real series vs what the brain predicts</b>
      <span class="sp" id="hz"></span></div>
    <canvas id="sig" role="img" aria-label="readout: real series versus brain prediction"></canvas>
    <div class="legend">
      <span><i style="background:var(--truth)"></i>truth</span>
      <span><i style="background:var(--hot)"></i>prediction</span>
      <span id="seriesinfo"></span>
      <span>replaying the held-out test slice, left to right</span>
    </div>
  </section>
</main>

<footer>
  <span>a <b>real</b> <i>Drosophila</i> nervous system, wired as electron microscopy found it. nothing inside it is trained.</span>
  <span>the decoder is cloned from a hand-built driver, then <b>frozen</b> &mdash; only the brain's global parameters change.</span>
  <span id="err" class="err"></span>
</footer>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const RC = ["#2f3a1e","#3f4f26","#55682f","#7a9430","#a07a2e","#c8f542","#8fd0ff","#ff5df2"];

let P = [], PR = [], REG = [], road = [], ROAD = {lane:1.5, off_road:3.0, track:1200};
let EXT = {xz: 1, y: 1};   // measured cloud extent, set from /positions
let fitCur = 0;            // smoothed projection scale
let frame = null, tr = {sober:[], drugged:[]}, lastRace = -1, errs = 0, series = "";

// ---------------------------------------------------------------- canvases
const cvS = $('cs'), cvD = $('cd'), cvB = $('brain'), cvG = $('sig');
const cB = cvB.getContext('2d'), cG = cvG.getContext('2d');
const DPR = Math.min(2, window.devicePixelRatio || 1);

function fit(c){
  const w = c.clientWidth, h = c.clientHeight;
  if (w > 0 && h > 0) { c.width = Math.round(w*DPR); c.height = Math.round(h*DPR); }
}
function fitAll(){ [cvS,cvD,cvB,cvG].forEach(fit); }
addEventListener('resize', fitAll);
if ('ResizeObserver' in window) {
  let raf = 0;
  new ResizeObserver(()=>{ cancelAnimationFrame(raf); raf = requestAnimationFrame(fitAll); })
    .observe(document.body);
}

// ---------------------------------------------------------------- data
fetch('/positions').then(r=>r.json()).then(d=>{
  P = d.pts; PR = d.region; REG = d.regions;
  EXT = {xz: d.radius_xz || 1, y: d.radius_y || 1};
  fitCur = 0;
  $('rlegend').innerHTML = REG.map((n,i)=>
    `<span><i style="background:${RC[i%RC.length]}"></i>${n}</span>`).join('');
  $('bars').innerHTML = REG.map((n,i)=>
    `<div class="bar"><span class="t">${n}</span><span class="track"><i class="fill" id="bar${i}" style="background:${RC[i%RC.length]}"></i></span><span class="n" id="barv${i}">0</span></div>`).join('');
  fitAll();
}).catch(e=>fail(e));

fetch('/road').then(r=>r.json()).then(d=>{
  road = d.pts; ROAD = {lane:d.lane, off_road:d.off_road, track:d.track};
}).catch(e=>fail(e));

fetch('/drugs').then(r=>r.json()).then(d=>{
  $('drug').innerHTML = d.map(x=>`<option>${x}</option>`).join('');
}).catch(e=>fail(e));

fetch('/results').then(r=>r.json()).then(d=>{
  const rows = (d.race && d.race.rows) || [];
  let html = '';
  if (rows.length) {
    const order = rows.map(r=>r.model);
    html +=
      `<div class="row head"><span>driver</span><span class="m">lane err</span><span class="m">lost</span><span class="m">in-lane</span></div>`
      + order.map(m=>{
          const r = rows.find(x=>x.model===m);
          const med = r.median_abs_err ?? r.mean_abs_err ?? 0;
          return `<div class="row" data-model="${m}"><span>${m}</span>`
            + `<span class="m"><b>${med.toFixed(3)}</b></span>`
            + `<span class="m"><b>${r.lost_road ?? '-'}/${r.seeds ?? '-'}</b></span>`
            + `<span class="m"><b>${((r.in_lane ?? 0)*100).toFixed(0)}%</b></span></div>`;
        }).join('');
  }
  const arena = Array.isArray(d.arena) ? d.arena : [];
  if (arena.length) {
    const tasks = [...new Set(arena.map(r=>r.task).filter(Boolean))];
    let flyWins = 0, scored = 0;
    for (const t of tasks) {
      const rs = arena.filter(r=>r.task===t && typeof r.nrmse === 'number');
      if (!rs.length) continue;
      scored++;
      if (rs.reduce((a,b)=>a.nrmse<=b.nrmse?a:b).model === 'fly') flyWins++;
    }
    if (scored)
      html += `<div class="row"><span>arena benchmark</span>`
        + `<span class="m"><b>fly best NRMSE ${flyWins}/${scored} tasks</b></span></div>`;
  }
  if (html) $('board').innerHTML = html;
}).catch(e=>fail(e));

$('drug').addEventListener('change', e=>{
  fetch('/reset?drug=' + encodeURIComponent(e.target.value)).catch(err=>fail(err));
});
$('restart').addEventListener('click', ()=>{
  fetch('/reset?drug=' + encodeURIComponent($('drug').value)).catch(err=>fail(err));
});

// ---------------------------------------------------------------- stream
let hot = new Set(), gTruth = [], gPred = [], ang = 0;
const es = new EventSource('/stream');
es.onmessage = e => {
  try { frame = JSON.parse(e.data); } catch (err) { fail(err); return; }
  const b = frame.brain, r = frame.race;
  if (!b || !r) return;
  if (r.race !== lastRace) { lastRace = r.race; tr = {sober:[], drugged:[]}; }
  tr.sober.push(r.cars.sober.e); tr.drugged.push(r.cars.drugged.e);
  if (tr.sober.length > 300) { tr.sober.shift(); tr.drugged.shift(); }

  $('score').textContent = (b.score*100).toFixed(1) + '%';
  $('fired').textContent = b.fired;
  $('fps').textContent = frame.fps.toFixed(0);
  $('task').textContent = (b.series || 'btc').toUpperCase();
  $('hz').textContent = b.fired + ' neurons fired this step';

  const S = r.cars.sober, D = r.cars.drugged;
  $('s_e').textContent = S.e.toFixed(2); $('d_e').textContent = D.e.toFixed(2);
  $('s_mae').textContent = S.mean_abs_err.toFixed(2); $('d_mae').textContent = D.mean_abs_err.toFixed(2);
  $('s_lane').textContent = (S.in_lane*100).toFixed(0); $('d_lane').textContent = (D.in_lane*100).toFixed(0);
  $('s_d').textContent = S.dist.toFixed(0); $('d_d').textContent = D.dist.toFixed(0);
  $('dname').textContent = r.drug.toUpperCase();
  $('s_bad').className = 'badge' + (S.crashed ? ' on' : '');
  $('d_bad').className = 'badge' + (D.crashed ? ' on' : '');
  $('raceinfo').textContent = 'race ' + r.race + ' of ' + ROAD.track + ' m';
  $('seriesinfo').textContent = 'series: ' + (b.series||'btc');

  const d = D.mean_abs_err - S.mean_abs_err;
  if (r.done) {
    $('who').textContent = 'race ' + r.race + ' finished';
    $('win').textContent = Math.abs(d) < 0.05 ? 'DEAD HEAT'
      : (d > 0 ? 'DRUG LOSES' : 'DRUG WINS');
    $('win').style.color = Math.abs(d) < 0.05 ? 'var(--dim)'
      : (d > 0 ? 'var(--sober)' : 'var(--drugged)');
  } else {
    $('who').textContent = r.drug + ' vs sober';
    $('win').textContent = (d >= 0 ? '+' : '') + d.toFixed(3) + ' m';
    $('win').style.color = Math.abs(d) < 0.05 ? 'var(--dim)'
      : (d > 0 ? 'var(--sober)' : 'var(--drugged)');
  }

  const mx = Math.max(1, ...b.regions);
  for (let i = 0; i < REG.length; i++) {
    const v = b.regions[i] || 0;
    const f = $('bar'+i), n = $('barv'+i);
    if (f) f.style.width = (100*v/mx) + '%';
    if (n) n.textContent = v;
  }
  hot = new Set(b.hot);
  gTruth.push(b.y); gPred.push(b.pred);
  if (gTruth.length > 420) { gTruth.shift(); gPred.shift(); }
};
es.onerror = () => { $('err').textContent = 'stream disconnected - retrying'; };

function fail(e){ if (errs++ < 4) $('err').textContent = 'page error: ' + (e && e.message ? e.message : e); }

// ---------------------------------------------------------------- drawing
function drawRace(cv, key, color){
  const c = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  c.clearRect(0,0,w,h);
  if (!frame || !frame.race || !road.length) return;
  const car = frame.race.cars[key];
  const s = car.dist, span = 110, sc = w/span;
  const vscale = (h*0.46)/Math.max(ROAD.off_road, 0.1);   // road fills the lane
  const toX = v => (v - (s - span*0.15)) * sc;
  const toY = v => h/2 + v*vscale;

  // verge
  c.fillStyle = '#0a0c07'; c.fillRect(0,0,w,h);
  // asphalt
  c.beginPath();
  for (let i=0;i<=span;i+=2){ const d = s-span*0.15+i; c.lineTo(toX(d), toY(-ROAD.off_road)); }
  for (let i=span;i>=0;i-=2){ const d = s-span*0.15+i; c.lineTo(toX(d), toY(ROAD.off_road)); }
  c.closePath(); c.fillStyle = '#242819'; c.fill();
  // edge kerbs, red/white
  const band = ROAD.off_road * 0.16;
  for (const side of [-1, 1]) {
    for (let i=0;i<span;i+=4){
      const d0 = s-span*0.15+i;
      c.beginPath();
      c.moveTo(toX(d0), toY(side*ROAD.off_road));
      c.lineTo(toX(d0+4), toY(side*ROAD.off_road));
      c.lineTo(toX(d0+4), toY(side*(ROAD.off_road-band)));
      c.lineTo(toX(d0), toY(side*(ROAD.off_road-band)));
      c.closePath();
      c.fillStyle = (((d0/4)|0) % 2) ? '#d8d3c2' : '#b4402c';
      c.globalAlpha = 0.85; c.fill(); c.globalAlpha = 1;
    }
  }
  // lane boundary lines
  c.strokeStyle = '#5d6440'; c.lineWidth = 2.2*DPR;
  for (const e of [-ROAD.lane, ROAD.lane]) {
    c.beginPath();
    for (let i=0;i<=span;i+=2){ const d = s-span*0.15+i; c.lineTo(toX(d), toY(e)); }
    c.stroke();
  }
  // centre dashes
  c.beginPath(); c.setLineDash([15*DPR, 15*DPR]); c.strokeStyle = '#6e7550'; c.lineWidth = 3*DPR;
  for (let i=0;i<=span;i+=2){ const d = s-span*0.15+i; c.lineTo(toX(d), toY(0)); }
  c.stroke(); c.setLineDash([]);
  // distance markers every 100 m: line + label kept inside the asphalt,
  // clear of the kerb bands at both edges
  c.font = `${12*DPR}px ui-monospace,Menlo,monospace`;
  c.textBaseline = 'middle'; c.textAlign = 'center';
  const kerbIn = ROAD.off_road - band;
  for (let m = 100; m < ROAD.track; m += 100) {
    const x = toX(m);
    if (x < -40 || x > w+40) continue;
    c.strokeStyle = '#2b3020'; c.lineWidth = 1.5*DPR;
    c.beginPath(); c.moveTo(x, toY(-kerbIn)); c.lineTo(x, toY(kerbIn)); c.stroke();
    c.fillStyle = '#6b705c';
    c.fillText(m + 'm', x, toY(-kerbIn) + 11*DPR);
  }
  c.textAlign = 'start';

  // travelled trail, fading
  const t = tr[key];
  for (let i=1;i<t.length;i++){
    const d = s - (t.length-i)*0.08*6;
    c.beginPath();
    c.moveTo(toX(d), toY(t[i-1]));
    c.lineTo(toX(d), toY(t[i]));
    c.strokeStyle = color; c.globalAlpha = 0.05 + 0.35*(i/t.length);
    c.lineWidth = 3*DPR; c.stroke();
  }
  c.globalAlpha = 1;
  // the car, as a rotated SVG sprite
  const size = Math.max(74, h*0.62);
  const cx = Math.max(size*0.5, Math.min(w-size*0.5, toX(s)));
  const cy = toY(car.e);
  c.save(); c.translate(cx, cy); c.rotate(-car.theta*1.5);
  c.globalAlpha = car.crashed ? 0.5 : 1;
  const img = sprite(color);
  // gate on decode: never draw a blank car on first frames
  if (img && img.complete && img.naturalWidth > 0)
    c.drawImage(img, -size/2, -size*0.30, size, size*0.60);
  c.restore(); c.globalAlpha = 1;
  // crash wash
  if (car.crashed) { c.fillStyle = 'rgba(255,60,40,.10)'; c.fillRect(0,0,w,h); }
}

// render the SVG sprite once per colour, at high res, then reuse the bitmap
const sprites = {};
function sprite(color){
  if (sprites[color]) return sprites[color];
  const sym = document.getElementById('car');
  if (!sym) return null;
  // a bare <symbol> renders nothing; inline its children into a real <svg> root
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-23 -14 46 28"`
    + ` width="250" height="150" style="color:${color}">${sym.innerHTML}</svg>`;
  const img = new Image();
  img.decoding = 'async';
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  sprites[color] = img;
  return img;
}

function drawBrain(){
  const w = cvB.width, h = cvB.height;
  cB.clearRect(0,0,w,h);
  if (!P.length) return;
  ang += 0.0035;
  const ca = Math.cos(ang), sa = Math.sin(ang);
  // Fit to the extent ACTUALLY projected at this angle, not the cloud's max
  // radius. The max radius is only reached at some angles, so fitting to it
  // leaves the cloud at half the panel most of the time. Smoothed so the scale
  // does not breathe as it rotates.
  let mx = 1e-6, my = 1e-6;
  for (let i=0;i<P.length;i++){
    const p = P[i];
    const X = p[0]*ca - p[2]*sa;
    const ax = X < 0 ? -X : X, ay = p[1] < 0 ? -p[1] : p[1];
    if (ax > mx) mx = ax;
    if (ay > my) my = ay;
  }
  const target = Math.min((w*0.47)/mx, (h*0.47)/my);
  fitCur = fitCur > 0 ? fitCur + (target - fitCur)*0.08 : target;   // snap first
  const fit = fitCur;
  const hotSet = hot;
  const dot = Math.max(1.6, 1.9*DPR);
  for (let i=0;i<P.length;i++){
    const p = P[i];
    const X = p[0]*ca - p[2]*sa, Z = p[0]*sa + p[2]*ca;
    const px = w/2 + X*fit, py = h/2 + p[1]*fit;
    const depth = 0.5 + 0.5*(Z / Math.max(EXT.xz, 1e-6));   // -1..1 -> 0..1
    if (hotSet.has(i)) {
      cB.fillStyle = '#ff5df2';
      cB.fillRect(px-2.8*DPR, py-2.8*DPR, 5.6*DPR, 5.6*DPR);
    } else {
      cB.globalAlpha = 0.30 + 0.62*depth;
      cB.fillStyle = RC[PR[i]] || '#2a3216';
      cB.fillRect(px, py, dot, dot);
    }
  }
  cB.globalAlpha = 1;
}

function drawSig(){
  const w = cvG.width, h = cvG.height;
  cG.clearRect(0,0,w,h);
  if (gTruth.length < 2) return;
  // auto-scale to the data: a fixed +-3 window left both lines pinned at the top
  let lo = Infinity, hi = -Infinity;
  for (const v of gTruth) { if (v<lo) lo=v; if (v>hi) hi=v; }
  for (const v of gPred)  { if (v<lo) lo=v; if (v>hi) hi=v; }
  if (!isFinite(lo) || !isFinite(hi)) return;
  const pad = Math.max(0.12, (hi-lo)*0.12);
  lo -= pad; hi += pad;
  // inset the plot so the top/bottom gridlines + labels never clip at the edge
  const padT = 16*DPR, padB = 20*DPR;
  const Y = v => padT + (1-(v-lo)/(hi-lo))*(h-padT-padB);
  const n = gTruth.length, X = i => i/(n-1)*w;

  // gridlines with values
  cG.font = `${12*DPR}px ui-monospace,Menlo,monospace`;
  cG.textBaseline = 'middle'; cG.textAlign = 'start';
  for (let g=0; g<=4; g++){
    const v = lo + (hi-lo)*g/4, y = Y(v);
    cG.strokeStyle = '#1c2016'; cG.lineWidth = 1;
    cG.beginPath(); cG.moveTo(0,y); cG.lineTo(w,y); cG.stroke();
    cG.fillStyle = '#6b705c';
    cG.fillText(v.toFixed(1), 6*DPR, y);
  }
  const line = (arr, col) => {
    cG.strokeStyle = col; cG.lineWidth = 1.9*DPR; cG.beginPath();
    arr.forEach((v,i)=> i ? cG.lineTo(X(i),Y(v)) : cG.moveTo(X(i),Y(v)));
    cG.stroke();
  };
  line(gTruth, '#6b7a3a');
  line(gPred,  '#ff5df2');
}

sprite('#c8f542'); sprite('#ff7a59');   // decode both sprites up front

let sized = false;
function loop(){
  requestAnimationFrame(loop);
  if (!sized && cvB.clientWidth > 0) { fitAll(); sized = true; }
  try {
    drawRace(cvS, 'sober',  '#c8f542');
    drawRace(cvD, 'drugged','#ff7a59');
    drawBrain();
    drawSig();
  } catch (e) { fail(e); }
}
fitAll();
loop();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    app: Combined = None

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        p = self.path
        if p.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"data: " + json.dumps(self.app.latest()).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1 / 30.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif p.startswith("/positions"):
            self._json(self.app.positions())
        elif p.startswith("/road"):
            self._json(self.app.road())
        elif p.startswith("/drugs"):
            self._json(DRUGS)
        elif p.startswith("/results"):
            self._json(load_results())
        elif p.startswith("/reset"):
            drug = None
            if "drug=" in p:
                from urllib.parse import unquote
                q = unquote(p.split("drug=", 1)[1].split("&")[0])
                if q in DRUGS:
                    drug = q
            self.app.request_reset(drug)
            self._json({"drug": drug or self.app.drive.drug, "queued": True})
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="btc")
    ap.add_argument("--drug", default=DRUGS[0], choices=DRUGS)
    ap.add_argument("--port", type=int, default=8933)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    Handler.app = Combined(args.task, args.drug)
    Handler.app.start_clock(args.fps)
    print(f"fly console on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
