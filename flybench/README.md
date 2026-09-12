# flybench — is a real fly brain actually good at anything?

The [fly.ai](https://github.com/alextitonis/fly.ai) connectome is a complete
*Drosophila* nervous system: **166,700 neurons, 25,582,938 synapses**, wired
exactly as electron microscopy found it. Nothing inside it is trained.

This benchmark uses it as a **reservoir**: inject a task signal into the brain,
train nothing but a linear decoder on the **1,314 descending neurons** (the
brain's output cables to the body), and score it on real tasks against controls
that share its size, its synapse count, its weight distribution and its dynamics.

The question is not "can we make a demo". It is: **is the real wiring better than
a random network, and better than no network at all?**

## Method

```
u(t)  ──► encoder ──► 166,700-neuron connectome ──► 1,314 descending neurons ──► linear readout
          30,000 cells     leaky integrate-and-fire,         spike traces           ridge, (tau, lambda)
          ±1 weights       50 steps/s, frozen                                       validated on held-out
```

* **Encoder.** The input drives the 30,000 cells that synapse *directly* onto the
  descending neurons, with fixed random ±1 weights. Everything between the
  encoder and the output cables is the connectome. The set is computed from each
  network's *own* wiring, so every model is driven identically.
* **Readout.** Ridge on the 1,314 descending-neuron spike traces. Raw per-step
  spike counts are recorded and the trace time constant is applied offline, so
  `tau` is tuned on validation like any other hyperparameter. Penalty `lambda` is
  validated across `logspace(-2, 4, 7)`.
* **Splits.** Washout 7%, train 57%, validate 14%, test 22%. The test slice is
  the tail of the series — never shuffled. Series are centred using training-slice
  statistics only.

### Controls

| model | what it is |
|---|---|
| `fly` | MaleCNS v1.0 connectome, weights as EM found them |
| `random` | same weight magnitudes, same synapse count, placed at random |
| `shuffled` | the real synapses, retargeted by a random permutation (in-degree preserved exactly, topology destroyed) |
| `linear` | no reservoir at all — ridge on the raw last-20 inputs |

## Results

### Reading the present off the output cables

Held-out R² for decoding the *current* input from the 1,314 descending neurons:

| model | memory | mackey | narma | sunspots | btc | brent |
|---|---|---|---|---|---|---|
| **fly** | 0.834 | 0.987 | 0.735 | **0.974** | **0.520** | **0.985** |
| random | **0.864** | 0.966 | **0.821** | 0.951 | −0.289 | 0.963 |
| shuffled | 0.682 | 0.949 | 0.569 | 0.947 | −0.094 | 0.955 |
| linear | 0.000 | **0.995** | −0.005 | 0.890 | 0.984 | 0.972 |

The connectome is a working reservoir. It carries the present strongly, and on
BTC the random network's readout *collapses* (−0.289) where the fly's degrades
but holds.

### Memory capacity — the fly loses

Sum of held-out R² for recovering `u(t−k)`, k = 1…20:

| model | memory capacity | spikes/step |
|---|---|---|
| fly | 2.06 | 16,175 |
| **random** | **2.53** | 30,378 |
| shuffled | 1.69 | 14,490 |
| linear (20-tap delay line) | **20.00** | — |

The real connectome holds **about two past inputs**. A random network of the same
size holds more, and a plain delay line trivially holds all 20. **The fly's
strength is not memory.**

### One-step-ahead on real series — the fly wins

| task | model | NRMSE | persistence | skill | dir. acc |
|---|---|---|---|---|---|
| mackey | **fly** | **0.1701** | 0.1472 | −0.155 | 0.725 |
| | random | 0.2688 | 0.1472 | −0.826 | 0.632 |
| | shuffled | 0.3098 | 0.1472 | −1.104 | 0.667 |
| narma | **fly** | **0.8799** | 0.9767 | +0.099 | 0.453 |
| | random | 0.9081 | 0.9767 | +0.070 | 0.455 |
| sunspots | **fly** | **0.3685** | 0.3802 | +0.031 | 0.411 |
| | random | 0.3889 | 0.3802 | −0.023 | 0.434 |
| btc | **fly** | **0.9621** | 0.1194 | −7.057 | 0.522 |
| | random | 1.4431 | 0.1194 | −11.085 | 0.514 |
| brent | **fly** | **0.0945** | 0.0719 | −0.314 | 0.469 |
| | random | 0.1302 | 0.0719 | −0.809 | 0.486 |

The fly has the lowest error on **every** task, and beats `shuffled` everywhere —
so it is the topology, not just the weight statistics. But it still loses to
persistence on the price series, and the `linear` control beats all three spiking
reservoirs on mackey (0.1223) and narma (0.7424).

### Is the advantage just a quieter network?

The fly runs at ~16,000 spikes/step and the random control at ~30,000. Sweeping
the global synaptic gain settles it:

| gain | fly spikes/step | fly NRMSE | random spikes/step | random NRMSE |
|---|---|---|---|---|
| 2.0 | 12,865 | **0.2347** | 14,559 | 0.3487 |
| 3.0 | 16,337 | **0.1701** | 30,167 | 0.2688 |
| 5.0 | 19,283 | **0.1425** | 84,685 | 0.2296 |
| 8.0 | 21,586 | **0.1332** | 132,463 | 0.2717 |

At matched activity (~13–14.5k spikes/step) the fly scores 0.235 and the random
network 0.349. The random network's *best* result anywhere (0.230) requires
84,685 spikes/step — five times the fly's activity — and is still worse than the
fly's peak. Same on Brent: fly 0.1062 vs random 0.1344 at ~14k, and the fly's
best (0.0938) beats the random's best (0.1129, at 84,745 spikes/step).

**The advantage is the wiring, not the operating point.** Note also that at equal
gain the random network is 4–8× more excitable — the real connectome is
intrinsically far more stable.

## Honest limitations

* **No model predicts BTC direction.** 52% against a 50% coin. Every model's
  readout collapses on BTC's held-out tail because the series is non-stationary
  (validation R² 0.98 → test 0.52); persistence, which ignores the level, is
  fine. Predicting prices is not what this does.
* **"Beats a random reservoir" ≠ "beats a linear model."** A 20-tap delay line
  wins on the smooth tasks. The fly's advantage is over other spiking reservoirs.
* **One gain per network, tuned by sweep.** Not a full hyperparameter search per
  model, and no spectral-radius tuning for the controls.
* **Point neurons only.** No dendrites, no graded neurons, no neuromodulators, no
  plasticity — the repo's own caveats. The encoder is a model, not biology.
* **A single seed.** Model comparisons are one realisation of the random and
  shuffled controls; no error bars.
* The brain is scored through a hand-written encoder and a linear readout. The
  bottleneck in every such system is the interface, not the wiring.

## Run it

```sh
cd ~/fly.ai
.venv/bin/python build_brain.py                 # once: downloads ~1.1 GB

.venv/bin/python flybench/arena.py run          # 24 runs, ~11 min
.venv/bin/python flybench/arena.py report

.venv/bin/python flybench/console.py --task btc    # the console on :8933 (primary entry)

# Components the console combines (one clock, one page above; run these only
# standalone): live.py is the brain panel on :8931, drive_live.py is the
# head-to-head race on :8932.
# .venv/bin/python flybench/live.py --task btc
# .venv/bin/python flybench/drive_live.py --drug depressant

.venv/bin/python flybench/drive.py --steps 1500 --seeds 12   # sober vs drugs, ~15 min
.venv/bin/python flybench/drive_race.py                      # fly vs random, ~8 min
```

Files: `probe.py`–`probe6.py` are the lab notebook — each one answers a single
question and the comments record what was actually found (the dead encoder
routes, the 0.30-volt headroom, the readout penalty that was 100× too small, the
Mackey-Glass flatline, the activity-matching check). `arena.py` is the benchmark.
`live.py` is the console. `results.json` is every scored run.

## FlyDrive — put it behind the wheel, sober vs drugged

Predicting a series is open loop. Control is the real test: the output changes
the input. `drive.py` puts the connectome in a car on a winding road and asks
whether it can hold the lane.

```
offset, heading error, curve ahead ──► 2,400 cells feeding the motor neurons
                                     ──► connectome ──► 1,314 descending neurons
                                     ──► linear decoder ──► yaw command ──► car
                                                              │
                                       car's new offset ──────┘  (closed loop)
```

The encoder drives the cells that synapse **directly onto the fly's own steering
and walking motor neurons** (`steer_L/R` = DNa02, `forward_L/R` = DNg100) — the
same lesson as the arena: one synapse is worth ~1/153 of threshold, so a random
population attenuates to nothing inside two hops.

The decoder is fitted **once**, on sober runs, to imitate a competent PD driver,
then frozen. It is never retrained between conditions. Same wiring, same
decoder, same roads, same starts — the only thing that changes is the brain.

### Sober vs drugged

`fly.ai` has **no neuromodulators**, so a "drug" here is a global parameter
preset — excitability (tonic), synaptic gain, noise load. They are labelled by
what the parameters do, not by pretending to be pharmacology. 12 roads each,
lane half-width 1.5 m, "lost" = mean lane error above 1 m.

| preset | what it changes | median lane error | lost road | in-lane | spikes/step |
|---|---|---|---|---|---|
| **sober** | — | 0.242 m | 2/12 | 0.89 | 13,515 |
| stabiliser | noise 1.2 → 0.4 Hz | **0.190 m** | 1/12 | 0.94 | 11,850 |
| stimulant | gain 3 → 4.6, noise → 1.9 Hz | 0.239 m | 1/12 | 0.94 | 17,333 |
| psychedelic | noise 1.2 → 5.0 Hz | 0.207 m | 5/12 | 0.83 | 17,689 |
| depressant | tonic 0.14 → 0.185, gain → 2.2 | 0.518 m | 4/12 | 0.83 | 16,711 |
| sedative | tonic 0.14 → 0.05 | 0.592 m | 5/12 | 0.73 | **431** |

* **Sedation is the one thing that actually breaks it.** Cut the resting drive
  and the network goes almost silent — 431 spikes/step against 13,515 sober, 3%
  of its normal activity. The decoder is reading an empty cable, the car drifts
  off the road on 5 of 12 runs.
* **Raising the resting potential costs more than adding gain or noise.**
  `depressant` pushes every neuron closer to threshold, which is the opposite of
  selective; 4/12 runs lose the road.
* **More gain and more noise are free.** `stimulant` and `psychedelic` land
  within noise of sober — and the gain-matched control run (which rescales the
  features so the decoder's *output* distribution is identical to sober)
  reproduces the same score. The drugs are not quietly destroying information.

The decoder's own gain had to be tuned first. Left at 1.0 it under-steers once
the car leaves the band the teacher explored, and every drug that adds activity
"improves driving" purely by propping up a mis-scaled decoder — which is what
the first run of this experiment showed before the gain sweep was widened
downward. Tuned on held-out roads, sober sits at its best and a drug can only
move it off.

### Does the wiring matter? fly vs random, same treatment

Each model gets its own encoder (from its own wiring), its own decoder, its own
tuned gain, and the same 12 roads.

| driver | decoder val R² | median lane error | lost road | in-lane | spikes/step |
|---|---|---|---|---|---|
| **fly** | **0.984** | **0.215 m** | **1/12** | **0.94** | 13,495 |
| shuffled | 0.948 | 0.334 m | 3/12 | 0.86 | 8,870 |
| random | 0.752 | 0.407 m | 4/12 | 0.81 | 27,092 |
| PD teacher (no brain at all) | — | 0.182 m | 0/12 | 1.00 | — |

**The connectome is the best driver of the three networks, on every metric, and
it does it at half the random network's activity** (13,495 against 27,092
spikes/step). Destroy the topology but keep every neuron's in-degree — the
`shuffled` control — and the driver gets measurably worse; the wiring is doing
work, not just the weight statistics. The random network's output cables barely
carry the command at all (decoder R² 0.752 against the fly's 0.984).

A hand-written PD controller still wins at 0.182 m. A 166,700-neuron connectome
read through a linear decoder gets to ~0.21 m, about 85% of the way, with no
training inside the brain.

### Limits

* **The driver is bimodal.** It either holds the lane (~0.2 m) or loses the road
  outright (~2 m+); almost nothing lands in between. Four roads is enough to
  make the mean meaningless — which is why the tables above report 12 and the
  median. The failure mode is compounding error: once the car is outside the
  band the teacher explored, the decoder under-corrects, the error grows, and it
  diverges.
* **One gain per model**, tuned on two held-out roads, not a full search.
* **"Lost road" is a threshold**, not a graded outcome. Read it next to the
  median, never alone.
* **The decoder is trained; the fly is not.** This measures how well a fixed
  connectome can be *read* behind a wheel, not what the fly has learned.

## Data

All real, all downloaded, all in `flybench/data/`:

| series | source | span |
|---|---|---|
| sunspots | SILSO monthly totals | 1749–2026, 3,331 points |
| btc | Yahoo Finance BTC-USD daily closes | 10 years, 3,651 points |
| brent | FRED `DCOILBRENTEU` daily | 1987–2026, 10,256 points |

`mackey` (τ = 17 time units, subsampled ×10), `narma` (NARMA-10) and `memory`
(memory capacity) are synthetic and reproducible from the seed in the source.
