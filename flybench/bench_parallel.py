import os, subprocess, sys, time

ROOT = os.path.expanduser("~/fly.ai")  # run: .venv/bin/python flybench/bench_parallel.py
ONE = os.path.join(ROOT, "flybench", "_one_brain.py")


def measure(n, threads):
    env = dict(os.environ, NUMBA_NUM_THREADS=str(threads))
    t0 = time.time()
    ps = [subprocess.Popen([sys.executable, ONE, f"b{i}"], cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
          for i in range(n)]
    out = [p.communicate()[0].strip() for p in ps]
    wall = time.time() - t0
    per = [float(o.split()[-2]) for o in out if o]
    print(f"{n} brains x {threads} threads: wall {wall:.1f}s for 200 steps "
          f"-> {wall*1000/200:.2f} ms/step wall, per-brain {sum(per)/len(per):.2f} ms/step, "
          f"{20/(wall*1000/200):.1f}x realtime budget")


for n, threads in [(1, 12), (4, 12), (4, 3), (8, 3)]:
    measure(n, threads)
