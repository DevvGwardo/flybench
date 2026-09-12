import time, sys
from fly_brain import FlyBrain
b = FlyBrain(device='cpu'); b.reset(64)
for _ in range(60): b.step()
t0 = time.time()
for _ in range(200): b.step()
print(f"{sys.argv[1]} {(time.time()-t0)/200*1000:.2f} ms/step")
