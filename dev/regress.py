"""Regression checks for sstvdec, against the simulator's ground truth.

    venv/bin/python dev/regress.py          # ~10 min on a Pi 5

For each mode family and signal level it checks: the right mode is detected,
line timing is within 0.1 ms, the tuning-offset estimate is near zero, and
image PSNR clears a floor. Run it after any decoder change; the floors are
set ~1.5 dB under what the decoder achieved on 2026-10-05.
"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from sstvdec import sim, decoder as D

DEV = os.path.join(ROOT, 'dev') + os.sep
IMG = Image.open(DEV + 'test_photo.png')
PAD_HDR = 3.0 + 0.910          # simulator padding + VIS header
# (mode, cnr_db, psnr_floor)
CASES = [("Robot36", 20, 27.0), ("Robot36", 6, 20.5), ("Robot36", 3, 13.0),
         ("PD120", 20, 30.0), ("PD120", 6, 21.5), ("PD90", 20, 30.0),
         ("PD180", 8, 26.0), ("Martin1", 20, 29.0), ("Martin1", 5, 21.0)]

fails = 0
for mode, cnr, floor in CASES:
    a, truth = sim.transmission(IMG, mode, cnr, seed=5, dev_hz=4000)     # with OGG
    res = D.decode(a, 48000)                                             # auto mode
    if not res:
        print(f"FAIL {mode} cnr{cnr}: nothing detected"); fails += 1; continue
    d = max(res, key=lambda r: r.score)
    psnr = sim.psnr(np.asarray(d.rgb), np.asarray(truth))
    terr = (d.start_s - PAD_HDR) * 1000
    probs = []
    if d.mode != mode:
        probs.append(f"mode {d.mode}")
    if abs(terr) > 0.1:
        probs.append(f"timing {terr:+.3f} ms")
    if abs(d.offset_hz) > 5:
        probs.append(f"offset {d.offset_hz:+.1f} Hz")
    if psnr < floor:
        probs.append(f"PSNR {psnr:.1f} < {floor}")
    status = "FAIL" if probs else "ok  "
    fails += bool(probs)
    print(f"{status} {mode:8s} cnr{cnr:2d}: detected {d.mode:8s} vis={d.vis} "
          f"timing {terr:+.3f} ms offset {d.offset_hz:+5.1f} Hz PSNR {psnr:5.1f} "
          f"{'; '.join(probs)}", flush=True)
print(f"\n{len(CASES) - fails}/{len(CASES)} passed")
sys.exit(1 if fails else 0)
