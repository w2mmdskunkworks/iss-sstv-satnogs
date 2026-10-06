"""Narrow-vs-wide filter crossover per mode, vs sync-measured noise (Hz).
usage: calib.py MODE [MODE ...]  -> dev/calib_<modes>.json + log lines
row = (cnr_db, psnr_narrow, psnr_wide, sigma_wide_hz)"""
import sys, json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from sstvdec import sim, decoder as D
from sstvdec.modes import MODES

DEV = os.path.join(ROOT, 'dev') + os.sep
imgs = [Image.open(DEV + 'test_photo.png'), Image.open(DEV + 'test_text.png')]
res = {}
for mode in sys.argv[1:]:
    m = MODES[mode]
    rows = []
    for cnr in (3, 4, 5, 6, 7, 8, 9, 11, 14):
        pn, pw, sg = [], [], []
        for im in imgs:
            a, tr = sim.transmission(im, mode, cnr, seed=13, ogg=False, dev_hz=4000)
            rn = D.decode(a, 48000, modes=[mode], read_header=False, lp_k=0.0)
            rw = D.decode(a, 48000, modes=[mode], read_header=False, lp_k=0.05)
            if not rn or not rw:
                continue
            rn, rw = rn[0], rw[0]
            pn.append(sim.psnr(np.asarray(rn.rgb), np.asarray(tr)))
            pw.append(sim.psnr(np.asarray(rw.rgb), np.asarray(tr)))
            x, raw = D.load(a, 48000)
            p = D.phasors(x, D.lp_for(m, 0.05), fc=1200.0)
            sg.append(np.nanmedian(D.sync_noise(p, m, rw.start_s, rw.unit_s)))
        if pn:
            rows.append((cnr, float(np.mean(pn)), float(np.mean(pw)), float(np.mean(sg))))
            print(mode, rows[-1], flush=True)
    res[mode] = rows
    json.dump(res, open(DEV + f'calib_{"_".join(sys.argv[1:])}.json', 'w'))
