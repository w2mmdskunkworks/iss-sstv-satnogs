import sys, itertools, pickle
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'dev'))
import numpy as np
from eval_stack import build, rgb, m_wmedian, m_bestrow, MODE
from sstvdec import sim, stack as S

def wmed(Y, W):
    order = np.argsort(Y, 0); Ys = np.take_along_axis(Y, order, 0)
    Ws = np.take_along_axis(np.broadcast_to(W, Y.shape), order, 0)
    cw = np.cumsum(Ws, 0); k = (cw >= cw[-1] / 2).argmax(0)
    return np.take_along_axis(Ys, k[None], 0)[0]

def stackarr(cs):
    Y = np.stack([c['ycc'].astype(float) for c in cs])
    N = np.stack([np.nan_to_num(c['noise'], nan=1e9) for c in cs])
    return Y, np.maximum(N, 1.0)

def wmed_p(p):
    def f(cs):
        Y, N = stackarr(cs); W = (1 / N ** p)[:, :, None, None]
        return wmed(Y, W)
    return f

def select(k, how):
    def f(cs):
        Y, N = stackarr(cs)
        ok = N <= k * N.min(0)                    # copies near the best row
        W = (ok / N ** 2)[:, :, None, None]
        if how == 'median':
            return wmed(Y, W)
        return (W * Y).sum(0) / W.sum(0)
    return f

M = {'wmedian(p2)': m_wmedian, 'bestrow': m_bestrow, 'wmed p4': wmed_p(4), 'wmed p8': wmed_p(8),
     'sel1.3 wmed': select(1.3, 'median'), 'sel1.5 wmed': select(1.5, 'median'),
     'sel1.5 wmean': select(1.5, 'mean'), 'sel2 wmed': select(2.0, 'median')}
data = build()
for k in (2, 3, 4):
    line = f'{k} copies: '
    for name, fn in M.items():
        ps = [sim.psnr(rgb(fn(list(sub))), trgb) for _, _, trgb, cs in data
              for sub in itertools.combinations(cs, k)]
        line += f'{name} {np.mean(ps):5.2f} | '
    print(line, flush=True)
