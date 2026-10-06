"""Merge + cleanup strategies scored against ground truth (simulator).
Stage 1 (slow, cached): K noisy copies per test image with fading profiles.
Stage 2: merge strategies x copy counts, cleanup methods -> PSNR table."""
import sys, os, pickle, itertools
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image
from scipy import ndimage
from sstvdec import sim, decoder as D, stack as S

DEV = os.path.join(ROOT, 'dev') + os.sep
MODE = sys.argv[1] if len(sys.argv) > 1 else 'Robot36'
CACHE = DEV + f'eval_copies_{MODE}.pkl'


def profile(base, rng):
    ph = rng.uniform(0, 2 * np.pi)
    fade_t = rng.uniform(5, 35)
    def f(t):
        c = base + 3 * np.sin(2 * np.pi * t / 25 + ph)
        return c - 8 * np.exp(-((t - fade_t) / 1.5) ** 2)
    return f


def build():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, 'rb'))
    data = []
    for k, name in enumerate(('test_photo.png', 'test_text.png')):
        im = Image.open(DEV + name)
        rng = np.random.default_rng(100 + k)
        a0, truth = sim.encode(im, MODE)
        a0 = np.concatenate([np.zeros(3 * 48000), a0, np.zeros(3 * 48000)])
        copies = []
        for j, base in enumerate((3.0, 4.5, 6.0, 8.0)):
            prof = profile(base, rng)
            cnr = prof(np.arange(len(a0)) / 48000 - 3.0)
            au = sim.fm_channel(a0, cnr, seed=1000 * k + j, ogg=False, dev_hz=4000)
            r = D.decode(au, 48000, modes=[MODE], read_header=False)
            if r:
                d = r[0]
                copies.append({'ycc': d.ycc, 'noise': d.noise, 'valid': d.valid,
                               'base': base})
                print(name, base, 'psnr', round(sim.psnr(np.asarray(d.rgb), np.asarray(truth)), 2),
                      flush=True)
        data.append((name, np.asarray(truth.convert('YCbCr')), np.asarray(truth), copies))
    pickle.dump(data, open(CACHE, 'wb'))
    return data


def rgb(ycc):
    return np.asarray(Image.fromarray(np.clip(ycc, 0, 255).astype(np.uint8), 'YCbCr').convert('RGB'))


# ---------------- merge strategies: list of copies -> ycc float
def m_wmedian(cs):
    return S.merge(cs, [(i, 0, 0) for i in range(len(cs))])[0].astype(float)


def m_wmean(cs):
    num = 0; den = 0
    for c in cs:
        w = S.inv_var(c)[:, None, None]
        num = num + w * c['ycc'].astype(float); den = den + w
    return np.where(den > 0, num / np.maximum(den, 1e-12), 128.0)


def m_median(cs):
    return np.median(np.stack([c['ycc'].astype(float) for c in cs]), 0)


def m_bestrow(cs):
    N = np.stack([np.nan_to_num(c['noise'], nan=1e9) for c in cs])
    best = N.argmin(0)
    Y = np.stack([c['ycc'].astype(float) for c in cs])
    return Y[best, np.arange(Y.shape[1])]


def m_hybrid(cs):
    """weighted median, but rows where one copy has >4x the inverse variance
    of all others combined take that copy directly"""
    out = m_wmedian(cs)
    iv = np.stack([S.inv_var(c) for c in cs])
    tot = iv.sum(0)
    top = iv.argmax(0)
    dom = iv.max(0) > 0.8 * tot
    Y = np.stack([c['ycc'].astype(float) for c in cs])
    out[dom] = Y[top[dom], np.flatnonzero(dom)]
    return out


MERGES = {'wmedian': m_wmedian, 'wmean': m_wmean, 'median': m_median,
          'bestrow': m_bestrow, 'hybrid': m_hybrid}


# ---------------- cleanup methods: (ycc float, noise rows) -> ycc float
def c_none(y, n):
    return y


def c_current(y, n):
    return S.denoise(np.clip(y, 0, 255).astype(np.uint8), n).astype(float)


def _chroma(img, n):
    q = S.row_quality(n)
    sat = ndimage.gaussian_filter1d(np.clip(q * 1.3, 0, 1), 2.0)[:, None]
    for c in (1, 2):
        ch = ndimage.gaussian_filter(ndimage.median_filter(img[..., c], size=7), 1.5)
        img[..., c] = 128.0 + (ch - 128.0) * sat
    return img


def c_nlm(y, n, k=1.0):
    import cv2
    img = y.copy()
    h = float(np.clip(k * np.nanmedian(n), 3, 40))
    Y = ndimage.median_filter(img[..., 0], size=3)
    img[..., 0] = cv2.fastNlMeansDenoising(np.clip(Y, 0, 255).astype(np.uint8), None, h, 7, 21)
    return _chroma(img, n)


def c_wavelet(y, n, k=1.0):
    import pywt
    img = y.copy()
    Y = ndimage.median_filter(img[..., 0], size=3)
    coeffs = pywt.wavedec2(Y, 'db2', level=3)
    sig = np.median(np.abs(coeffs[-1][2])) / 0.6745
    out = [coeffs[0]]
    for det in coeffs[1:]:
        nd = []
        for d in det:
            sx = max(np.var(d) - sig ** 2, 1e-6) ** 0.5
            t = k * sig ** 2 / sx                    # BayesShrink
            nd.append(pywt.threshold(d, t, 'soft'))
        out.append(tuple(nd))
    img[..., 0] = pywt.waverec2(out, 'db2')[:Y.shape[0], :Y.shape[1]]
    return _chroma(img, n)


def c_med3_adapt(y, n):
    img = y.copy()
    q = S.row_quality(n)[:, None]
    m3 = ndimage.median_filter(img[..., 0], size=3)
    m5 = ndimage.median_filter(img[..., 0], size=5)
    img[..., 0] = q * m3 + (1 - q) * m5
    return _chroma(img, n)


CLEAN = {'none': c_none, 'current': c_current, 'med3/5 adapt': c_med3_adapt,
         'nlm k.7': lambda y, n: c_nlm(y, n, 0.7), 'nlm k1.2': lambda y, n: c_nlm(y, n, 1.2),
         'wavelet': c_wavelet, 'wavelet k2': lambda y, n: c_wavelet(y, n, 2.0)}


def main():
    data = build()
    print('\n== merge strategies (no cleanup), PSNR vs truth, mean over images & subsets')
    for k in (1, 2, 3, 4):
        line = f'{k} cop{"y" if k == 1 else "ies"}: '
        for name, fn in MERGES.items():
            ps = []
            for _, tycc, trgb, cs in data:
                for sub in itertools.combinations(cs, k):
                    ps.append(sim.psnr(rgb(fn(list(sub))), trgb))
            line += f'{name} {np.mean(ps):5.2f}  '
        print(line, flush=True)
    print('\n== cleanup methods on single copies and on 4-copy wmedian merges')
    for name, fn in CLEAN.items():
        p1, p4 = [], []
        for _, tycc, trgb, cs in data:
            for c in cs:
                p1.append(sim.psnr(rgb(fn(c['ycc'].astype(float), c['noise'])), trgb))
            y, n = S.merge(cs, [(i, 0, 0) for i in range(len(cs))])
            p4.append(sim.psnr(rgb(fn(y.astype(float), n)), trgb))
        print(f'{name:14s} single {np.mean(p1):5.2f}   merged4 {np.mean(p4):5.2f}', flush=True)


if __name__ == '__main__':
    main()
