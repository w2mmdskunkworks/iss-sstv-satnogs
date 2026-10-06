"""
decoder.py - offline, noise-tolerant SSTV decoder for satellite recordings.

Pipeline per recording:
  1. resample to 16 kHz, band-pass 1.0-2.5 kHz
  2. per-sample SNR from FM quieting (in-band tones vs 3-7 kHz hiss)
  3. for each candidate mode: 1200 Hz sync-tone ratio -> comb correlation
     over the mode's full image (every sync pulse of the image, not just the
     VIS header) -> image start candidates; best-scoring mode wins overlaps
  4. VIS header decoded where readable (confirms mode; not required)
  5. per-line sync location + robust linear fit (no slant, no jitter)
  6. tuning offset from the sync tone
  7. phasor-sum discriminator: per pixel, sum z[n]*conj(z[n-1]) over the
     pixel's time slot and take one angle (amplitude-weighted ML-style
     frequency estimate; suppresses FM clicks). Pre-filter bandwidth scales
     with the mode's pixel rate.
  8. assemble channels per mode -> YCbCr image + per-row SNR

decode(path_or_audio, fs=None, modes=None) -> list[Decoded]
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import signal

from .modes import MODES, BY_VIS, ISS_COMMON, FREQ_BLACK, FREQ_WHITE

FS = 16000
FC = 1900.0          # demod centre: pure noise averages here -> neutral grey
# pre-filter cutoff = LP_BASE + LP_K * pixel rate. LP_BASE must pass the
# whole tone range (1500-2300 Hz = +/-400 Hz around FC); narrower biases dark
# and bright pixels toward FC. LP_K tuned on the simulator.
# Two renders, blended per row by measured SNR: weak rows want the narrowest
# filter that still passes the tone range (450 Hz; narrower biases tones),
# strong rows a wider one (LP_K * pixel rate extra) for sharper detail.
# Simulator: narrow wins by 3-4 dB at CNR 4, wide by 1-2 dB at CNR 9.
LP_BASE, LP_K, LP_MAX = 450.0, 0.05, 1400.0
# Blend criterion: pixel-frequency noise measured in the sync pulses with the
# WIDE filter (Hz rms; deviation-independent). Wide wins below the per-mode
# crossover (simulator calibration); ramp +/-25% around it.
BLEND_SIGMA = {   # Hz; calibrated with dev/calib.py (2026-10-05)
    "Robot36": 23.0, "PD120": 14.5, "PD90": 9.2, "PD180": 9.3, "Martin1": 20.0,
    # not calibrated - nearest sibling by pixel time / layout:
    "PD50": 9.3, "PD160": 9.2, "PD240": 9.2, "PD290": 9.3, "Martin2": 20.0,
}
BLEND_SIGMA_DEFAULT = 12.0
LP_SYNC = 1000.0     # wide filter for locating sync edges
OFFSET_LP = 250.0    # narrow 1200-centred filter for the tuning-offset estimate
SYNC_BIAS_S = 0.00027  # edge locator's constant lag (measured on simulator)


@dataclass
class Decoded:
    mode: str
    ycc: np.ndarray          # H x W x 3 uint8 (Y, Cb, Cr)
    valid: np.ndarray        # H bool: row inside the recording
    snr: np.ndarray          # H float dB (FM-quieting SNR per row)
    noise: np.ndarray        # H float: pixel noise, 0-255 units rms per row
    start_s: float           # time of first sync in the recording
    unit_s: float            # fitted line(unit) period
    offset_hz: float         # tuning offset removed
    score: float             # comb detection z-score
    syncs: int               # sync pulses located
    vis: int | None          # VIS code read (None if unreadable)

    @property
    def rgb(self):
        return Image.fromarray(self.ycc, "YCbCr").convert("RGB")


# ---------------------------------------------------------------- audio
def load(src, fs=None):
    """File path or (array, fs) -> (band-passed 16 kHz, raw 16 kHz)."""
    if isinstance(src, (str, bytes)) or hasattr(src, "__fspath__"):
        import soundfile as sf
        x, fs = sf.read(str(src), always_2d=True)
        x = x.mean(axis=1)
    else:
        x = np.asarray(src, float)
    g = np.gcd(int(fs), FS)
    raw = signal.resample_poly(x, FS // g, int(fs) // g)
    sos = signal.butter(6, [1000, 2500], btype="bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, raw), raw


def snr_envelope(raw):
    """dB: signal band 1.1-2.4 kHz vs FM hiss 3-7 kHz (per Hz), 20 ms smooth."""
    def bandpow(lo, hi):
        sos = signal.butter(6, [lo, hi], btype="bandpass", fs=FS, output="sos")
        y = signal.sosfiltfilt(sos, raw)
        k = int(0.02 * FS)
        return np.convolve(y * y, np.ones(k) / k, mode="same") / (hi - lo)
    return 10 * np.log10((bandpow(1100, 2400) + 1e-15) / (bandpow(3000, 7000) + 1e-15))


def _fir(cut, ntaps=129):
    return signal.firwin(ntaps, cut, fs=FS)


def phasors(x, lp_hz, fc=FC):
    """Lag-1 products of the complex baseband around fc (see module doc)."""
    n = np.arange(len(x))
    z = signal.oaconvolve(x * np.exp(-2j * np.pi * fc * n / FS), _fir(lp_hz), mode="same")
    p = z[1:] * np.conj(z[:-1])
    return np.concatenate([p[:1], p])


def _to_hz(v):
    return np.angle(v) * FS / (2 * np.pi) + FC


def tone_power(x, f, smooth_ms):
    n = np.arange(len(x))
    z = signal.oaconvolve(x * np.exp(-2j * np.pi * f * n / FS), _fir(120, 257), mode="same")
    k = max(1, int(smooth_ms * FS / 1000))
    return np.convolve(np.abs(z) ** 2, np.ones(k) / k, mode="same")


def sync_ratio(x, sync_ms):
    """Fraction of band power on 1200 Hz, smoothed over one sync pulse."""
    k = max(1, int(sync_ms * FS / 1000))
    pt = np.convolve(x * x, np.ones(k) / k, mode="same")
    floor = 0.01 * np.median(pt[pt > 0]) if np.any(pt > 0) else 1e-12
    return tone_power(x, 1200.0, sync_ms) / (pt + floor)    # silence -> 0, not 0/0


# ---------------------------------------------------------------- detection
def comb_scores(r, mode):
    """z-score vs time for an image of `mode` starting at each ms."""
    step = FS // 1000
    c = r[: len(r) // step * step].reshape(-1, step).mean(axis=1)
    c = c - np.median(c)
    teeth = np.round(np.arange(mode.units) * mode.unit_ms).astype(int)
    comb = np.zeros(teeth[-1] + 1)
    comb[teeth] = 1.0
    pad = len(comb)
    cp = np.concatenate([np.zeros(pad), c, np.zeros(pad)])
    score = signal.fftconvolve(cp, comb[::-1], mode="valid")   # t0 = i - pad (ms)
    med = np.median(score)
    mad = np.median(np.abs(score - med)) * 1.4826 + 1e-12
    return (score - med) / mad, pad


def find_candidates(r, mode, dur_ms, min_frac=0.25, zmin=8.0, max_images=12):
    z, pad = comb_scores(r, mode)
    out, zz = [], z.copy()
    span = int(mode.unit_ms * mode.units)
    for _ in range(max_images):
        i = int(np.argmax(zz))
        if zz[i] < zmin:
            break
        t0 = (i - pad) / 1000.0 - mode.sync_ms / 2000.0   # ratio peaks mid-pulse
        lo, hi = max(0, i - span + int(mode.unit_ms)), i + span - int(mode.unit_ms)
        zz[lo:hi] = -np.inf
        inside = np.sum((t0 + np.arange(mode.units) * mode.unit_ms / 1000 >= 0) &
                        (t0 + np.arange(mode.units) * mode.unit_ms / 1000 < dur_ms / 1000 - mode.unit_ms / 1000))
        if inside >= min_frac * mode.units:
            out.append((t0, float(z[i])))
    return out


def read_vis(x, t_img):
    """Try to read the VIS code ending near t_img (s). Returns code or None.
    VIS: 1900 leader | start bit 1200 | 7 data bits LSB first (1100=1,
    1300=0) | even parity | stop bit 1200, 30 ms each."""
    p11 = tone_power(x, 1100.0, 30)
    p13 = tone_power(x, 1300.0, 30)
    p12 = tone_power(x, 1200.0, 30)
    p19 = tone_power(x, 1900.0, 30)
    best = None
    for end_ms in np.arange(-450, 60, 2.0):          # stop bit end vs image start
        e = t_img + end_ms / 1000.0
        bit_c = [e - (10 - i) * 0.030 + 0.015 for i in range(10)]   # centres
        idx = [int(c * FS) for c in bit_c]
        if idx[0] - int(0.1 * FS) < 0 or idx[-1] >= len(x):
            continue
        lead = p19[idx[0] - int(0.1 * FS)]
        frame = p12[idx[0]] + p12[idx[9]] + lead
        if best is None or frame > best[0]:
            bits = [1 if p11[i] > p13[i] else 0 for i in idx[1:9]]
            best = (frame, bits)
    if best is None:
        return None
    bits = best[1]
    code = sum(b << i for i, b in enumerate(bits[:7]))
    if sum(bits[:7]) % 2 != bits[7]:
        return None
    return code if code in BY_VIS else None


def sync_edges(psync, mode):
    """Matched-template sync locator. Soft 'is sync tone' indicator from the
    wide-filter discriminator, correlated with +1 over the pulse and -1 over
    the following porch. Keys on the TRAILING 1200->1500 Hz edge, which is
    identical every line (the leading edge depends on the previous pixel).
    Returns score per sample: score[i] = template aligned with pulse start i."""
    k = max(1, int(0.0008 * FS))                 # 0.8 ms phasor smoothing
    g = np.convolve(psync, np.ones(k) / k, mode="same")
    f = _to_hz(g)
    ind = np.clip((1450.0 - f) / 250.0, 0.0, 1.0) * 2 - 1     # +1 sync, -1 other
    pre = int(0.002 * FS)
    syn = int(mode.sync_ms / 1000 * FS)
    post = max(int(0.0005 * FS), int(min(0.002, 0.25 * mode.sync_ms / 1000) * FS))
    tmpl = np.concatenate([np.zeros(pre), np.ones(syn), -np.ones(post)])
    tmpl /= len(tmpl)
    c = np.correlate(ind, tmpl, mode="full")[len(tmpl) - 1:]   # c[i]: tmpl starts at i
    return np.concatenate([np.full(pre, -1.0), c])[: len(ind)]  # index = pulse start


def refine(edge, mode, t0, win_ms=None):
    """Locate each unit's sync start near prediction; robust fit t = a + b*k."""
    win = int((win_ms or max(6.0, 0.6 * mode.sync_ms)) * FS / 1000)
    ks, ts, ws = [], [], []
    for k in range(mode.units):
        c = int(round((t0 + k * mode.unit_ms / 1000) * FS))
        lo, hi = c - win, c + win
        if lo < 0 or hi >= len(edge):
            continue
        seg = edge[lo:hi]
        j = int(np.argmax(seg))
        if seg[j] < 0.35:
            continue
        # parabolic sub-sample peak
        dj = 0.0
        if 0 < j < len(seg) - 1:
            y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
            den = y0 - 2 * y1 + y2
            dj = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        ks.append(k)
        ts.append((lo + j + dj) / FS)
        ws.append(seg[j])
    unit = mode.unit_ms / 1000
    if len(ks) < max(10, mode.units // 10):
        return t0, unit, len(ks)
    ks, ts, ws = map(np.asarray, (ks, ts, ws))
    keep = np.ones(len(ks), bool)
    a, b = t0, unit
    for _ in range(5):
        A = np.vstack([np.ones(keep.sum()), ks[keep]]).T
        a, b = np.linalg.lstsq(A * ws[keep, None], ts[keep] * ws[keep], rcond=None)[0]
        res = ts - (a + b * ks)
        keep = np.abs(res) < max(0.0003, 3 * np.std(res[keep]))
    if abs(b / unit - 1) > 0.002:                 # implausible clock error
        b = unit
        a = float(np.median(ts - b * ks))
    return float(a), float(b), int(keep.sum())


def fix_line_slip(edge, mode, a, b):
    """The comb can lock one unit early/late (e.g. VIS stop bit, 1200 Hz,
    sits right before line 0). Compare sync evidence at the two ends."""
    def ev(k):
        c = int(round((a + b * k) * FS))
        w = int(0.001 * FS)
        if c - w < 0 or c + w >= len(edge):
            return -1.0
        return float(np.max(edge[c - w:c + w]))
    for _ in range(2):
        n = mode.units
        if ev(n) > ev(0) + 0.3:          # real sync after our last line
            a += b
        elif ev(-1) > ev(n - 1) + 0.3:   # real sync before our first line
            a -= b
        else:
            break
    return a


# ---------------------------------------------------------------- extraction
def _slots(cs, t_start, t_len, n_px):
    n = len(cs) - 1
    edges = (t_start + np.arange(n_px + 1) * t_len / n_px) * FS
    e = np.clip(edges, 0, n)
    i = np.minimum(e.astype(np.int64), n - 1)
    v = cs[i] + (cs[i + 1] - cs[i]) * (e - i)
    out = np.diff(v)
    bad = (edges[1:] <= 0) | (edges[:-1] >= n)
    return out, bad


def render(p, env, mode, a, b, offset):
    """Phasor sums -> image. Returns ycc (H,W,3 uint8), valid rows, snr rows."""
    cs = np.concatenate([[0], np.cumsum(p)])
    H, W = mode.height, mode.width
    chans = {c: np.full((H, W), np.nan) for c in ("Y", "Cb", "Cr", "R", "G", "B")}
    valid = np.zeros(H, bool)
    snr = np.full(H, np.nan)
    robot_c = np.full((H, W), np.nan)
    parity = np.zeros(H)

    def val(t_start, dur, npx):
        s, bad = _slots(cs, t_start, dur, npx)
        v = (_to_hz(s) - offset - FREQ_BLACK) / (FREQ_WHITE - FREQ_BLACK) * 255.0
        v[bad] = np.nan
        return v

    for k in range(mode.units):
        ts = a + b * k
        rows = [k * mode.rows + r for r in range(mode.rows)]
        span_lo = ts + min(c.t0 for c in mode.comps) / 1000
        span_hi = ts + max(c.t0 + c.dur for c in mode.comps) / 1000
        if span_hi <= 0 or span_lo * FS >= len(p):
            continue
        lo, hi = int(max(span_lo, 0) * FS), int(min(span_hi * FS, len(env)))
        s = float(np.median(env[lo:hi])) if hi > lo else np.nan
        for comp in mode.comps:
            v = val(ts + comp.t0 / 1000, comp.dur / 1000, comp.px)
            if comp.px != W:
                v = np.repeat(v, W // comp.px)
            targets = rows if comp.row < 0 else [rows[comp.row]]
            for r in targets:
                if comp.chan == "C":
                    robot_c[r] = v
                else:
                    chans[comp.chan][r] = v
        for r in rows:
            valid[r] = True
            snr[r] = s
        if mode.name == "Robot36":
            sep = _to_hz(_slots(cs, ts + mode.sep_ms / 1000 + 0.0005, 0.0035, 1)[0])[0] - offset
            parity[rows[0]] = 1.0 if sep > 1900 else -1.0

    if mode.name == "Robot36":
        vote = sum(parity[k] * (1 if k % 2 else -1) for k in range(H) if valid[k])
        odd_is_cb = vote >= 0
        for k in range(0, H - 1, 2):
            cr_l, cb_l = (k, k + 1) if odd_is_cb else (k + 1, k)
            chans["Cr"][k] = chans["Cr"][k + 1] = robot_c[cr_l]
            chans["Cb"][k] = chans["Cb"][k + 1] = robot_c[cb_l]

    if mode.color == "RGB":
        rgb = np.stack([chans["R"], chans["G"], chans["B"]], -1)
        rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 255).astype(np.uint8)
        ycc = np.asarray(Image.fromarray(rgb, "RGB").convert("YCbCr"))
        ycc = ycc.copy()
        ycc[~valid] = (0, 128, 128)
    else:
        ycc = np.stack([chans["Y"], chans["Cb"], chans["Cr"]], -1)
        fill = np.array([0.0, 128.0, 128.0])
        ycc = np.where(np.isnan(ycc), fill, ycc)
        ycc = np.clip(ycc, 0, 255).astype(np.uint8)
    return ycc, valid, snr


def sync_noise(p, mode, a, b):
    """Per-row pixel-frequency noise (Hz rms), measured where the true
    frequency is known: inside each sync pulse (constant 1200 Hz), in
    pixel-sized slots. `p` must have the SAME bandwidth as the render stream
    but be centred on 1200 Hz (phasors(x, lp, fc=1200)): discriminator noise
    depends on bandwidth and signal level, not on the centre, and a
    1900-centred narrow filter would not pass the sync tone at all.
    Deviation-independent, unlike the hiss SNR."""
    cs = np.concatenate([[0], np.cumsum(p)])
    px = min(c.dur / c.px for c in mode.comps) / 1000
    n = max(4, int((mode.sync_ms / 1000 - 0.002) / px))
    out = np.full(mode.units, np.nan)
    for k in range(mode.units):
        ts = a + b * k + 0.001
        s, bad = _slots(cs, ts, n * px, n)
        if not bad.any():
            f = np.angle(s) * FS / (2 * np.pi)          # Hz relative to 1200
            out[k] = np.sqrt(np.mean((f - np.median(f)) ** 2))
    # smooth over neighbouring units (few slots per pulse -> noisy estimate)
    ok = ~np.isnan(out)
    if ok.sum() > 3:
        sm = np.convolve(np.where(ok, out, 0), np.ones(5), "same") / \
             np.maximum(np.convolve(ok.astype(float), np.ones(5), "same"), 1)
        out = np.where(ok | (sm > 0), sm, np.nan)
    return np.repeat(out, mode.rows)[: mode.height]


def lp_for(mode, k=None):
    px_ms = min(c.dur / c.px for c in mode.comps)
    return float(min(LP_BASE + (LP_K if k is None else k) * 1000.0 / px_ms, LP_MAX))


# ---------------------------------------------------------------- top level
def decode(src, fs=None, modes=None, zmin=8.0, lp_k=None, read_header=True,
           blend=True):
    """Decode every SSTV image in a recording. modes: names to try
    (default: modes the ISS has used). Returns list[Decoded], time-ordered."""
    x, raw = load(src, fs)
    env = snr_envelope(raw)
    dur_ms = len(x) / FS * 1000
    names = list(modes) if modes else list(ISS_COMMON)
    ratio_cache = {}
    cands = []
    for name in names:
        m = MODES[name]
        if m.sync_ms not in ratio_cache:
            ratio_cache[m.sync_ms] = sync_ratio(x, m.sync_ms)
        for t0, z in find_candidates(ratio_cache[m.sync_ms], m, dur_ms, zmin=zmin):
            cands.append((z, name, t0))
    # strongest first; drop weaker candidates overlapping a chosen image
    cands.sort(reverse=True)
    chosen = []
    for z, name, t0 in cands:
        m = MODES[name]
        t1 = t0 + m.duration_s
        if any(t0 < c1 - 1.0 and t1 > c0 + 1.0 for _, _, c0, c1 in chosen):
            continue
        chosen.append((z, name, t0, t1))

    out = []
    p_cache, edge_cache, p1200 = {}, {}, {}

    def _ps1200(lp_hz):
        if lp_hz not in p1200:
            p1200[lp_hz] = phasors(x, lp_hz, fc=1200.0)
        return p1200[lp_hz]
    for z, name, t0, _ in sorted(chosen, key=lambda c: c[2]):
        m = MODES[name]
        lp = lp_for(m, lp_k)
        lp_n = LP_BASE
        for f in (lp, lp_n, LP_SYNC):
            if f not in p_cache:
                p_cache[f] = phasors(x, f)
        p, psync = p_cache[lp], p_cache[LP_SYNC]
        if m.sync_ms not in edge_cache:
            edge_cache[m.sync_ms] = sync_edges(psync, m)
        a, b, nsync = refine(edge_cache[m.sync_ms], m, t0)
        a = fix_line_slip(edge_cache[m.sync_ms], m, a, b) - SYNC_BIAS_S
        # tuning offset from the located sync tones, measured with a narrow
        # stream CENTRED ON 1200 Hz: noise then pulls the estimate toward 0
        # (harmless). Measuring through the 1900-centred filter, the sync tone
        # sits in the roll-off and in-band noise dragged the estimate up by
        # 0-150 Hz on real weak passes (-> green images).
        p12 = _ps1200(OFFSET_LP)
        acc = 0j
        for k in range(m.units):
            s = int((a + b * k + 0.15 * m.sync_ms / 1000) * FS)
            e = int((a + b * k + 0.85 * m.sync_ms / 1000) * FS)
            if 0 <= s < e < len(p12):
                acc += p12[s:e].sum()
        offset = float(np.angle(acc) * FS / (2 * np.pi)) if acc != 0 else 0.0
        if abs(offset) > 120 or m.sync_ms < 8.0:
            # short syncs (Martin, 4.9 ms) are shorter than any filter that
            # separates 1200 Hz from the adjacent 1500 Hz gap -> biased
            # (+14 Hz in sim); real offsets are a few Hz, so leave at 0
            offset = 0.0
        ycc, valid, snr = render(p, env, m, a, b, offset)
        sig_w = sync_noise(_ps1200(lp), m, a, b)
        sig = sig_w
        if blend and lp_k is None and lp_n != lp:
            ycc_n, _, _ = render(p_cache[lp_n], env, m, a, b, offset)
            sig_n = sync_noise(_ps1200(lp_n), m, a, b)
            T = BLEND_SIGMA.get(name, BLEND_SIGMA_DEFAULT)
            w = np.clip((1.25 * T - np.nan_to_num(sig_w, nan=1e4)) / (0.5 * T), 0, 1)
            ycc = np.clip(w[:, None, None] * ycc + (1 - w[:, None, None]) * ycc_n.astype(float),
                          0, 255).astype(np.uint8)
            sig = w * sig_w + (1 - w) * sig_n
        noise = sig * 255.0 / (FREQ_WHITE - FREQ_BLACK)
        vis = read_vis(x, a) if read_header else None
        out.append(Decoded(name, ycc, valid, snr, noise, a, b, offset, z, nsync, vis))
    return out
