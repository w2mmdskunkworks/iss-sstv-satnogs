"""
sim.py - ground-truth test bench: SSTV encode -> NBFM radio channel with
noise -> FM discriminator -> (optional OGG round trip), i.e. what a SatNOGS
station records. Lets every decoder/stacker change be scored against the
exact transmitted image instead of against other noisy copies.

    audio, truth = transmission(img, "PD120", cnr_db=10)
"""

import numpy as np
from PIL import Image
from scipy import signal

FS = 48000


def encode(img, mode_name, fs=FS):
    """pysstv encode (independent reference implementation) -> float audio."""
    import pysstv.color as c
    cls = {"Robot36": c.Robot36, "PD90": c.PD90, "PD120": c.PD120,
           "PD160": c.PD160, "PD180": c.PD180, "PD240": c.PD240,
           "PD290": c.PD290, "Martin1": c.MartinM1, "Martin2": c.MartinM2,
           "Scottie1": c.ScottieS1, "Scottie2": c.ScottieS2,
           "ScottieDX": c.ScottieDX}[mode_name]
    im = img.convert("RGB").resize((cls.WIDTH, cls.HEIGHT), Image.LANCZOS)
    s = cls(im, fs, 16)
    s.vox_enabled = False
    a = np.fromiter(s.gen_samples(), dtype=np.float64)
    return a / 32767.0, im


def fm_channel(audio, cnr_db, fs=FS, dev_hz=3000.0, chan_bw=16000.0,
               rate_err=0.0, ogg=True, seed=None, return_iq=False):
    """Audio -> NBFM -> complex AWGN (CNR measured in chan_bw) -> channel
    filter -> quadrature discriminator -> audio, like gr-satnogs' FM chain.
    cnr_db may be a scalar or an array (per-sample, for pass profiles).
    rate_err: transmitter/receiver clock mismatch (slant + tone offset)."""
    rng = np.random.default_rng(seed)
    if rate_err:
        n = len(audio)
        audio = np.interp(np.arange(0, n, 1 + rate_err), np.arange(n), audio)
    a = audio / (np.max(np.abs(audio)) + 1e-12)
    phase = 2 * np.pi * dev_hz * np.cumsum(a) / fs
    x = np.exp(1j * phase)
    cnr = np.broadcast_to(np.asarray(cnr_db, float), x.shape)
    # noise power over full fs band such that power inside chan_bw = 1/CNR
    sigma2 = (fs / chan_bw) * 10 ** (-cnr / 10)
    noise = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    x = x + noise * np.sqrt(sigma2 / 2)
    if return_iq:                       # what a SatNOGS IQ dump would hold
        return x
    h = signal.firwin(129, chan_bw / 2, fs=fs)
    y = signal.oaconvolve(x, h, mode="same")
    d = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi * dev_hz)
    d = np.concatenate([d[:1], d])
    d = signal.sosfiltfilt(signal.butter(4, 10000, fs=fs, output="sos"), d)
    d = 0.5 * d / (np.std(d) * 3 + 1e-12)
    if ogg:
        d = ogg_roundtrip(d, fs)
    return d


def ogg_roundtrip(d, fs=FS, quality=10):
    """Vorbis encode/decode like SatNOGS uploads (gr-satnogs ogg_encoder at
    quality 1.0 == libvorbis -q 10; ~110 kbps on real station-223 files).
    Uses ffmpeg/libvorbis: the venv's soundfile can't set Vorbis quality."""
    import subprocess
    pcm = np.clip(d, -1, 1).astype("<f4").tobytes()
    ogg = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(fs),
                          "-ac", "1", "-i", "-", "-c:a", "libvorbis",
                          "-q:a", str(quality), "-f", "ogg", "-"],
                         input=pcm, capture_output=True, check=True).stdout
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "f32le",
                          "-ac", "1", "-ar", str(fs), "-"],
                         input=ogg, capture_output=True, check=True).stdout
    out = np.frombuffer(raw, "<f4").astype(float)
    ogg_roundtrip.last_kbps = len(ogg) * 8 / (len(d) / fs) / 1000
    return out[: len(d)]


def transmission(img, mode_name, cnr_db, pad_s=(3.0, 3.0), **kw):
    """One image with silence-noise padding. Returns (audio, truth RGB)."""
    a, truth = encode(img, mode_name)
    a = np.concatenate([np.zeros(int(pad_s[0] * FS)), a, np.zeros(int(pad_s[1] * FS))])
    return fm_channel(a, cnr_db, **kw), truth


def pass_audio(imgs, mode_name, gap_s, cnr_profile, **kw):
    """A pass: several transmissions separated by gap_s of dead air (noise).
    cnr_profile(t_seconds_array) -> dB, e.g. a low-high-low elevation curve.
    Returns (audio, [(start_s, truth), ...])."""
    parts, truths, t = [np.zeros(int(5 * FS))], [], 5.0
    for im in imgs:
        a, tr = encode(im, mode_name)
        truths.append((t, tr))
        parts += [a, np.zeros(int(gap_s * FS))]
        t += len(a) / FS + gap_s
    audio = np.concatenate(parts)
    cnr = cnr_profile(np.arange(len(audio)) / FS)
    return fm_channel(audio, cnr, **kw), truths


def iq_demod(iq, fs=FS, dev_hz=3000.0, chan_bw=16000.0, method="quad", smooth=1):
    """FM demod of IQ. method="quad": standard discriminator (SatNOGS).
    method="phasor": amplitude-weighted discriminator - sums y[n]*conj(y[n-1])
    over `smooth` samples before taking the angle, so the low-amplitude
    samples where FM clicks originate carry little weight."""
    h = signal.firwin(129, chan_bw / 2, fs=fs)
    y = signal.oaconvolve(iq, h, mode="same")
    p = y[1:] * np.conj(y[:-1])
    if method == "phasor" and smooth > 1:
        p = np.convolve(p, np.ones(smooth) / smooth, mode="same")
    d = np.angle(p) * fs / (2 * np.pi * dev_hz)
    d = np.concatenate([d[:1], d])
    d = signal.sosfiltfilt(signal.butter(4, 10000, fs=fs, output="sos"), d)
    return 0.5 * d / (np.std(d) * 3 + 1e-12)


def psnr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
