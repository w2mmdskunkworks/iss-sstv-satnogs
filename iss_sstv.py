#!/usr/bin/env python3
"""
iss_sstv.py - fetch ISS SSTV recordings from SatNOGS (or decode local files),
decode every image with a noise-tolerant multi-mode decoder, and merge the
repeat transmissions of each slide into the cleanest possible picture.

Typical use for an ARISS / ISS SSTV event:

    python iss_sstv.py --start 2026-10-02T09:00 --end 2026-10-06T16:00

    python iss_sstv.py --files rec1.ogg rec2.wav      # local recordings
    python iss_sstv.py --files iq_15110080.raw        # SatNOGS IQ dump
    python iss_sstv.py --station 1234 --freq-mhz 145.800 --days 3
    SATNOGS_STATION=1234 python iss_sstv.py ...       # or set your station once
    python iss_sstv.py --stations all --start ...     # whole SatNOGS network
                                                       # (NOT your own reception)

Decoder (sstvdec/decoder.py):
  * finds every image in a recording by correlating the full sync-pulse comb
    of each candidate mode (Robot36, PD50-290, Martin) - no VIS needed;
    reads the VIS header where possible as confirmation
  * line timing from a matched-template sync locator + robust fit (no slant)
  * tuning offset removed using the sync tone
  * amplitude-weighted (phasor-sum) per-pixel frequency estimate, rendered at
    two filter widths and blended per row by noise measured inside the sync
    pulses (narrow for weak rows, wide for strong rows)
Stacker (sstvdec/stack.py):
  * groups repeats via the event's transmission schedule (fixed period, deck
    of N slides, restarts handled) plus template-removed picture matching
  * merges with a per-pixel inverse-noise-variance weighted median
  * noise-adaptive cleanup for display

Output (default ./sstv_event/):
    audio/      downloaded recordings (kept; reruns don't re-download)
    decoded/    decode cache (.npz) - invalidated automatically when the
                decoder changes
    images/     every decoded image  <obs>_<n>_<mode>.png
    stacked/    merged slides        <mode>_slide_NN_x<copies>.png
    results.csv one row per decoded image
    index.html  contact sheet: merged slides first, then single decodes
"""

import argparse
import csv
import os
import html
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sstvdec import decoder, stack                     # noqa: E402
from sstvdec.modes import MODES, ISS_COMMON           # noqa: E402

API = "https://network.satnogs.org/api/observations/"
ISS_NORAD = 25544
ISS_SSTV_MHZ = (145.800, 437.550)       # ARISS SSTV downlinks used so far
UA = {"User-Agent": "iss-sstv-satnogs/2.0"}


# ---------------------------------------------------------------- SatNOGS
def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def fetch_observations(station, start, end, max_pages=200):
    """station: int or None (whole network). Note the API filter names:
    norad_cat_id (satellite__norad_cat_id is silently ignored) and the audio
    URL is in 'payload'."""
    import requests
    params = {"norad_cat_id": ISS_NORAD, "start": iso(start), "end": iso(end)}
    if station is not None:
        params["ground_station"] = station
    url, first, pages = API, True, 0
    while url and pages < max_pages:
        for attempt in range(4):
            try:
                r = requests.get(url, params=params if first else None,
                                 headers=UA, timeout=60)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        yield from r.json()
        url = r.links.get("next", {}).get("url")
        first, pages = False, pages + 1
        time.sleep(0.5)


def obs_freq(obs):
    for k in ("observation_frequency", "transmitter_downlink_low"):
        if obs.get(k):
            return int(obs[k])
    return None


def audio_url(obs):
    if isinstance(obs.get("payload"), str) and obs["payload"]:
        return obs["payload"]
    for p in obs.get("payloads") or []:
        if isinstance(p, str):
            return p
        if isinstance(p, dict) and p.get("payload"):
            return p["payload"]
    return None


def download(url, dest):
    import requests
    if dest.exists() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    tmp.rename(dest)
    time.sleep(0.5)
    return True


# ---------------------------------------------------------------- decoding
def decoder_version():
    """Hash of the decoder source: cached decodes are redone when it changes."""
    import hashlib
    h = hashlib.sha1()
    for f in ("decoder.py", "modes.py"):
        h.update((Path(decoder.__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def load_iq(path, fs=48000):
    """SatNOGS IQ dump: interleaved int16 I/Q, Doppler-corrected, 48 kS/s.
    Demodulated here with a narrower channel filter than the SatNOGS FM
    flowgraph (+/-7 kHz vs +/-8 kHz) and an amplitude-weighted
    discriminator; worth ~1-2 dB on the weakest signals (simulator)."""
    from scipy import signal
    raw = np.fromfile(path, dtype="<i2").astype(float)
    iq = raw[0::2] + 1j * raw[1::2]
    y = signal.oaconvolve(iq, signal.firwin(129, 7000, fs=fs), mode="same")
    p = y[1:] * np.conj(y[:-1])
    p = np.convolve(p, np.ones(4) / 4, mode="same")
    d = np.angle(p) * fs / (2 * np.pi * 5000.0)
    return np.concatenate([d[:1], d]), fs


def decode_cached(src, cache, version, modes, is_iq=False):
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["version"]) == version and list(z["modes"]) == list(modes):
                return [decoder.Decoded(str(z[f"mode{i}"]), z[f"ycc{i}"], z[f"valid{i}"],
                                        z[f"snr{i}"], z[f"noise{i}"], float(z[f"start{i}"]),
                                        float(z[f"unit{i}"]), float(z[f"off{i}"]),
                                        float(z[f"score{i}"]), int(z[f"syncs{i}"]),
                                        None if int(z[f"vis{i}"]) < 0 else int(z[f"vis{i}"]))
                        for i in range(int(z["n"]))]
        except Exception:
            pass
    if is_iq:
        audio, fs = load_iq(src)
        res = decoder.decode(audio, fs, modes=modes)
    else:
        res = decoder.decode(src, modes=modes)
    arr = {"version": version, "modes": np.array(modes), "n": len(res)}
    for i, d in enumerate(res):
        arr.update({f"mode{i}": d.mode, f"ycc{i}": d.ycc, f"valid{i}": d.valid,
                    f"snr{i}": d.snr, f"noise{i}": d.noise, f"start{i}": d.start_s,
                    f"unit{i}": d.unit_s, f"off{i}": d.offset_hz, f"score{i}": d.score,
                    f"syncs{i}": d.syncs, f"vis{i}": -1 if d.vis is None else d.vis})
    np.savez_compressed(cache, **arr)
    return res


# ---------------------------------------------------------------- output
def write_index(out, rows, slides, title):
    def card(href, cap):
        return (f'<figure><a href="{html.escape(href)}"><img src="{html.escape(href)}" '
                f'loading="lazy"></a><figcaption>{cap}</figcaption></figure>')
    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.</p>"]
    for mode in sorted({s["mode"] for s in slides}):
        ss = [s for s in slides if s["mode"] == mode]
        parts.append(f"<h2>{mode}: merged slides ({len(ss)})</h2><main>")
        parts += [card(f"stacked/{s['file']}",
                       f"{s['copies']} cop{'y' if s['copies'] == 1 else 'ies'} &middot; "
                       f"q={s['quality']:.2f}<br><small>{html.escape(', '.join(s['sources']))}</small>")
                  for s in ss]
        parts.append("</main>")
    singles = [r for r in rows if r.get("image")]
    parts.append(f"<h2>Individual decodes ({len(singles)})</h2><main>")
    parts += [card(f"images/{r['image']}",
                   f"#{r['obs_id']} img {r['img_n']} {r['mode']} q={r['quality']}"
                   f"<br>{html.escape(str(r['image_utc']))} &middot; stn {r['station']}")
              for r in singles]
    parts.append("</main>")
    page = ("<!doctype html><meta charset='utf-8'><meta name=viewport "
            "content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>"
            "body{font-family:sans-serif;margin:1rem;background:#fff;color:#111}"
            "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem}"
            "img{width:100%;border:1px solid #888}figcaption{font-size:.8rem;color:#444}"
            "@media (prefers-color-scheme:dark){body{background:#111;color:#eee}"
            "figcaption{color:#aaa}}</style>" + "".join(parts))
    (out / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source")
    default_station = int(os.environ.get("SATNOGS_STATION", 223))
    src.add_argument("--station", type=int, default=default_station,
                     help=f"SatNOGS station id (default {default_station}; set the "
                     "SATNOGS_STATION environment variable to change the default)")
    src.add_argument("--stations", help="comma list of station ids, or 'all' for the whole "
                     "network (other stations' recordings are NOT your own reception)")
    src.add_argument("--files", nargs="+", help="decode local recordings instead of SatNOGS "
                     "(.ogg/.wav/.flac audio, or .raw/.iq SatNOGS IQ dumps)")
    src.add_argument("--days", type=float, default=7)
    src.add_argument("--start", help="UTC, e.g. 2026-10-02T09:00")
    src.add_argument("--end", help="UTC, default now")
    src.add_argument("--freq-mhz", type=float, nargs="+", default=list(ISS_SSTV_MHZ),
                     help="downlink(s) to accept (default 145.800 437.550)")
    src.add_argument("--tol-khz", type=float, default=15.0)
    src.add_argument("--max-obs", type=int, default=400, help="safety cap on downloads")
    dec = ap.add_argument_group("decoding")
    dec.add_argument("--modes", nargs="+", default=list(ISS_COMMON),
                     help=f"modes to search (default: {' '.join(ISS_COMMON)}; "
                     f"all: {' '.join(MODES)})")
    dec.add_argument("--slides", type=int, default=12,
                     help="slides in the event's deck (the images say 'SLIDE x/N')")
    dec.add_argument("--min-quality", type=float, default=0.05,
                     help="drop decodes with less usable signal than this (0..1)")
    dec.add_argument("--no-stack", action="store_true")
    dec.add_argument("--raw", action="store_true", help="also save un-cleaned PNGs")
    ap.add_argument("--out", default="sstv_event")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("audio", "decoded", "images", "stacked") + (("raw",) if args.raw else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("images", "stacked", "raw"):
        for old in (out / sub).glob("*.png"):
            old.unlink()
    version = decoder_version()
    modes = [m for m in args.modes if m in MODES]

    # ---- collect recordings: (id, path, utc_start, station, freq, is_iq)
    recs = []
    if args.files:
        for f in args.files:
            p = Path(f)
            is_iq = p.suffix.lower() in (".raw", ".iq", ".dat")
            utc0 = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
            recs.append((p.stem, p, utc0, "local", None, is_iq))
        print(f"{len(recs)} local recording(s); file mtimes used as start times "
              "(schedule grouping is only as good as those)")
    else:
        end = parse_date(args.end) if args.end else datetime.now(timezone.utc)
        start = parse_date(args.start) if args.start else end - timedelta(days=args.days)
        if args.stations == "all":
            stations = [None]
        elif args.stations:
            stations = [int(s) for s in args.stations.split(",")]
        else:
            stations = [args.station]
        want = [int(round(f * 1e6)) for f in args.freq_mhz]
        for st in stations:
            label = "all stations" if st is None else f"station {st}"
            print(f"ISS observations, {label}, {iso(start)} -> {iso(end)}")
            for obs in fetch_observations(st, start, end):
                f = obs_freq(obs)
                if f and not any(abs(f - w) <= args.tol_khz * 1000 for w in want):
                    continue
                url = audio_url(obs)
                if not url:
                    continue
                if len(recs) >= args.max_obs:
                    print(f"  --max-obs {args.max_obs} reached")
                    break
                utc0 = parse_date(obs["start"])
                dest = out / "audio" / f"{obs['id']}_{utc0:%Y%m%d_%H%M%S}.ogg"
                try:
                    if download(url, dest):
                        print(f"  downloaded {dest.name}")
                except Exception as e:
                    print(f"  download failed {obs['id']}: {e}", file=sys.stderr)
                    continue
                recs.append((str(obs["id"]), dest, utc0, obs.get("ground_station"), f, False))

    # ---- decode
    rows, items = [], []
    for rid, path, utc0, station, freq, is_iq in sorted(recs, key=lambda r: r[2]):
        cache = out / "decoded" / f"{path.stem}.npz"
        try:
            res = decode_cached(path, cache, version, modes, is_iq)
        except Exception as e:
            print(f"  ERROR {path.name}: {e}", file=sys.stderr)
            continue
        qs = []
        for n, d in enumerate(res, 1):
            it = {"ycc": d.ycc, "noise": d.noise, "valid": d.valid, "mode": d.mode,
                  "utc": utc0.timestamp() + d.start_s, "src": f"{rid}#{n}", "station": station}
            q = stack.quality(it)
            qs.append(f"{d.mode} q={q:.2f}")
            row = {"obs_id": rid, "station": station, "freq_hz": freq or "", "img_n": n,
                   "mode": d.mode, "vis": "" if d.vis is None else d.vis,
                   "image_utc": datetime.fromtimestamp(it["utc"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                   "quality": round(q, 2), "syncs": d.syncs, "tune_hz": round(d.offset_hz, 1),
                   "image": ""}
            if q >= args.min_quality:
                name = f"{path.stem}_{n}_{d.mode}.png"
                stack.to_rgb(stack.denoise(d.ycc, d.noise)).save(out / "images" / name)
                if args.raw:
                    d.rgb.save(out / "raw" / name)
                row["image"] = name
                items.append(it)
            rows.append(row)
        print(f"  {path.name}: " + (", ".join(qs) if qs else "no SSTV image found"))

    # ---- stack per mode
    slides = []
    if items and not args.no_stack:
        for mode in sorted({it["mode"] for it in items}):
            its = [it for it in items if it["mode"] == mode]
            clusters, info = stack.group(its, args.slides)
            if info.get("deck_warning"):
                print("WARNING:", info["deck_warning"])
            if info.get("period"):
                print(f"{mode}: schedule {info['period']:.2f} s, {info['runs']} run(s), "
                      f"{info['linked_groups']} linked group(s), {info['n_slides']}-slide deck")
            ranked = []
            for cl in clusters:
                ycc, noise = stack.merge(its, cl)
                q = float(np.mean(stack.row_quality(noise)))
                ranked.append((q, cl, ycc, noise))
            ranked.sort(key=lambda r: -r[0])
            for n, (q, cl, ycc, noise) in enumerate(ranked, 1):
                name = f"{mode}_slide_{n:02d}_x{len(cl)}.png"
                stack.to_rgb(stack.denoise(ycc, noise)).save(out / "stacked" / name)
                slides.append({"mode": mode, "file": name, "copies": len(cl), "quality": q,
                               "sources": [its[i]["src"] for i, _, _ in cl]})
            print(f"{mode}: {len(its)} decodes -> {len(clusters)} slide(s)")

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["obs_id", "station", "freq_hz", "img_n", "mode", "vis",
                                          "image_utc", "quality", "syncs", "tune_hz", "image"])
        w.writeheader()
        w.writerows(rows)
    (out / "stacked" / "slides.json").write_text(json.dumps(slides, indent=1))
    title = args.title or ("ISS SSTV - " + ("local files" if args.files else
                           ("SatNOGS network" if args.stations == "all" else
                            f"SatNOGS station {args.stations or args.station}")))
    write_index(out, rows, slides, title)
    print(f"\nDone: {len(recs)} recording(s), {len(items)} usable image(s), "
          f"{len(slides)} merged slide(s). Open {out / 'index.html'}")


if __name__ == "__main__":
    main()
