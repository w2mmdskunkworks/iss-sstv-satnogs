"""
stack.py - group repeat transmissions of the same picture and merge them.

ISS SSTV events cycle a fixed slide deck on a fixed schedule (one image
every P seconds, deck of N slides). Grouping uses that schedule first:
  * transmission start times sit on a grid of period P; the grid phase jumps
    when the onboard system restarts, splitting the event into "runs";
  * within a run, grid slot mod N identifies the slide with certainty;
  * runs are linked by one cyclic offset each, found by picture matching of
    the run's slots (template-removed correlation; every slide shares the
    deck's header/footer, so the common template is subtracted first).
Without a usable schedule it falls back to picture matching alone.

Merging: per-pixel weighted median of aligned copies, weights 1/noise^8 per
row (noise measured by the decoder in the sync pulses). The steep exponent
was chosen on the simulator: copies' row quality differs a lot, so the
cleanest copy should dominate (1/noise^2 lost ~1 dB with 4 copies), while
copies of similar quality still vote pixel by pixel, rejecting clicks.

Items are dicts: ycc (H,W,3 uint8), noise (H), valid (H), utc (s), mode.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

MAX_DY, MAX_DX = 4, 3
# row quality from measured noise (0-255 rms units): >= NOISE_BAD is useless,
# <= NOISE_GOOD is clean. Calibrated on the simulator (see calibrate.py).
NOISE_GOOD, NOISE_BAD = 6.0, 25.0


# ---------------------------------------------------------------- weights
def row_quality(noise, valid=None):
    q = np.clip((NOISE_BAD - np.nan_to_num(noise, nan=1e9)) / (NOISE_BAD - NOISE_GOOD), 0, 1)
    if valid is not None:
        q = np.where(valid, q, 0.0)
    return q


def row_weights(item):
    """Grouping weight per row (0..1)."""
    return row_quality(item["noise"], item.get("valid")) ** 2


def inv_var(item):
    """Merge weight per row: inverse noise variance (0 for unusable rows)."""
    n = np.nan_to_num(item["noise"], nan=1e9)
    w = 1.0 / np.maximum(n, 1.0) ** 2
    return np.where(row_quality(item["noise"], item.get("valid")) > 0, w, 0.0)


def quality(item):
    """0..1 image quality: mean row quality."""
    return float(np.mean(row_quality(item["noise"], item.get("valid"))))


# ---------------------------------------------------------------- matching
def _feat(ycc):
    return ndimage.median_filter(ycc[:, :, 0].astype(float), size=(3, 5))


def _shift(a, dy, dx, fill=np.nan):
    out = np.full(a.shape, fill, dtype=float)
    H, W = a.shape[:2]
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[ys, xs] = a[yd, xd]
    return out


def similarity(fa, wa, fb, wb):
    """Best weighted correlation of b shifted onto a. Returns (r, dy, dx);
    r = -1 when the two share too few usable rows to judge."""
    best = (-1.0, 0, 0)
    need = 0.08 * len(wa)
    for dy in range(-MAX_DY, MAX_DY + 1):
        wbs = _shift(wb[:, None], dy, 0, 0.0)[:, 0]
        ww = wa * wbs
        if ww.sum() < need:
            continue
        for dx in range(-MAX_DX, MAX_DX + 1):
            fbs = _shift(fb, dy, dx)
            m = ~np.isnan(fbs)
            W = ww[:, None] * m
            if W.sum() == 0:
                continue
            b = np.nan_to_num(fbs)
            mu_a = (W * fa).sum() / W.sum()
            mu_b = (W * b).sum() / W.sum()
            cov = (W * (fa - mu_a) * (b - mu_b)).sum()
            va = (W * (fa - mu_a) ** 2).sum()
            vb = (W * (b - mu_b) ** 2).sum()
            r = cov / np.sqrt(va * vb + 1e-12)
            if r > best[0]:
                best = (float(r), dy, dx)
    return best


def template(items):
    """Per-pixel median over usable rows of all items: the deck's shared
    background/header/footer."""
    F = np.stack([_feat(it["ycc"]) for it in items])
    Wl = np.stack([row_weights(it) for it in items])
    T = np.zeros(F.shape[1:])
    for y in range(F.shape[1]):
        m = Wl[:, y] > 0
        if m.sum() >= 3:
            T[y] = np.median(F[m, y], axis=0)
    return T


def _cluster_feat(R, W, cl):
    acc = np.zeros_like(R[cl[0][0]])
    accw = np.zeros(acc.shape[0])
    for j, dy, dx in cl:
        wj = _shift(W[j][:, None], dy, 0, 0.0)[:, 0]
        acc += np.nan_to_num(_shift(R[j], dy, dx)) * wj[:, None]
        accw += wj
    return acc / np.maximum(accw[:, None], 1e-12), accw


def group_by_picture(items, thresh=0.6, join_thresh=0.5, min_quality=0.05):
    """Greedy clustering on template-removed features, then whole-cluster
    joins. Returns [[(idx, dy, dx), ...], ...]."""
    T = template(items)
    R = [_feat(it["ycc"]) - T for it in items]
    W = [row_weights(it) for it in items]
    order = sorted((i for i in range(len(items)) if quality(items[i]) >= min_quality),
                   key=lambda i: -quality(items[i]))
    used, clusters = set(), []
    for seed in order:
        if seed in used:
            continue
        used.add(seed)
        cl = [(seed, 0, 0)]
        while True:
            mR, mW = _cluster_feat(R, W, cl)
            best = None
            for j in order:
                if j in used:
                    continue
                r, dy, dx = similarity(mR, mW, R[j], W[j])
                if r >= thresh and (best is None or r > best[0]):
                    best = (r, j, dy, dx)
            if best is None:
                break
            used.add(best[1])
            cl.append(best[1:])
        clusters.append(cl)
    return join_clusters(items, clusters, join_thresh, R, W)


def join_clusters(items, clusters, thresh=0.5, R=None, W=None):
    if R is None:
        T = template(items)
        R = [_feat(it["ycc"]) - T for it in items]
        W = [row_weights(it) for it in items]
    clusters = [list(c) for c in clusters]
    while len(clusters) > 1:
        feats = [_cluster_feat(R, W, c) for c in clusters]
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                r, dy, dx = similarity(*feats[a], *feats[b])
                if r >= thresh and (best is None or r > best[0]):
                    best = (r, a, b, dy, dx)
        if best is None:
            break
        _, a, b, dy, dx = best
        clusters[a] += [(j, djy + dy, djx + dx) for j, djy, djx in clusters[b]]
        del clusters[b]
    return clusters


# ---------------------------------------------------------------- schedule
def fit_period(utc, min_s=20.0, max_s=1200.0, tol=0.02):
    """Transmission period: the most common gap between consecutive images
    (inside passes). None if there is no repeated gap."""
    g = np.diff(np.sort(utc))
    g = g[(g > min_s) & (g < max_s)]
    if len(g) < 2:
        return None
    counts = [(np.sum(np.abs(g / c - 1) < tol), -c) for c in g]
    n, c = max(counts)
    if n < 2:
        return None
    c = -c
    return float(np.median(g[np.abs(g / c - 1) < tol]))


MAX_RUN_GAP_S = 4 * 3600   # never assume the schedule ran on unobserved for longer


def _runs(utc, period, tol=0.03):
    """Split transmissions into runs of one continuous schedule. A new run
    starts when the grid phase jumps (onboard restart) OR after a gap longer
    than MAX_RUN_GAP_S: over a night with no passes the phase can line up by
    chance (Oct 6 2026: 340.004 periods) while the deck itself was paused or
    restarted, which mislabelled every slide of the next pass by one."""
    order = np.argsort(utc)
    run = np.zeros(len(utc), int)
    slot = np.zeros(len(utc), int)
    rid, start, prev = -1, None, None
    for i in order:
        if start is not None:
            n_prev = (utc[i] - utc[prev]) / period
            if abs(n_prev - round(n_prev)) > tol or utc[i] - utc[prev] > MAX_RUN_GAP_S:
                start = None
        if start is None:
            rid += 1
            start = utc[i]
        run[i] = rid
        slot[i] = int(round((utc[i] - start) / period))
        prev = i
    return run, slot


LINK_MARGIN = 0.10     # best offset must beat the runner-up by this much


def group_by_schedule(items, n_slides, period, link_thresh=0.5):
    utc = np.array([it["utc"] for it in items])
    run, slot = _runs(utc, period)
    key = {}
    for i in range(len(items)):
        key.setdefault((run[i], slot[i] % n_slides), []).append(i)
    T = template(items)
    R = [_feat(it["ycc"]) - T for it in items]
    W = [row_weights(it) for it in items]
    subs = list(key)
    feats = {k: _cluster_feat(R, W, [(i, 0, 0) for i in key[k]]) for k in subs}
    runs = sorted(set(run))
    run_q = {r: sum(quality(items[i]) for i in range(len(items)) if run[i] == r) for r in runs}

    # Incremental linking. Start from the strongest run; repeatedly link the
    # unlinked run whose best cyclic offset most clearly beats its runner-up,
    # scoring each offset against the MERGED slots of everything linked so
    # far (much cleaner than single noisy copies). Small runs (1-2 images)
    # can coincidentally resemble the wrong slide: requiring a clear margin
    # leaves them unlinked rather than guessing (Oct 6 2026: a 1-slot-off
    # link put Sophie Adenot copies into the soldering slide).
    off = {}
    acc, accw = {}, {}

    def add_run(r, o, sign=1):
        if sign > 0:
            off[r] = o
        for (rr, sl) in key:
            if rr != r:
                continue
            lab = (sl + o) % n_slides
            f, w = feats[(rr, sl)]
            acc[lab] = acc.get(lab, 0) + sign * f * w[:, None]
            accw[lab] = accw.get(lab, 0) + sign * w
        if sign < 0:
            del off[r]

    def offset_scores(r):
        slots = [(sl, feats[(rr, sl)]) for (rr, sl) in key if rr == r]
        scores = []
        for o in range(n_slides):
            vals = []
            for sl, (f, w) in slots:
                lab = (sl + o) % n_slides
                if lab in acc and np.sum(accw[lab]) > 1e-9:
                    gf = acc[lab] / np.maximum(accw[lab][:, None], 1e-12)
                    v = similarity(gf, accw[lab], f, w)[0]
                    if v > -1:
                        vals.append(v)
            scores.append(np.mean(vals) if vals else -1.0)
        order = np.argsort(scores)[::-1]
        return int(order[0]), scores[order[0]], scores[order[0]] - scores[order[1]]

    anchor = max(runs, key=lambda r: run_q[r])
    add_run(anchor, 0)
    while True:
        best = None
        for r in runs:
            if r in off:
                continue
            o, top, margin = offset_scores(r)
            if top >= link_thresh and margin >= LINK_MARGIN:
                if best is None or margin > best[0]:
                    best = (margin, r, o)
        if best is None:
            break
        add_run(best[1], best[2])

    # Refinement: a run linked early was scored against a deck that later
    # became much cleaner. Re-score every linked run against all the others
    # (leave-one-out) and move it if another offset now clearly wins.
    for _ in range(5):
        moved = False
        for r in [r for r in runs if r in off and r != anchor]:
            o_old = off[r]
            add_run(r, o_old, -1)
            o, top, margin = offset_scores(r)
            if o != o_old and top >= link_thresh and margin >= LINK_MARGIN:
                add_run(r, o)
                moved = True
            else:
                add_run(r, o_old)
        if not moved:
            break

    def find(r):
        return (0, off[r]) if r in off else (r + 1, 0)

    label = {}
    for i in range(len(items)):
        root, o = find(run[i])
        label.setdefault((root, (slot[i] + o) % n_slides), []).append(i)
    clusters = []
    for members in label.values():
        members.sort(key=lambda i: -quality(items[i]))
        ref = members[0]
        cl = [(ref, 0, 0)]
        for j in members[1:]:
            r, dy, dx = similarity(R[ref], W[ref], R[j], W[j])
            cl.append((j, dy, dx) if r > 0.3 else (j, 0, 0))
        clusters.append(cl)
    groups = len({find(r)[0] for r in runs})
    if groups > 1:                       # unlinked runs: strict picture joins
        clusters = join_clusters(items, clusters, 0.65, R, W)
    return clusters, {"period": period, "runs": len(runs), "linked_groups": groups}


def check_deck_size(items, period, n_slides, candidates=range(4, 25)):
    """Sanity check of the deck size from timing + pictures. Within a run,
    images N slots apart should look alike. Returns (best_N or None, scores);
    ties between multiples are common (passes ~3 decks apart) - callers
    should only override the user's N when it is clearly contradicted."""
    utc = np.array([it["utc"] for it in items])
    run, slot = _runs(utc, period)
    T = template(items)
    R = [_feat(it["ycc"]) - T for it in items]
    W = [row_weights(it) for it in items]
    pairs = {}
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if run[a] == run[b] and slot[a] != slot[b]:
                r = similarity(R[a], W[a], R[b], W[b])[0]
                if r > -1:
                    pairs.setdefault(abs(slot[a] - slot[b]), []).append(r)
    if not pairs:
        return None, {}
    base = np.mean([r for v in pairs.values() for r in v])
    scores = {N: float(np.mean([r for d, v in pairs.items() if d % N == 0 for r in v]) - base)
              for N in candidates
              if sum(len(v) for d, v in pairs.items() if d % N == 0) >= 2}
    if not scores:
        return None, scores
    N = max(scores, key=scores.get)
    return (N if scores[N] > 0.15 else None), scores


def group(items, n_slides=12):
    """Group same-mode items. Schedule-aware when a period is found."""
    utc = np.array([it["utc"] for it in items])
    period = fit_period(utc) if len(items) >= 3 else None
    if period is None:
        return group_by_picture(items), {"period": None}
    N, scores = check_deck_size(items, period, n_slides)
    info = {}
    if scores and n_slides in scores and scores[n_slides] < max(scores.values()) - 0.1:
        info["deck_warning"] = (f"timing suggests a {N}-slide deck, not {n_slides}; "
                                f"using {N}")
        n_slides = N or n_slides
    clusters, i2 = group_by_schedule(items, n_slides, period)
    info.update(i2, n_slides=n_slides)
    return clusters, info


# ---------------------------------------------------------------- merging
MERGE_POWER = 4      # weight = inv_var ** MERGE_POWER = 1/noise^8


def merge(items, cluster):
    """Weighted per-pixel median of aligned copies. Returns (ycc, noise_rows)
    where noise_rows estimates the merged noise (inverse-variance sum)."""
    S, Wt = [], []
    H = items[cluster[0][0]]["ycc"].shape[0]
    ivsum = np.zeros(H)
    for idx, dy, dx in cluster:
        S.append(_shift(items[idx]["ycc"].astype(float), dy, dx))
        iv = _shift(inv_var(items[idx])[:, None], dy, 0, 0.0)[:, 0]
        ivsum += iv
        Wt.append(np.broadcast_to((iv ** MERGE_POWER)[:, None], S[-1].shape[:2]))
    S = np.stack(S)
    Wt = np.stack(Wt)[..., None] * ~np.isnan(S)
    S = np.nan_to_num(S, nan=128.0)
    if len(cluster) == 1:
        out = S[0]
    else:
        order = np.argsort(S, axis=0)
        Ss = np.take_along_axis(S, order, 0)
        Ws = np.take_along_axis(np.broadcast_to(Wt, S.shape), order, 0)
        cw = np.cumsum(Ws, axis=0)
        tot = cw[-1]
        k = (cw >= tot / 2).argmax(axis=0)
        out = np.take_along_axis(Ss, k[None], 0)[0]
        out = np.where(tot == 0, np.median(S, axis=0), out)
    noise = np.where(ivsum > 0, 1.0 / np.sqrt(np.maximum(ivsum, 1e-12)), np.inf)
    return np.clip(out, 0, 255).astype(np.uint8), noise


# ---------------------------------------------------------------- cleanup
def denoise(ycc, noise):
    """Noise-adaptive display cleanup (method chosen on the simulator).
    Placeholder: luma median blended per row by noise, chroma smoothing
    faded to grey on noisy rows."""
    q = row_quality(noise)
    img = ycc.astype(float)
    a = (1.0 - q)[:, None]
    Y = img[..., 0]
    img[..., 0] = a * ndimage.median_filter(Y, size=5) + (1 - a) * Y
    sat = ndimage.gaussian_filter1d(np.clip(q * 1.3, 0, 1), 2.0)[:, None]
    for c in (1, 2):
        ch = ndimage.gaussian_filter(ndimage.median_filter(img[..., c], size=7), 1.5)
        img[..., c] = 128.0 + (ch - 128.0) * sat
    return np.clip(img, 0, 255).astype(np.uint8)


def to_rgb(ycc):
    return Image.fromarray(ycc, "YCbCr").convert("RGB")
