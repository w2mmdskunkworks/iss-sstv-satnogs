"""Audit slide grouping on a finished run: python dev/audit_grouping.py [SRC ...]
With no args, checks every decode is in the slide it best matches (leave-one-out).
With sources (e.g. 15124611#1), shows their top matching slides."""
import os, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from datetime import datetime, timezone
from sstvdec import stack as S
import iss_sstv
items = []
for f in sorted(glob.glob(os.path.join(ROOT, 'sstv_event/decoded/*.npz'))):
    z = np.load(f); stem = os.path.basename(f)[:-4]; oid, d, t = stem.split('_')
    t0 = datetime.strptime(d + t, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).timestamp()
    for i in range(int(z['n'])):
        it = {'ycc': z[f'ycc{i}'], 'noise': z[f'noise{i}'], 'valid': z[f'valid{i}'],
              'utc': t0 + float(z[f'start{i}']), 'src': f'{oid}#{i+1}'}
        if S.quality(it) >= 0.05:
            items.append(it)
clusters, info = S.group(items, 12)
T = S.template(items); R = [S._feat(it['ycc']) - T for it in items]; W = [S.row_weights(it) for it in items]
names = []
for cl in clusters:
    best = max(cl, key=lambda c: S.quality(items[c[0]]))
    names.append(items[best[0]]['src'])
feats = [S._cluster_feat(R, W, cl) for cl in clusters]
for src in sys.argv[1:]:
    j = [k for k, it in enumerate(items) if it['src'] == src][0]
    sims = [S.similarity(f, w, R[j], W[j])[0] for f, w in feats]
    mine = [ci for ci, cl in enumerate(clusters) if any(c[0] == j for c in cl)][0]
    order = np.argsort(sims)[::-1]
    print(src, 'q=%.2f' % S.quality(items[j]), 'assigned to cluster', mine,
          '| top matches:', [(int(c), round(sims[c], 2), names[c]) for c in order[:4]])

if not sys.argv[1:]:
    # audit: every item vs its assigned cluster (cluster features WITHOUT the item itself)
    bad = 0
    for j, it in enumerate(items):
        mine = [ci for ci, cl in enumerate(clusters) if any(c[0] == j for c in cl)][0]
        sims = []
        for ci, cl in enumerate(clusters):
            cl2 = [c for c in cl if c[0] != j]
            if not cl2:
                sims.append(-1.0); continue
            f, w = S._cluster_feat(R, W, cl2)
            sims.append(S.similarity(f, w, R[j], W[j])[0])
        best = int(np.argmax(sims))
        flag = '' if best == mine or sims[best] < 0.5 else '  <-- better match elsewhere'
        if flag: bad += 1
        print(f"{it['src']:12s} q={S.quality(it):.2f} in {mine:2d} (sim {sims[mine]:5.2f})  best {best:2d} (sim {sims[best]:5.2f}){flag}")
    print('suspicious:', bad, 'of', len(items))
