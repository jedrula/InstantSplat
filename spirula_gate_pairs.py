#!/usr/bin/env python3
"""Drop the matched pairs that ARKit says cannot overlap, from a Spirula match database (matches.bin).

Why: on a repetitive scene (block2ha's identical facades) look-alike pairing links DIFFERENT buildings,
and those false pairs fold the model (3cc96e15, 23.6 m). Turning look-alike pairing off instead
(92b1e093) leaves only walk neighbours, so at a fast turn the chain breaks into separate models that
nothing can rejoin. ARKit drifts metres, but identical buildings are tens of metres apart: a pair is
kept when it is a walk neighbour (|i-j| <= overlap) or when its two ARKit positions are within
`radius` metres. Everything else is a look-alike the phone already knows is impossible.

matches.bin (Spirula 2026.9.30, "VKMT" v4), little-endian u32 throughout, decoded and checked to parse
exactly: magic, version, n_images; per image: name_len, name, n_features; n_pairs; per pair: i, j,
config, n_matches, n_matches x (idx_i, idx_j); then a trailer (camera groups / focals) copied as is.

usage: spirula_gate_pairs.py MATCHES.BIN FRAMES.TRAJ RADIUS_M OVERLAP
FRAMES.TRAJ is positional: line k is the k-th image by sorted name, as everywhere else in the pipeline.
"""
import math, struct, sys

path, traj, radius, overlap = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
b = open(path, 'rb').read()
assert b[:4] == b'VKMT', f'{path}: not a Spirula match database'
ver, n = struct.unpack_from('<II', b, 4)
assert ver == 4, f'{path}: matches.bin version {ver}, this reader knows 4 -- re-check the layout'
o, names = 12, []
for _ in range(n):
    L, = struct.unpack_from('<I', b, o); names.append(b[o + 4:o + 4 + L].decode()); o += 4 + L + 4
rows = [ln.split() for ln in open(traj) if len(ln.split()) == 7]
assert len(rows) == n, f'{traj} has {len(rows)} poses for {n} images; the traj is positional'
order = {nm: k for k, nm in enumerate(sorted(names))}
pos = [tuple(map(float, rows[order[nm]][4:7])) for nm in names]
seq = [order[nm] for nm in names]

np_, = struct.unpack_from('<I', b, o); head, o = b[:o], o + 4
kept, dropped_far, n_far = [], [], 0
for _ in range(np_):
    i, j, c, m = struct.unpack_from('<IIII', b, o)
    rec = b[o:o + 16 + 8 * m]; o += 16 + 8 * m
    d = math.dist(pos[i], pos[j])
    if abs(seq[i] - seq[j]) <= overlap or d <= radius:
        kept.append(rec)
    else:
        dropped_far.append((d, m))
trailer = b[o:]
open(path, 'wb').write(head + struct.pack('<I', len(kept)) + b''.join(kept) + trailer)
nk = len(kept); nd = len(dropped_far)
print(f'gate: {np_} matched pairs -> kept {nk} (walk neighbours within {overlap} or ARKit within {radius:g} m), '
      f'dropped {nd} as impossible')
if dropped_far:
    dropped_far.sort()
    strong = sum(1 for _, m in dropped_far if m >= 150)
    print(f'gate: dropped pairs span {dropped_far[0][0]:.1f}-{dropped_far[-1][0]:.1f} m apart, '
          f'{strong} of them with >=150 matches (look-alikes strong enough to fold a model)')
