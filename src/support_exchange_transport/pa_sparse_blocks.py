"""Finite-grid PA in sparse 2D blocks with analytic disk responses.

This is a geometry-adapted Boolean response pilot, NOT SO(2)/Bessel ACFO.
An 8x8 leaf uses one uint64; full squares use (x,y,size). Empty squares
are implicit. No dense pixel/tree array is used by the sparse builders.
All coordinates refer to the SAME physical midpoint projection grid.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from .pa_dyadic_bitmask import low_bits, pop64, row_span, row_bounds, or_span


@njit(cache=True)
def tile_disks(xy, radii, lower, spacing, n):
    """Matched row-span geometry, sparse hash of nonempty 8x8 tiles."""
    tiles = {np.int64(-1): np.uint64(0)}
    del tiles[np.int64(-1)]
    stride = n // 8
    rows = 0
    writes = 0
    full_skips = 0
    for a in range(len(radii)):
        first, stop = row_bounds(xy[a, 1], radii[a], lower, spacing, n)
        for y in range(first, stop):
            lo, hi = row_span(xy[a, 0], xy[a, 1], radii[a], lower, spacing, y, n)
            rows += 1
            if hi <= lo:
                continue
            for tx in range(lo // 8, (hi - 1) // 8 + 1):
                key = (y // 8) * stride + tx
                before = tiles.get(key, np.uint64(0))
                if before == low_bits(64):
                    full_skips += 1
                    continue
                left = max(lo - 8 * tx, 0)
                right = min(hi - 8 * tx, 8)
                bits = (low_bits(right) ^ low_bits(left)) << np.uint64((y % 8) * 8)
                tiles[key] = before | bits
                writes += 1
    full = np.empty((len(tiles), 3), np.int64)
    partial = np.empty((len(tiles), 3), np.uint64)
    nf = 0
    npart = 0
    for key, bits in tiles.items():
        x = (key % stride) * 8
        y = (key // stride) * 8
        if bits == low_bits(64):
            full[nf] = (x, y, 8)
            nf += 1
        else:
            partial[npart] = (np.uint64(x), np.uint64(y), bits)
            npart += 1
    return full[:nf].copy(), partial[:npart].copy(), np.array([rows, writes, full_skips, len(tiles)], np.int64)


@njit(cache=False)
def _visit(x, y, size, ids, xy, radii, lower, spacing, n,
           use_full, full, partial, stats, ancestor_bytes):
    # stats: node visits, disk/block tests, leaf/atom rows, early-full nodes,
    #        empty nodes, max simultaneously live candidate-array payload.
    stats[0] += 1
    candidates = np.empty(len(ids), np.int64)
    live = ancestor_bytes + candidates.nbytes
    stats[5] = max(stats[5], live)
    nc = 0
    x0 = lower[0] + (x + .5) * spacing
    x1 = lower[0] + (x + size - .5) * spacing
    y0 = lower[1] + (y + .5) * spacing
    y1 = lower[1] + (y + size - .5) * spacing
    for k in range(len(ids)):
        a = ids[k]
        dx0 = x0 - xy[a, 0]
        dx1 = x1 - xy[a, 0]
        dy0 = y0 - xy[a, 1]
        dy1 = y1 - xy[a, 1]
        nearx = max(dx0, -dx1, 0.)
        neary = max(dy0, -dy1, 0.)
        farx = max(abs(dx0), abs(dx1))
        fary = max(abs(dy0), abs(dy1))
        r2 = radii[a] * radii[a]
        far2 = farx * farx + fary * fary
        # Ambiguous roundoff cases descend to the reference row inequality.
        # This guard is conservative in tested floating-point cases, not a
        # general interval-arithmetic certificate for arbitrary coordinates.
        margin = 64 * np.finfo(np.float64).eps * (far2 + r2 + 1.)
        stats[1] += 1
        if use_full and far2 < r2 - margin:
            full.append((x, y, size))
            stats[3] += 1
            return 2
        if nearx * nearx + neary * neary <= r2 + margin:
            candidates[nc] = a
            nc += 1
    if nc == 0:
        stats[4] += 1
        return 0
    if size == 8:
        bits = np.uint64(0)
        for k in range(nc):
            a = candidates[k]
            for yy in range(y, y + 8):
                lo, hi = row_span(xy[a, 0], xy[a, 1], radii[a], lower, spacing, yy, n)
                lo = max(lo, x)
                hi = min(hi, x + 8)
                stats[2] += 1
                if hi > lo:
                    row = low_bits(hi - x) ^ low_bits(lo - x)
                    bits |= row << np.uint64((yy - y) * 8)
            if bits == low_bits(64):
                full.append((x, y, size))
                return 2
        if bits != 0:
            partial.append((np.uint64(x), np.uint64(y), bits))
            return 1
        return 0
    half = size // 2
    all_full = True
    any_nonempty = False
    for q in range(4):
        state = _visit(x + (q % 2) * half, y + (q // 2) * half, half,
                       candidates[:nc], xy, radii, lower, spacing, n,
                       use_full, full, partial, stats, live)
        all_full = all_full and state == 2
        any_nonempty = any_nonempty or state != 0
    if use_full and all_full:
        for q in range(4):
            full.pop()
        full.append((x, y, size))
        return 2
    return 1 if any_nonempty else 0


@njit(cache=False)
def block_disks(xy, radii, lower, spacing, n, use_full=True):
    """Direct geometry -> sparse block responses; optional full-node ablation.

    Recursive JIT and its caller deliberately do not use disk caching: on this
    Windows/Numba runtime loading the recursive cached linkage crashed a fresh
    process. Compilation is warmed outside benchmark timing.
    """
    full = [(np.int64(0), np.int64(0), np.int64(0))]
    partial = [(np.uint64(0), np.uint64(0), np.uint64(0))]
    full.pop()
    partial.pop()
    stats = np.zeros(6, np.int64)
    ids = np.arange(len(radii), dtype=np.int64)
    _visit(0, 0, n, ids, xy, radii, lower, spacing, n, use_full,
           full, partial, stats, ids.nbytes)
    f = np.empty((len(full), 3), np.int64)
    p = np.empty((len(partial), 3), np.uint64)
    for k in range(len(full)):
        f[k] = full[k]
    for k in range(len(partial)):
        p[k] = partial[k]
    return f, p, stats


@njit(cache=True)
def block_count(full, partial):
    total = 0
    for k in range(len(full)):
        total += full[k, 2] ** 2
    for k in range(len(partial)):
        total += pop64(partial[k, 2])
    return total


@njit(cache=True)
def decode_blocks(full, partial, n):
    out = np.zeros((n, (n + 63) // 64), np.uint64)
    for k in range(len(full)):
        x, y, size = full[k]
        for yy in range(y, y + size):
            or_span(out, yy, x, x + size)
    for k in range(len(partial)):
        x = int(partial[k, 0])
        y = int(partial[k, 1])
        bits = partial[k, 2]
        for j in range(8):
            row = (bits >> np.uint64(8 * j)) & np.uint64(255)
            out[y + j, x // 64] |= row << np.uint64(x % 64)
    return out


@dataclass
class SparseBlockMask:
    n: int
    full: np.ndarray
    partial: np.ndarray
    stats: np.ndarray

    @classmethod
    def from_disks(cls, xy, radii, lower, spacing, n, method='adaptive'):
        if not isinstance(n, int) or n < 8 or n & (n - 1):
            raise ValueError('n must be a power of two >= 8')
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)
        if xy.shape != (len(radii), 2) or lower.shape != (2,):
            raise ValueError('Incompatible geometry shapes')
        if not (np.isfinite(xy).all() and np.isfinite(radii).all() and
                np.isfinite(lower).all() and np.isfinite(spacing) and spacing > 0 and
                (radii >= 0).all()):
            raise ValueError('Expected finite geometry, nonnegative radii and positive spacing')
        if method == 'tiles':
            f, p, stats = tile_disks(xy, radii, lower, spacing, n)
        elif method in ('adaptive', 'leaf_only'):
            f, p, stats = block_disks(xy, radii, lower, spacing, n, method == 'adaptive')
        else:
            raise ValueError(method)
        return cls(n, f, p, stats)

    @property
    def nbytes(self):
        return self.full.nbytes + self.partial.nbytes

    def count(self):
        return block_count(self.full, self.partial)

    def packed(self):
        return decode_blocks(self.full, self.partial, self.n)
