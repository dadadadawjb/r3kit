import numpy as np


def mask_grid_sample(mask:np.ndarray, num_points:int) -> np.ndarray:
    '''
    mask: (H, W), binary mask
    pts: (N, 2), (x, y)
    '''
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=int)

    H, W = mask.shape
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    area = int(mask[y0:y1, x0:x1].sum())
    if area <= num_points:
        return np.stack([xs, ys], axis=1)

    bh, bw = (y1 - y0), (x1 - x0)
    Ny = max(1, int(round(np.sqrt(num_points * (bh / (bw + 1e-9))))))
    Nx = max(1, int(np.ceil(num_points / Ny)))
    gy = max(1, int(np.ceil(bh / Ny)))
    gx = max(1, int(np.ceil(bw / Nx)))

    tiles, counts = [], []
    for y in range(y0, y1, gy):
        yy1 = min(y1, y + gy)
        for x in range(x0, x1, gx):
            xx1 = min(x1, x + gx)
            tile = mask[y:yy1, x:xx1]
            c = int(tile.sum())
            if c == 0:
                continue
            tiles.append((y, yy1, x, xx1))
            counts.append(c)

    counts = np.asarray(counts, dtype=int)
    weights = counts / counts.sum()
    raw = weights * num_points
    quota = np.floor(raw).astype(int)
    remain = num_points - int(quota.sum())
    if remain > 0:
        frac = raw - quota
        order = np.argsort(-frac)
        for idx in order:
            if remain == 0:
                break
            cap = counts[idx] - quota[idx]
            if cap > 0:
                quota[idx] += 1
                remain -= 1
    assert quota.sum() == num_points

    pts = []
    for (y, yy1, x, xx1), q in zip(tiles, quota):
        if q <= 0:
            continue
        tile = mask[y:yy1, x:xx1]
        tys, txs = np.nonzero(tile)
        sel = np.random.choice(tys.size, size=q, replace=False)
        sel_y = y + tys[sel]
        sel_x = x + txs[sel]
        pts.extend(zip(sel_x.tolist(), sel_y.tolist()))
    return np.asarray(pts, dtype=int)
