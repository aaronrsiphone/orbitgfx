"""Lookup textures (LUTs) for the 2.0 shaders: bake in NumPy, cache as PNG.

GLSL ES 1.0 has no float or 3D textures, so every table is an 8-bit RGB PNG
(RGB only: alpha would be premultiplied on load), bound to the shader with
`shader.set_uniform(name, texture)`.

Phase 0 measured the bakes at < 0.1 s on an iPhone 16 Pro Max, so the cache
mostly saves PNG encoding; it is keyed on earth.TEXTURE_VERSION, the LUT name
and its parameters. Bump TEXTURE_VERSION whenever any baked texture changes.

    img = cached_lut('transmittance', bake_transmittance, size=(256, 64))
    tex = lut_texture(img)
"""
import hashlib
import math
import os

import numpy as np
from PIL import Image, ImageFilter

from .earth import TEXTURE_VERSION

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.orbitgfx_cache')


def to_rgb8(arr):
    """float array (h, w, 3) or (h, w) in [0, 1] -> RGB PIL image, row 0 = top."""
    a = np.asarray(arr, float)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] != 3:
        raise ValueError('LUTs are RGB only (alpha is premultiplied on load)')
    return Image.fromarray(np.uint8(np.clip(np.round(a * 255.0), 0, 255)), 'RGB')


def encode_sqrt(x):
    """sqrt encoding for values near 0 (e.g. transmittance); decode with c * c."""
    return np.sqrt(np.clip(x, 0.0, 1.0))


def lut_key(name, params):
    h = hashlib.sha1(repr((TEXTURE_VERSION, name, sorted(params.items()))).encode())
    return '%s_v%d_%s' % (name, TEXTURE_VERSION, h.hexdigest()[:12])


def cached_lut(name, bake, cache_dir=None, **params):
    """bake(**params) -> RGB PIL image, cached under .orbitgfx_cache/<key>.png."""
    path = os.path.join(cache_dir or CACHE_DIR, lut_key(name, params) + '.png')
    if os.path.exists(path):
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            pass
    img = bake(**params)
    if img.mode != 'RGB':
        raise ValueError('LUT %r must be RGB, got %s' % (name, img.mode))
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
    except Exception:
        pass
    return img


def lut_texture(img, nearest=False):
    """RGB PIL image -> scene.Texture (linear filtering unless nearest)."""
    from .earth import pil_texture
    import scene
    tex = pil_texture(img)
    if nearest:
        tex.filtering_mode = getattr(scene, 'FILTERING_NEAREST', 0)
    return tex


# ============================================================ atmosphere
# Bruneton 2008 / Hillaire 2020 coefficients, per km at sea level.
ATMO = dict(
    beta_r=(5.802e-3, 13.558e-3, 33.1e-3), h_r=8.0,          # Rayleigh scattering
    beta_ms=3.996e-3, beta_me=4.44e-3, h_m=1.2,              # Mie scattering / extinction
    beta_o=(0.650e-3, 1.881e-3, 0.085e-3), o_c=25.0, o_w=15.0,  # ozone absorption, tent
    top_km=100.0,
)
TRANS_SIZE = (256, 64)        # (mu axis, r axis)


def planet(atmo_scale=3.0, re_km=6378.137):
    """(ground, top) radii in km of the drawn planet.

    The on-screen atmosphere is `atmo_scale` times thicker than the real one
    relative to the Earth. That is modelled as a real 100 km atmosphere on a
    planet `atmo_scale` times smaller, so all optics stay in real km and the
    shader's unit radius is rg km.
    """
    rg = re_km / float(atmo_scale)
    return rg, rg + ATMO['top_km']


def extinction(h_km):
    """Extinction coefficient (..., 3) per km at altitude h."""
    h = np.asarray(h_km, float)[..., None]
    a = ATMO
    rr = np.exp(-h / a['h_r'])
    rm = np.exp(-h / a['h_m'])
    ro = np.maximum(0.0, 1.0 - np.abs(h - a['o_c']) / a['o_w'])
    return (np.array(a['beta_r']) * rr + a['beta_me'] * rm + np.array(a['beta_o']) * ro)


def _dist_top(r, mu, rt):
    return np.maximum(0.0, -r * mu + np.sqrt(np.maximum(r * r * (mu * mu - 1.0) + rt * rt, 0.0)))


def transmittance_direct(r, mu, rg, rt, steps=40):
    """T (..., 3) from radius r (km) along cos-zenith mu to the top of the atmosphere."""
    r = np.asarray(r, float)
    mu = np.asarray(mu, float)
    d = _dist_top(r, mu, rt)
    s = (np.arange(steps) + 0.5) / steps
    t = d[..., None] * s
    rs = np.sqrt(t * t + 2.0 * r[..., None] * mu[..., None] * t + r[..., None] ** 2)
    od = (extinction(rs - rg) * (d[..., None, None] / steps)).sum(-2)
    return np.exp(-od)


def trans_coords(xm, xr, rg, rt):
    """Texture parameters (0..1) -> (r, mu), Bruneton's mapping."""
    H = math.sqrt(rt * rt - rg * rg)
    rho = xr * H
    r = np.sqrt(rho * rho + rg * rg)
    dmin, dmax = rt - r, rho + H
    d = dmin + xm * (dmax - dmin)
    mu = np.where(d == 0.0, 1.0, (H * H - rho * rho - d * d) / np.maximum(2.0 * r * d, 1e-9))
    return r, np.clip(mu, -1.0, 1.0)


def trans_params(r, mu, rg, rt):
    """(r, mu) -> texture parameters (xm, xr), the inverse of trans_coords."""
    H = math.sqrt(rt * rt - rg * rg)
    rho = np.sqrt(np.maximum(r * r - rg * rg, 0.0))
    d = _dist_top(r, mu, rt)
    dmin, dmax = rt - r, rho + H
    return (d - dmin) / np.maximum(dmax - dmin, 1e-9), rho / H


def bake_transmittance(atmo_scale=3.0, size=TRANS_SIZE, steps=40):
    """RGB LUT: x = mu parameter, y = r parameter (v up), value sqrt(T)."""
    W, H = size
    rg, rt = planet(atmo_scale)
    xm = np.arange(W) / (W - 1.0)
    xr = np.arange(H) / (H - 1.0)
    r, mu = trans_coords(xm[None, :], xr[:, None], rg, rt)
    T = transmittance_direct(r, mu, rg, rt, steps)          # (H, W, 3), row 0 = r ground
    return to_rgb8(np.flipud(encode_sqrt(T)))                # PIL row 0 = top = v 1


def lookup_transmittance(img, r, mu, rg, rt):
    """NumPy model of the shader lookup (bilinear, texel-centred), for tests."""
    a = np.asarray(img, float)[::-1] / 255.0                 # back to v-up rows
    Hh, W = a.shape[:2]
    xm, xr = trans_params(np.asarray(r, float), np.asarray(mu, float), rg, rt)
    fx = np.clip(xm, 0, 1) * (W - 1)
    fy = np.clip(xr, 0, 1) * (Hh - 1)
    x0 = np.minimum(np.floor(fx).astype(int), W - 2)
    y0 = np.minimum(np.floor(fy).astype(int), Hh - 2)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    c = (a[y0, x0] * (1 - tx) * (1 - ty) + a[y0, x0 + 1] * tx * (1 - ty)
         + a[y0 + 1, x0] * (1 - tx) * ty + a[y0 + 1, x0 + 1] * tx * ty)
    return c * c


def transmittance_lut(atmo_scale=3.0, cache_dir=None):
    """Cached PIL image of the transmittance LUT for this atmo_scale."""
    return cached_lut('transmittance', bake_transmittance, cache_dir=cache_dir,
                      atmo_scale=float(atmo_scale), size=TRANS_SIZE, steps=40)


# ============================================================ clouds (phase 3)
# Noise atlas: a 30^3 tiling 3D noise in 30 slices of 30 x 30 texels, each slice
# in a 32 x 32 cell with a 1-texel wrap pad (so hardware bilinear filtering wraps
# correctly inside a slice), 8 x 4 cells in a 256 x 128 RGB image.
#   R  Perlin-Worley (base shape)    G  Worley fbm (detail)    B  finer Worley fbm
NOISE_TILE = 30
NOISE_CELL = 32
NOISE_GRID = (8, 4)
NOISE_ATLAS = (NOISE_GRID[0] * NOISE_CELL, NOISE_GRID[1] * NOISE_CELL)   # (256, 128)
WEATHER_SIZE = (1024, 512)


def _ihash(ix, iy, iz, seed):
    """Integer lattice hash -> uint32 array (vectorised, deterministic)."""
    h = (np.asarray(ix, np.int64) * 73856093) ^ (np.asarray(iy, np.int64) * 19349663) \
        ^ (np.asarray(iz, np.int64) * 83492791) ^ (int(seed) * 2654435761)
    h = h & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return (h ^ (h >> 16)) & 0xFFFFFFFF


def _rand01(ix, iy, iz, seed):
    return _ihash(ix, iy, iz, seed).astype(np.float64) / 4294967296.0


_GRADS = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
                   [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
                   [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]], float)


def gradient_noise(p, period=None, seed=0):
    """3D gradient (Perlin) noise in about [-1, 1]; `period` (int) tiles the lattice."""
    p = np.asarray(p, float)
    pi = np.floor(p)
    f = p - pi
    pi = pi.astype(np.int64)
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = pi + np.array([dx, dy, dz])
                if period:
                    c = c % period
                g = _GRADS[_ihash(c[..., 0], c[..., 1], c[..., 2], seed) & 15]
                d = f - np.array([dx, dy, dz], float)
                w = ((u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1])
                     * (u[..., 2] if dz else 1 - u[..., 2]))
                out = out + w * (g * d).sum(-1)
    return out


def fbm_noise(p, octaves=5, lac=2.0, gain=0.5, period=None, seed=0):
    p = np.asarray(p, float)
    s, a, n = 0.0, 0.5, 0.0
    for k in range(octaves):
        per = int(round(period * lac ** k)) if period else None
        s = s + a * gradient_noise(p * lac ** k, per, seed + 17 * k)
        n += a
        a *= gain
    return s / n


def worley(p, period, seed=0):
    """1 - distance to the nearest feature point (one per cell), lattice tiled by `period`."""
    p = np.asarray(p, float)
    pi = np.floor(p).astype(np.int64)
    f = p - pi
    best = np.full(p.shape[:-1], 9.0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = (pi + np.array([dx, dy, dz])) % period
                fp = np.stack([_rand01(c[..., 0], c[..., 1], c[..., 2], seed + k) for k in range(3)], -1)
                d = np.array([dx, dy, dz], float) + fp - f
                best = np.minimum(best, (d * d).sum(-1))
    return 1.0 - np.clip(np.sqrt(best), 0.0, 1.0)


def _norm01(x, lo=0.5, hi=99.5):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)


def noise_volume(seed=7):
    """(30, 30, 30, 3) tiling noise [z, y, x, channel] in [0, 1]."""
    T = NOISE_TILE
    g = (np.arange(T) + 0.0) / T
    z, y, x = np.meshgrid(g, g, g, indexing='ij')
    P = np.stack([x, y, z], -1)
    perlin = _norm01(fbm_noise(P * 3, octaves=3, period=3, seed=seed))
    w0 = 0.625 * worley(P * 3, 3, seed + 1) + 0.25 * worley(P * 6, 6, seed + 2) \
        + 0.125 * worley(P * 12, 12, seed + 3)
    w0 = _norm01(w0)
    pw = np.clip((perlin - (w0 - 1.0)) / (2.0 - w0) * 1.0, 0.0, 1.0)      # remap(perlin, w-1, 1, 0, 1)
    R = _norm01(pw)
    G = _norm01(0.625 * worley(P * 5, 5, seed + 4) + 0.25 * worley(P * 10, 10, seed + 5)
                + 0.125 * worley(P * 15, 15, seed + 6))
    B = _norm01(0.6 * worley(P * 6, 6, seed + 7) + 0.4 * worley(P * 15, 15, seed + 8))
    return np.stack([R, G, B], -1)


def bake_noise(seed=7):
    """The 256 x 128 atlas (see NOISE_TILE). Cell for slice s: (s % 8, s // 8), v up."""
    vol = noise_volume(seed)
    T, C = NOISE_TILE, NOISE_CELL
    W, H = NOISE_ATLAS
    A = np.zeros((H, W, 3))
    idx = np.r_[T - 1, np.arange(T), 0]                     # 32 rows: pad, 30 content, pad
    for s in range(T):
        cx, cy = s % NOISE_GRID[0], s // NOISE_GRID[0]
        sl = vol[s][idx][:, idx]                             # (32, 32, 3) rows = y (v up)
        A[cy * C:(cy + 1) * C, cx * C:(cx + 1) * C] = sl
    return to_rgb8(np.flipud(A))                             # PIL row 0 = top = v 1


def lookup_noise(img, p):
    """NumPy model of the shader's noise3 (bilinear in the atlas, linear across slices)."""
    a = np.asarray(img, float)[::-1] / 255.0
    T, C = NOISE_TILE, NOISE_CELL
    p = np.asarray(p, float) % T
    z0 = np.floor(p[..., 2])
    fz = (p[..., 2] - z0)[..., None]
    z1 = (z0 + 1) % T

    def slice_(z):
        cx, cy = z % NOISE_GRID[0], np.floor(z / NOISE_GRID[0])
        u = cx * C + 1.5 + p[..., 0] - 0.5                  # pixel-centre space -> index space
        v = cy * C + 1.5 + p[..., 1] - 0.5
        x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
        tx, ty = (u - x0)[..., None], (v - y0)[..., None]
        return (a[y0, x0] * (1 - tx) * (1 - ty) + a[y0, x0 + 1] * tx * (1 - ty)
                + a[y0 + 1, x0] * (1 - tx) * ty + a[y0 + 1, x0 + 1] * tx * ty)
    return slice_(z0) * (1 - fz) + slice_(z1) * fz


def noise_lut(seed=7, cache_dir=None):
    return cached_lut('noise3d', bake_noise, cache_dir=cache_dir, seed=int(seed))


def _land_mask(size):
    """Land fraction (H, W) in [0, 1] from the coastline texture, row 0 = north."""
    from .astro import RE
    from .coastlines import coastlines_xyz
    from .earth import cached_earth_image
    img = cached_earth_image(coastlines_xyz(RE)).getchannel('R').resize(size, Image.BILINEAR)
    return np.asarray(img, float) / 255.0


def _rotate(P, axis, ang):
    """Rodrigues rotation of unit vectors P (..., 3) about unit `axis` by `ang` (...)."""
    c, s = np.cos(ang)[..., None], np.sin(ang)[..., None]
    k = np.asarray(axis, float)
    return P * c + np.cross(k, P) * s + k * (P @ k)[..., None] * (1 - c)


def _unit(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def bake_weather(size=WEATHER_SIZE, seed=3, land=None):
    """Global cloud weather map (equirectangular, row 0 = north, ECEF longitude).

    R coverage, G cloud type (0 stratus .. 1 deep convection), B cirrus.
    Latitude bands (broken ITCZ clusters, subtropical clear belts, storm
    tracks), domain-warped zonally stretched fbm, extratropical cyclones with
    spiral arms, tropical cyclones with eyes, clear subtropical deserts and
    marine stratocumulus decks off the subtropical west coasts. Built at half
    resolution (the 3D noise atlas adds the fine detail) and upsampled.
    """
    W, H = size[0] // 2, size[1] // 2
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    lon = (np.arange(W) + 0.5) / W * 360.0 - 180.0
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    alat = np.abs(LA)
    P = np.stack([np.cos(np.radians(LA)) * np.cos(np.radians(LO)),
                  np.cos(np.radians(LA)) * np.sin(np.radians(LO)), np.sin(np.radians(LA))], -1)
    if land is None:
        land = _land_mask((W, H))
    else:
        land = np.asarray(Image.fromarray(np.float32(land)).resize((W, H), Image.BILINEAR), float)
    rng = np.random.RandomState(seed)

    # cyclones: (lat, lon, radius rad, swirl rad, coverage boost, tropical?)
    cyc = []
    for hemi in (1, -1):
        for k in range(6):
            cyc.append((hemi * rng.uniform(38, 62), rng.uniform(-180, 180), rng.uniform(0.13, 0.22),
                        hemi * rng.uniform(2.2, 3.4), rng.uniform(0.20, 0.35), False))
        for k in range(2 if hemi > 0 else 1):
            cyc.append((hemi * rng.uniform(13, 22), rng.uniform(-180, 180), rng.uniform(0.09, 0.12),
                        hemi * rng.uniform(5.0, 6.5), 0.0, True))
    Q = P.copy()
    boost = np.zeros(LA.shape)
    eyes = np.zeros(LA.shape)
    tc = np.zeros(LA.shape)
    for la, lo, R, S, B, trop in cyc:
        C = _unit(la, lo)
        ang = np.arccos(np.clip(P @ C, -1, 1))
        Q = _rotate(Q, C, S * np.exp(-(ang / R) ** 1.3))       # spiral arms in the noise domain
        if trop:
            eyes = np.maximum(eyes, np.exp(-(ang / (0.06 * R)) ** 2))
            tc = np.maximum(tc, np.exp(-(ang / R) ** 2))
        else:
            boost += B * np.exp(-(ang / (0.75 * R)) ** 2)
    # zonal stretch, then domain warp
    q = Q * np.array([1.0, 1.0, 1.7]) * 2.4
    wv = np.stack([fbm_noise(q * 0.9 + o, 3, seed=seed + j) for j, o in
                   enumerate((np.array([1.7, 9.2, 3.1]), np.array([8.3, 2.8, 5.5]), np.array([4.4, 7.1, 0.6])))], -1)
    n = fbm_noise(q + 1.1 * wv, 6, seed=seed + 11)                # about [-0.6, 0.6]
    cl = fbm_noise(P * 9.0, 3, seed=seed + 31)                    # convective clusters

    def band(c, w):
        return np.exp(-((alat - c) / w) ** 2)
    itcz = np.exp(-((LA - 6.0) / 6.5) ** 2)
    base = (0.40 + itcz * (0.10 + 0.9 * np.maximum(cl, 0.0)) - 0.32 * band(24.0, 8.0)
            + 0.30 * band(52.0, 11.0) + 0.10 * (1.0 - np.exp(-np.maximum(alat - 62.0, 0.0) / 6.0)))

    desert = land * band(24.0, 8.5)
    east = np.zeros_like(land)
    step = max(1, W // 256)
    for k in range(step, 14 * step, step):                       # land within ~10 deg east
        east = np.maximum(east, np.roll(land, -k, axis=1) * (1 - k / (20.0 * step)))
    deck = (1 - land) * (east > 0.4) * band(22.0, 9.0)
    deck = np.asarray(Image.fromarray(np.uint8(np.clip(deck, 0, 1) * 255)).filter(
        ImageFilter.GaussianBlur(4)), float) / 255.0
    deck = np.clip(deck * 1.3, 0.0, 1.0)

    cov = base + 0.95 * n + boost - 0.60 * desert + 0.50 * deck
    cov = cov + 0.20 * land * np.exp(-(LA / 9.0) ** 2)            # rainforest convection
    cov = np.clip((cov - 0.30) / 0.48, 0.0, 1.0) ** 1.2
    spiral = np.clip(0.80 + 1.8 * n, 0.0, 1.0)                    # arms from the swirled noise
    cov = np.maximum(cov, tc * spiral * np.clip(1.6 * tc, 0, 1))
    cov = cov * (1.0 - 0.97 * eyes)

    typ = 0.40 + 0.30 * n + 0.50 * itcz * np.clip(0.4 + 1.5 * cl, 0.0, 1.0)
    typ = typ - 0.35 * deck + 0.7 * tc - 0.15 * np.clip((alat - 60.0) / 10.0, 0.0, 1.0)
    typ = np.clip(typ, 0.02, 1.0)
    # cirrus: anvil outflow from the ITCZ and tropical cyclones, jet-stream cirrus on the
    # storm tracks (the shader draws fibrous streaks from this amount)
    jetband = band(45.0, 10.0)
    cir = np.clip(0.9 * itcz * np.clip(cl * 2.0, 0.0, 1.0) + 0.8 * tc + 0.7 * jetband * np.clip(n * 2.5, 0.0, 1.0)
                  + 0.5 * boost, 0.0, 1.0)
    cir = np.clip((cir - 0.25) / 0.6, 0.0, 1.0)

    def up(x):
        im = Image.fromarray(np.float32(x)).resize(tuple(size), Image.BICUBIC)
        return np.clip(np.asarray(im, float), 0.0, 1.0)
    return to_rgb8(np.stack([up(cov), up(typ), up(cir)], -1))


def weather_lut(seed=3, cache_dir=None):
    return cached_lut('weather', bake_weather, cache_dir=cache_dir, size=WEATHER_SIZE, seed=int(seed))
