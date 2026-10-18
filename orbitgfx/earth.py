"""Bake the equirectangular Earth texture the shaders sample.

Channels: R = land mask, G = coastline stroke, B = city lights.
The texture is RGB only: alpha would be premultiplied by iOS on load and
corrupt the data channels.
"""
import hashlib
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .astro import latlon

# Bump whenever earth_image() output changes, so stale cached textures are ignored.
TEXTURE_VERSION = 6          # 3: 2.0 transmittance LUT (luts.py); 4: default atmo_scale 1.5; 5: clouds; 6: B = cirrus


def stitch_rings(paths, tol_km=1.0):
    """Join open coastline pieces whose endpoints meet into longer chains.

    Natural Earth stores large continents as several open LineStrings; they
    must be joined into closed rings before they can be filled as land.
    Works in Cartesian space, so pieces meeting at the date line join too.
    Pieces may be reversed to connect. Already-closed rings pass through.
    """
    out, pool = [], []
    for p in paths:
        p = np.asarray(p, float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
            continue
        (out if np.linalg.norm(p[0] - p[-1]) < tol_km and len(p) > 3 else pool).append(p)
    while pool:
        chain = pool.pop(0)
        grown = True
        while grown and np.linalg.norm(chain[0] - chain[-1]) >= tol_km:
            grown = False
            for i, q in enumerate(pool):
                if np.linalg.norm(chain[-1] - q[0]) < tol_km:
                    chain = np.vstack((chain, q[1:]))
                elif np.linalg.norm(chain[-1] - q[-1]) < tol_km:
                    chain = np.vstack((chain, q[::-1][1:]))
                elif np.linalg.norm(chain[0] - q[-1]) < tol_km:
                    chain = np.vstack((q[:-1], chain))
                elif np.linalg.norm(chain[0] - q[0]) < tol_km:
                    chain = np.vstack((q[::-1][:-1], chain))
                else:
                    continue
                pool.pop(i)
                grown = True
                break
        out.append(chain)
    return out


def _closes(xyz, tol_km=1.0, gap_frac=0.02, min_pts=20):
    """Treat a chain as a ring if its ends meet, or if the remaining gap is a
    tiny fraction of a long chain (e.g. Afro-Eurasia's 131 km date-line gap)."""
    gap = float(np.linalg.norm(xyz[0] - xyz[-1]))
    if gap < tol_km:
        return len(xyz) > 3
    length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    return len(xyz) >= min_pts and gap < gap_frac * length


def earth_image(coasts_xyz, width=2048, ss=2, fill_land=True, light_seed=11):
    """Build the data texture from coastline polylines (each an (N, 3) array).

    width:     texture width; height is width / 2.
    ss:        supersampling factor for anti-aliased land / coast.
    fill_land: fill closed rings as land (open pieces are stitched first;
               circumpolar rings are closed via the pole).
    """
    W, H = width * ss, width * ss // 2
    land = Image.new('L', (W, H), 0)
    coast = Image.new('L', (W, H), 0)
    dl, dc = ImageDraw.Draw(land), ImageDraw.Draw(coast)

    def px(lon, lat):
        return (lon + 180.0) / 360.0 * W, (90.0 - lat) / 180.0 * H

    for xyz in stitch_rings(coasts_xyz):
        lat, lon = latlon(xyz)
        lon_u = np.degrees(np.unwrap(np.radians(lon)))
        closed = _closes(xyz)
        net = lon_u[-1] - lon_u[0]
        for sh in (-360.0, 0.0, 360.0):
            pts = [px(a + sh, b) for a, b in zip(lon_u, lat)]
            dc.line(pts, fill=255, width=3 * ss // 2 + 1)
            if fill_land and closed:
                if abs(net) > 180.0:          # circumpolar ring (Antarctica)
                    pole = -90.0 if np.mean(lat) < 0 else 90.0
                    pts = pts + [px(lon_u[-1] + sh, pole), px(lon_u[0] + sh, pole)]
                dl.polygon(pts, fill=255)
    land = land.resize((width, width // 2), Image.LANCZOS)
    coast = coast.resize((width, width // 2), Image.LANCZOS)

    # City lights: clustered speckle + sprawl glow, mostly coastal, |lat| < 62.
    L = np.asarray(land, float) / 255.0
    near = np.asarray(coast.filter(ImageFilter.GaussianBlur(12)), float)
    near = near / max(1.0, near.max())
    h = width // 2
    lat = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    band = (np.abs(lat) < 62)[:, None] * 1.0
    rng = np.random.RandomState(light_seed)
    lo_res = Image.fromarray(np.uint8(rng.rand(24, 48) * 255)).resize((width, h), Image.BICUBIC)
    cluster = (np.asarray(lo_res, float) / 255.0) ** 3
    cluster /= max(1e-6, cluster.max())
    p = (0.16 * near ** 0.7 + 0.004) * (L > 0.5) * band * (0.15 + 1.6 * cluster)
    dots = (rng.rand(*L.shape) < p) * (0.30 + 0.70 * rng.rand(*L.shape) ** 2)
    glow = np.asarray(Image.fromarray(np.uint8(np.clip(dots, 0, 1) * 255))
                      .filter(ImageFilter.GaussianBlur(3)), float) / 255.0
    li = np.clip(dots * 1.15 + glow * 2.2, 0.0, 1.0)
    li = Image.fromarray(np.uint8(li * 255))
    return Image.merge('RGB', (land, coast, li))


def cached_earth_image(coasts_xyz, width=2048, cache_dir=None, **kw):
    """earth_image() with a PNG cache keyed on the input data and options.

    Saves ~1-3 s of launch time on a phone after the first run.
    cache_dir defaults to a `.orbitgfx_cache` folder next to this package.
    """
    coasts = [np.asarray(c, float) for c in coasts_xyz]
    hsh = hashlib.sha1(repr((TEXTURE_VERSION, width, sorted(kw.items()))).encode())
    for c in coasts:
        hsh.update(np.ascontiguousarray(c[:, :3], dtype=np.float32).tobytes())
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '.orbitgfx_cache')
    path = os.path.join(cache_dir, 'earth_%d_%s.png' % (width, hsh.hexdigest()[:12]))
    if os.path.exists(path):
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            pass
    img = earth_image(coasts, width=width, **kw)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        img.save(path)
    except Exception:
        pass
    return img


def pil_texture(img):
    """PIL image -> scene.Texture (via ui.Image)."""
    import scene
    import ui
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return scene.Texture(ui.Image.from_data(buf.getvalue()))
