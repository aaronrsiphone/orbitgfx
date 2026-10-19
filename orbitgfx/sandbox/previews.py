"""Contact sheet of the globe at the plan's sun geometries, per quality.

    python -m orbitgfx.sandbox.previews -o sheet.png [--scale 3] [--cols LEGACY,LOW,MED]

Rows: noon, terminator, backlit, night (sun angle from the view direction:
20, 90, 172, 130 degrees) and a chase view at zoom 2.6. Columns: presets,
optionally with suffixes:  +T = 1.x look tone-mapped (no SCATTER),  /P = perspective.
Panel size is the iPhone 16 Pro Max portrait globe panel (410 x 552 pt).
"""
import argparse
import math
import os
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from orbitgfx.sandbox.runner import Runner  # noqa: E402,F401  (installs the shim)
import scene  # noqa: E402

from orbitgfx import FEATURES, cached_earth_image, pil_texture  # noqa: E402
from orbitgfx.astro import RE  # noqa: E402
from orbitgfx.coastlines import coastlines_xyz  # noqa: E402
from orbitgfx.views import Camera, GlobeView  # noqa: E402

PANEL = (410.0, 552.0)
BG = (0.023, 0.039, 0.078)
GEOMETRIES = (('noon', 20.0), ('terminator', 90.0), ('backlit', 172.0), ('night', 130.0))
_TEX = []


def _texture():
    if not _TEX:
        _TEX.append(pil_texture(cached_earth_image(coastlines_xyz(RE))))
    return _TEX[0]


def _orbit(u=0.9, inc=51.6, raan=0.4, alt=500.0):
    i = math.radians(inc)
    r = RE + alt
    p = np.array([math.cos(u), math.sin(u), 0.0]) * r
    v = np.array([-math.sin(u), math.cos(u), 0.0]) * math.sqrt(398600.4418 / r)
    Rx = np.array([[1, 0, 0], [0, math.cos(i), -math.sin(i)], [0, math.sin(i), math.cos(i)]])
    Rz = np.array([[math.cos(raan), -math.sin(raan), 0], [math.sin(raan), math.cos(raan), 0], [0, 0, 1]])
    return Rz @ Rx @ p, Rz @ Rx @ v


class Sheet(scene.Scene):
    pass


def render_one(quality, sun_deg, chase=False, scale=3.0, u_time=40.0, **globe_kw):
    """One panel as a PIL RGB image (composited over the panel background)."""
    from PIL import Image
    scene.SCALE = scale
    tone = '+T' in quality
    persp = '/P' in quality
    q = quality.replace('+T', '').replace('/P', '')
    sat, vel = _orbit()
    if chase:
        cam = Camera(az=math.atan2(sat[1], sat[0]) - 0.35 / 2.6,
                     el=math.asin(sat[2] / np.linalg.norm(sat)) + 0.30 / 2.6, zoom=2.6)
    else:
        cam = Camera(az=0.55, el=0.42)
    M = cam.matrix()
    fwd, up = M[2], M[1]
    a = math.radians(sun_deg)
    sun = fwd * math.cos(a) + up * math.sin(a) * 0.35 + M[0] * math.sin(a)
    sun /= np.linalg.norm(sun)
    host = Sheet()
    host._fw_set(size=scene.Size(*PANEL), bounds=scene.Rect(0, 0, *PANEL))
    host.background_color = BG
    feats = None
    if tone and q != 'LEGACY':                 # +T: the 1.x look, tone-mapped (no SCATTER)
        feats = list(FEATURES) + ['TONEMAP']
    gv = GlobeView(host, _texture(), features=feats, quality=q,
                   perspective=persp, fill=0.44, inset=0.0, **globe_kw)
    gv.layout((0.0, 0.0) + PANEL)
    st = [((RE * 0.62, RE * 0.55, RE * 0.56), True), ((-RE * 0.4, RE * 0.75, RE * 0.52), False)]
    gv.update(cam, sun, 1.3, sat=sat, vel=vel, trail_rad=0.8, stations=st, cloud_hours=3.0)
    snap = scene.render(host, None, u_time=u_time)
    import skia
    arr = snap.toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3]
    return Image.fromarray(np.ascontiguousarray(arr))


def sheet(cols, scale=3.0, thumb=0.5, rows=None):
    from PIL import Image, ImageDraw
    rows = list(rows or GEOMETRIES) + [('chase 2.6x', 20.0, True)]
    tiles = []
    for row in rows:
        deg = row[1]
        chase = len(row) > 2
        tiles.append([render_one(c, deg, chase, scale) for c in cols])
    tw = int(PANEL[0] * scale * thumb)
    th = int(PANEL[1] * scale * thumb)
    lab = 28
    out = Image.new('RGB', (110 + len(cols) * (tw + 6), lab + len(rows) * (th + 6)), (12, 14, 20))
    d = ImageDraw.Draw(out)
    for j, c in enumerate(cols):
        d.text((110 + j * (tw + 6) + 6, 8), c, fill=(220, 230, 240))
    for i, row in enumerate(rows):
        d.text((8, lab + i * (th + 6) + th // 2), row[0], fill=(220, 230, 240))
        for j, im in enumerate(tiles[i]):
            out.paste(im.resize((tw, th), Image.LANCZOS), (110 + j * (tw + 6), lab + i * (th + 6)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='sheet.png')
    ap.add_argument('--cols', default='LEGACY,LOW,MED,LOW+T,LOW/P')
    ap.add_argument('--scale', type=float, default=3.0)
    ap.add_argument('--thumb', type=float, default=0.5)
    a = ap.parse_args(argv)
    sheet(a.cols.split(','), a.scale, a.thumb).save(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
