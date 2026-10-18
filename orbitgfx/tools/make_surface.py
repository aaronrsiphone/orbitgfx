"""Bake the 2.0 surface textures (phase 5) from source imagery, offline.

    python -m orbitgfx.tools.make_surface BLUE_MARBLE BLACK_MARBLE HEIGHT [--width 4096] [--aux-width 2048]

BLUE_MARBLE   NASA Blue Marble Next Generation, e.g. world.topo.200403.3x5400x2700 (March)
BLACK_MARBLE  NASA Black Marble 2016, grey, any resolution (area-averaged down)
HEIGHT        8-bit land elevation map, equirectangular (0 = sea level, ~quadratic in
              height, 242 = Everest), e.g. three-globe's example/img/earth-topology.png

Writes orbitgfx/data/surface/color.jpg  (RGB: land colour, sea ice painted on the water)
       orbitgfx/data/surface/earth.png  (the ray core's data texture, same layout as
                                         earth.earth_image: R land, G coastline stroke,
                                         B city lights; replaces the 1.x one on SURFACE)
       orbitgfx/data/surface/aux.png    (R height sqrt-encoded with 0 = water,
                                         G sea-ice concentration)
Blue Marble shows no sea ice (its water is a flat colour), so March-like ice extents
are drawn here: the Arctic basin, Hudson and Baffin Bays, the Canadian archipelago,
the East Greenland current, Barents/Kara, Bering and Okhotsk Seas, the Baltic's
northern gulfs, and the Antarctic coastal pack near its March minimum.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None          # Black Marble 500 m is 13500 x 6750

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'data', 'surface')
H_MAX = 8.85            # km at the height map's Everest value
H_REF = 242.0
H_ENC = 9.0             # stored as sqrt(h / H_ENC)
WATER_RGB = np.array((2.0, 5.0, 20.0))

# northern ice edge (latitude) by longitude, March climatology, sketched from NSIDC maps
N_EDGE = ((-180, 61), (-172, 61), (-165, 59.5), (-160, 58), (-156, 59), (-152, 71.5), (-140, 70.5),
          (-125, 69.5), (-100, 54), (-82, 51), (-70, 54), (-62, 52), (-56, 54), (-52, 60), (-48, 63),
          (-45, 59.5), (-40, 62), (-33, 65.5), (-25, 69), (-18, 72), (-10, 75), (0, 77.5),
          (8, 79.5), (15, 78), (25, 77), (35, 75.5), (45, 71), (55, 69.5), (70, 66), (100, 66),
          (130, 66), (136, 60), (138, 46), (148, 45.5), (155, 52), (159, 57), (163, 59),
          (170, 61), (180, 61))
# southern ice edge (latitude) by longitude, March (near the yearly minimum)
S_EDGE = ((-180, -74.5), (-160, -75.5), (-140, -73), (-120, -72), (-100, -71), (-80, -70), (-62, -64),
          (-50, -63.5), (-35, -66), (-20, -69.5), (0, -69.5), (40, -67.5), (80, -65.5), (120, -65.5),
          (150, -67), (165, -71.5), (180, -74.5))
# extra boxes (lat0, lat1, lon0, lon1): Baltic gulfs, Gulf of St Lawrence, northern Caspian, Bohai
BOXES = ((60.2, 66.5, 17.0, 26.5), (59.5, 61.0, 22.0, 30.5), (46.0, 50.0, -66.5, -59.0),
         (45.0, 47.5, 46.5, 53.5), (38.5, 41.0, 117.5, 122.0))


def fbm(shape, seed, octaves=6, base=8):
    """Seamless-ish value noise in [0, 1] (bicubic upsampled random fields)."""
    rng = np.random.RandomState(seed)
    H, W = shape
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        h, w = base * 2 ** o // 2, base * 2 ** o
        f = rng.rand(h, w)
        f = np.concatenate([f, f[:, :3]], axis=1)          # wrap in longitude
        im = Image.fromarray(np.uint8(f * 255)).resize((W + int(W * 3 / w), H), Image.BICUBIC)
        out += amp * np.asarray(im, float)[:, :W] / 255.0
        tot += amp
        amp *= 0.55
    return out / tot


def sea_ice(lat, lon, water, W, H):
    n1, n2 = fbm((H, W), 1), fbm((H, W), 2, base=16)
    n3, n4 = fbm((H, W), 3, base=128, octaves=4), fbm((H, W), 5, base=48, octaves=5)
    # edge wobble: large bays and tongues plus a ragged small-scale fringe (degrees)
    jitter = (n1 - 0.5) * 4.0 + (n2 - 0.5) * 1.5 + (n3 - 0.5) * 1.6
    en = np.interp(lon, [p[0] for p in N_EDGE], [p[1] for p in N_EDGE])
    es = np.interp(lon, [p[0] for p in S_EDGE], [p[1] for p in S_EDGE])
    cn = np.clip((lat - en + jitter) / 0.7, 0.0, 1.0)
    cs = np.clip((es - lat + jitter) / 0.7, 0.0, 1.0)
    cb = np.zeros_like(lat)
    for la0, la1, lo0, lo1 in BOXES:
        d = np.minimum.reduce([lat - la0, la1 - lat, (lon - lo0) * np.cos(np.radians(lat)),
                               (lo1 - lon) * np.cos(np.radians(lat))])
        cb = np.maximum(cb, np.clip((d + (n3 - 0.5) * 0.8) / 0.5, 0.0, 1.0))
    c = np.maximum.reduce([cn, cs, cb])
    # sparse leads (thin cracks of open water) and a slightly lower pack concentration
    leads = np.clip(1.0 - np.abs(n4 - 0.5) / 0.006, 0.0, 1.0) * (n2 > 0.45)
    c = c * (0.94 + 0.06 * n2) * (1.0 - 0.45 * leads)
    return np.clip(c, 0.0, 1.0) * water


def bake(blue, black, height, width=4096, aux_width=2048):
    W, H = width, width // 2
    bm = Image.open(blue).convert('RGB')
    a = np.asarray(bm, float)
    water_hi = (np.abs(a - WATER_RGB).sum(-1) < 24.0).astype(np.uint8) * 255
    water = np.asarray(Image.fromarray(water_hi).resize((W, H), Image.BOX), float) / 255.0
    col = np.asarray(bm.resize((W, H), Image.LANCZOS), float) / 255.0

    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    lon = (np.arange(W) + 0.5) / W * 360.0 - 180.0
    LON, LAT = np.meshgrid(lon, lat)
    # ice composited over a slightly grown water mask, so the coast's dark fringe goes
    wgrow = np.asarray(Image.fromarray(np.uint8(water * 255)).filter(ImageFilter.MaxFilter(5)), float) / 255.0
    ice = sea_ice(LAT, LON, wgrow, W, H)
    tone = 0.90 + 0.10 * fbm((H, W), 4, base=128)
    ice_rgb = np.stack([0.90 * tone, 0.93 * tone, 0.96 * tone], -1)
    col = col * (1.0 - ice[..., None]) + ice_rgb * ice[..., None]
    Image.fromarray(np.uint8(np.clip(col, 0, 1) * 255 + 0.5)).save(
        os.path.join(OUT, 'color.jpg'), quality=90, subsampling=0)

    # height (km), zero over water; stored sqrt-encoded with a land floor so G > 0 means land
    land = 1.0 - water
    AW, AH = aux_width, aux_width // 2
    t = np.asarray(Image.open(height).convert('L').resize((AW, AH), Image.BICUBIC), float)
    h = H_MAX * (t / H_REF) ** 2
    landa = np.asarray(Image.fromarray(np.uint8(land * 255)).resize((AW, AH), Image.BOX), float) / 255.0
    icea = np.asarray(Image.fromarray(np.uint8(ice * water * 255)).resize((AW, AH), Image.BOX), float) / 255.0
    g = np.sqrt(np.clip(h / H_ENC, 0.0, 1.0))
    g = np.where(landa > 0.5, np.maximum(g, 3.0 / 255.0), 0.0) * np.clip(landa * 2.0, 0.0, 1.0)

    # lights: area-average, normalise on a high percentile
    li = Image.open(black).convert('L')
    li = np.asarray(li.resize((W, H), Image.BOX), float)
    ref = np.percentile(li[li > 0], 99.95) if (li > 0).any() else 255.0
    li = np.clip(li / ref, 0.0, 1.0) ** 0.7 * land            # lift small towns a little
    aux = np.stack([g, icea, np.zeros_like(g)], -1)
    Image.fromarray(np.uint8(aux * 255 + 0.5)).save(os.path.join(OUT, 'aux.png'))

    # coastline stroke from the same water mask: smooth contour at 4x (bilinear upsample of
    # the soft mask, then threshold), ~1.5 texels wide, box-filtered back down (anti-aliased)
    soft = Image.fromarray(water_hi).filter(ImageFilter.GaussianBlur(0.8))
    hi = soft.resize((W * 4, H * 4), Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    edge = np.asarray(hi.filter(ImageFilter.MaxFilter(5)), float) - np.asarray(hi.filter(ImageFilter.MinFilter(3)), float)
    coast = np.asarray(Image.fromarray(np.uint8(edge)).resize((W, H), Image.BOX), float) / 255.0
    coast = np.clip(coast * 1.6, 0.0, 1.0)
    earth = np.stack([land, coast, li], -1)
    Image.fromarray(np.uint8(np.clip(earth, 0, 1) * 255 + 0.5)).save(os.path.join(OUT, 'earth.png'))
    return col, aux


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('blue')
    ap.add_argument('black')
    ap.add_argument('height')
    ap.add_argument('--width', type=int, default=4096)
    ap.add_argument('--aux-width', type=int, default=2048)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    bake(a.blue, a.black, a.height, a.width, a.aux_width)
    for f in ('color.jpg', 'earth.png', 'aux.png'):
        print(os.path.join(OUT, f), os.path.getsize(os.path.join(OUT, f)) // 1024, 'KB')


if __name__ == '__main__':
    main()
