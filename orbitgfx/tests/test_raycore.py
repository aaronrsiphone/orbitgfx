"""orbitgfx 2.0 phase 1 checks: ray core vs LEGACY, presets, passes, NaN scan.

    python -m orbitgfx.tests.test_raycore
Desktop only (sandbox moderngl + skia). Writes diagnostics to tests/out/.
"""
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(PKG))
from orbitgfx.sandbox.runner import Runner  # noqa: E402,F401  (installs the shim)
import scene  # noqa: E402

from orbitgfx import luts  # noqa: E402
from orbitgfx.astro import RE  # noqa: E402
from orbitgfx.shaders import (FEATURES, PASSES, RAY_FEATURES, build_globe_shader,  # noqa: E402
                              build_ray_shader, lint_glsl)
from orbitgfx.views import QUALITY, Camera, GlobeView  # noqa: E402
from orbitgfx.tests._gpu import gpu, run_all  # noqa: E402

OUT = os.path.join(HERE, 'out')
PANEL = (10.0, 10.0, 410.0, 552.0)            # the iPhone 16 Pro Max portrait panel
RED = {'CYAN': (1, 0, 0), 'ECLC': (1, 0, 0), 'BGC': (1, 0, 0),
       'NIGHT': (0, 0, 0.3), 'DEEP': (0, 0, 0.3), 'OCEAN': (0, 0, 0.3), 'COAST': (0, 0, 0.3)}


class _Host(scene.Scene):
    pass


def _texture():
    from PIL import Image
    img = Image.new('RGB', (64, 32), (0, 0, 0))
    return scene.Texture(img)


def orbit(inc_deg=51.6, raan=0.3, u=0.8, alt=500.0):
    """Circular orbit: ECI position (km) and velocity (km/s)."""
    i, O = math.radians(inc_deg), raan
    r = RE + alt
    p = np.array([math.cos(u), math.sin(u), 0.0]) * r
    v = np.array([-math.sin(u), math.cos(u), 0.0]) * math.sqrt(398600.4418 / r)
    Rx = np.array([[1, 0, 0], [0, math.cos(i), -math.sin(i)], [0, math.sin(i), math.cos(i)]])
    Rz = np.array([[math.cos(O), -math.sin(O), 0], [math.sin(O), math.cos(O), 0], [0, 0, 1]])
    M = Rz @ Rx
    return M @ p, M @ v


def view(quality='LEGACY', features=None, palette=None, cam=None, sun=(0.6, -0.5, 0.62),
         sat=None, vel=None, render_scale=None, perspective=False, debug=(), trail=0.7, **kw):
    """Render one GlobeView; returns premultiplied RGBA float arrays per pass and the view."""
    host = _Host()
    gv = GlobeView(host, _texture(), features=features, palette=palette, quality=quality,
                   render_scale=render_scale, perspective=perspective, fill=0.44, debug=debug, **kw)
    gv.layout(PANEL)
    cam = cam or Camera(az=0.7, el=0.45)
    sun = np.asarray(sun, float) / np.linalg.norm(sun)
    if sat is None:
        sat, vel = orbit()
    gv.update(cam, sun, 1.1, sat=sat, vel=vel, trail_rad=trail,
              stations=[((RE * 0.5, RE * 0.6, RE * 0.62), True)], cloud_hours=2.0)
    out = {'overlay': scene._render_sprite(gv.node, 3.0).astype(float) / 255.0}
    if gv.rad_node is not None:
        out['radiance'] = scene._render_sprite(gv.rad_node, 3.0).astype(float) / 255.0
    return out, gv


def _pts(mask):
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], 1).astype(float)


def hausdorff(a, b, chunk=2048):
    """Symmetric Hausdorff distance (px) between two boolean masks."""
    A, B = _pts(a), _pts(b)
    if len(A) == 0 or len(B) == 0:
        return float('inf')

    def directed(P, Q):
        worst = 0.0
        for i in range(0, len(P), chunk):
            d = ((P[i:i + chunk, None, :] - Q[None, :, :]) ** 2).sum(-1)
            worst = max(worst, float(np.sqrt(d.min(1)).max()))
        return worst
    return max(directed(A, B), directed(B, A))


def ring_mask(img):
    return (img[..., 0] > 0.35) & (img[..., 1] < 0.3)


def disc_mask(img):
    return img[..., 3] > 0.5


def edge(m):
    """Boundary pixels of a mask (the panel border counts as a boundary)."""
    p = np.pad(m, 1, constant_values=False)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


POSES = {
    'default': dict(cam=Camera(az=0.7, el=0.45)),
    'polar': dict(cam=Camera(az=-1.2, el=1.3)),
}


def _edge_on():
    sat, vel = orbit(inc_deg=51.6, raan=0.3, u=0.8)
    e2 = np.cross(np.cross(sat, vel), sat)
    e2 /= np.linalg.norm(e2)
    return dict(cam=Camera(az=math.atan2(e2[1], e2[0]), el=math.asin(e2[2])), sat=sat, vel=vel)


POSES['edge_on'] = _edge_on()
_s, _ = orbit()
POSES['chase2.6'] = dict(cam=Camera(az=math.atan2(_s[1], _s[0]) - 0.15,
                                    el=math.asin(_s[2] / np.linalg.norm(_s)) + 0.12, zoom=2.6))


# --------------------------------------------------------------------------- tests
@gpu(what='render')
def test_legacy_pixel_identical():
    """tests/baseline was rendered by 1.x; LEGACY must match it exactly (max diff 0)."""
    from orbitgfx.tests import test_mission
    report = test_mission.test_regression_against_baseline()
    assert report and all(r[1] == 0 for r in report), report
    return report


def _ray_variants():
    for pass_ in PASSES:
        for feats in (None, RAY_FEATURES, (), ('ORBIT', 'SATELLITE'), ('ATMOSPHERE', 'TONEMAP'),
                      ('ORBIT', 'REFRACTION'), ('ATMOSPHERE', 'REFRACTION')):
            for dbg in ((), ('NANCHECK',)):
                yield build_ray_shader(feats, {'CYAN': (1, 0, 0)}, pass_, dbg)


def test_variants_lint():
    """Static text checks only; compiling is test_variants_compile."""
    n = 0
    for src in _ray_variants():
        assert not lint_glsl(src), lint_glsl(src)
        n += 1
    assert build_globe_shader() == build_globe_shader(FEATURES)
    return '%d ray variants' % n


@gpu
def test_variants_compile():
    """Desktop GLSL 120 compile (not Pythonista's compiler: that needs the device)."""
    n = 0
    for src in _ray_variants():
        assert scene.Shader(src).compiled
        n += 1
    return '%d ray variants' % n


def test_shader_heights_use_one_unit():
    """Every radius-to-km conversion goes through altitudeKm(): the drawn planet is
    6378.137 / atmo_scale km per unit, so a literal Earth radius would read a known
    altitude 1.5x too high at the default atmo_scale (weaker refraction)."""
    import re
    from orbitgfx import shaders
    src = build_ray_shader(RAY_FEATURES, None, 'single')
    code = re.sub(r'//[^\n]*', '', src)
    assert not re.search(r'\b6371(\.0)?\b', code)
    assert 'float altitudeKm(float r)' in src and '#define KM_PER_UNIT u_atm.y' in src
    for name in ('RAY_COMMON_GLSL', 'ATMOS_GLSL', 'RAY_MAIN'):
        body = re.sub(r'//[^\n]*', '', getattr(shaders, name))
        # heights from the drawn radius: only via altitudeKm (true-scale orbit uses RE_KM)
        assert not re.search(r'-\s*1\.0\)\s*\*\s*(u_atm\.y|6378|6371)', body), name
    for line in src.splitlines():
        if 'refrDeg(' in line and '=' in line and 'float refrDeg' not in line:
            assert re.search(r'refrDeg\([^,]+,\s*(h|hk)\)', line), line


@gpu
def test_refraction_height_follows_atmo_scale():
    """A known physical altitude stays the same km whatever atmo_scale draws: the
    shader's altitudeKm() is run on the GPU for points h km up on each drawn planet."""
    import re
    from orbitgfx.luts import planet
    src = build_ray_shader(RAY_FEATURES, None, 'single')
    units = re.search(r'// Units:.*?float altitudeKm\(float r\) \{[^}]*\}', src, re.S).group(0)
    probe = ('precision highp float;\n#define SCATTER\nuniform vec4 u_atm;\nuniform float u_r;\n'
             + units + '\nvoid main() { gl_FragColor = vec4(altitudeKm(u_r) / 50.0, 0.0, 0.0, 1.0); }\n')
    node = scene.SpriteNode(None, size=(4, 4))
    node.shader = scene.Shader(probe)
    rep = {}
    for scale in (1.0, 1.5, 3.0):
        rg, rt = planet(scale, RE)
        node.shader.set_uniform('u_atm', (rt / rg, rg, 1.0, 0.0))
        for h in (2.0, 20.0, 40.0):
            node.shader.set_uniform('u_r', 1.0 + h / rg)
            got = scene._render_sprite(node, 0.0)[1, 1, 0] / 255.0 * 50.0
            assert abs(got - h) < 0.5, (scale, h, got)
            rep[(scale, h)] = round(float(got), 1)
    return rep


@gpu
def test_presets_build():
    sizes = {}
    for q in QUALITY:
        img, gv = view(q)
        assert gv.quality == q
        sizes[q] = (gv.render_scale, sorted(img))
        if QUALITY[q].get('render_scale', 1.0) < 1.0:
            rs = QUALITY[q]['render_scale']
            assert gv.effect is not None and abs(gv.rad_node.size[0] - (PANEL[2] - 2.0) * rs) < 1e-6
    return sizes


@gpu
def test_ring_and_disc_match_legacy():
    """Gate: ray core ring and disc within 1 px of LEGACY (3x pixels).

    Edge-on is excluded: there LEGACY drops half of the near side of the ring
    (its Newton solve picks one of two overlapping sheets); the ray core draws
    both, checked against the exact projection below.
    """
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    for name, pose in POSES.items():
        if name == 'edge_on':
            continue
        kw = dict(pose, trail=0.0)
        ring_l = ring_mask(view('LEGACY', ('ORBIT',), RED, **kw)[0]['overlay'])
        ring_r = ring_mask(view('LOW', ('ORBIT',), RED, **kw)[0]['overlay'])
        disc_l = disc_mask(view('LEGACY', (), RED, **kw)[0]['overlay'])
        disc_r = disc_mask(view('LOW', (), RED, **kw)[0]['overlay'])
        hr, hd = hausdorff(ring_l, ring_r), hausdorff(edge(disc_l), edge(disc_r))
        rep[name] = (round(hr, 2), round(hd, 2), int(ring_l.sum()), int((ring_l != ring_r).sum()))
        assert ring_l.sum() > 100, (name, 'no ring drawn')
        # 1 px including a diagonal step on anti-aliased edges (sqrt 2)
        assert hr <= 1.415 and hd <= 1.415, (name, rep[name])
    return rep


@gpu
def test_perspective_disc_size():
    """A pinhole camera at the auto distance shows the same apparent Earth radius."""
    rep = {}
    for zoom in (1.0, 2.6):
        cam = Camera(az=0.7, el=0.45, zoom=zoom)
        a = disc_mask(view('LOW', (), RED, cam=cam)[0]['overlay']).sum()
        b = disc_mask(view('LOW', (), RED, cam=cam, perspective=True)[0]['overlay']).sum()
        ra, rb = math.sqrt(a / math.pi), math.sqrt(b / math.pi)
        rep[zoom] = (round(ra, 1), round(rb, 1))
        assert abs(ra - rb) <= 1.0, rep
    return rep


@gpu
def test_no_nan_any_preset_pass_pose():
    bad = {}
    no_phys = tuple(f for f in RAY_FEATURES if f not in ('SCATTER', 'WEATHER', 'VCLOUDS', 'SURFACE', 'RELIEF',
                                                          'RELIEF_SHADOW'))
    for q in ('LOW', 'MED', 'HIGH', 'ULTRA'):
        for name, pose in POSES.items():
            for persp in (False, True):
                for feats in (None, RAY_FEATURES, no_phys):
                    imgs, _ = view(q, feats, cam=pose['cam'], sat=pose.get('sat'), vel=pose.get('vel'),
                                   perspective=persp, debug=('NANCHECK',))
                    for k, im in imgs.items():
                        mag = (im[..., 0] > 0.99) & (im[..., 1] < 0.01) & (im[..., 2] > 0.99)
                        if mag.any():
                            bad[(q, name, persp, k)] = int(mag.sum())
    assert not bad, bad
    return 'clean'


@gpu
def test_two_pass_matches_single_pass():
    """Radiance + overlay at render_scale 1.0 composite to (nearly) the single pass."""
    single = view('LOW')[0]['overlay']
    two = view('LOW', render_scale=0.999999)[0]
    rad, ovl = two['radiance'], two['overlay']
    comp = ovl + rad * (1.0 - ovl[..., 3:4])
    h = min(comp.shape[0], single.shape[0])
    w = min(comp.shape[1], single.shape[1])
    d = np.abs(comp[:h, :w, :3] - single[:h, :w, :3]).max(-1)
    frac = float((d > 8 / 255.0).mean())
    assert frac < 0.01, frac
    return 'pixels off by >8/255: %.4f%%' % (100 * frac)


def test_lut_cache(tmp=None):
    import tempfile
    d = tmp or tempfile.mkdtemp()
    calls = []

    def bake(size=(8, 4)):
        calls.append(1)
        return luts.to_rgb8(np.linspace(0, 1, size[0] * size[1] * 3).reshape(size[1], size[0], 3))
    a = luts.cached_lut('t', bake, cache_dir=d, size=(8, 4))
    b = luts.cached_lut('t', bake, cache_dir=d, size=(8, 4))
    luts.cached_lut('t', bake, cache_dir=d, size=(8, 2))
    assert len(calls) == 2 and np.array_equal(np.asarray(a), np.asarray(b))
    assert 'v%d' % luts.TEXTURE_VERSION in luts.lut_key('t', {})
    return 'ok'


def truth_points(gv, cam, sat, vel, n=40000):
    """Visible orbit-circle points projected exactly, in 3x sprite pixels."""
    p, v = np.asarray(sat, float), np.asarray(vel, float)
    e1 = p / np.linalg.norm(p)
    e2 = np.cross(np.cross(p, v), p)
    e2 /= np.linalg.norm(e2)
    R = np.linalg.norm(p) / RE
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    C = (np.outer(np.cos(ph), e1) + np.outer(np.sin(ph), e2)) * R
    persp, f, d = gv.projection(cam)
    M = cam.matrix()
    if persp:
        eye = M[2] * d
        to = eye - C
        L = np.linalg.norm(to, axis=1)
        to /= L[:, None]
    else:
        to = np.tile(M[2], (n, 1))
        L = np.full(n, 1e9)
    b = (C * to).sum(1)
    c = (C * C).sum(1) - 1.0
    h = b * b - c
    t0 = -b - np.sqrt(np.maximum(h, 0))
    hidden = (h > 0) & (t0 > 0) & (t0 < L)
    x, y, _ = gv.project(C[~hidden] * RE, cam)
    gx, gy, gw, gh = PANEL
    i = gv.inset
    px = (x - (gx + i)) * 3.0
    py = ((gy + gh - i) - y) * 3.0
    return np.stack([px, py], 1)


@gpu
def test_ring_matches_exact_projection():
    """Every ring pixel lies on the visible circle and every visible circle point is drawn."""
    rep = {}
    for name, pose in POSES.items():
        for persp in (False, True):
            kw = dict(pose, trail=0.0, perspective=persp)
            sat, vel = kw.pop('sat', None), kw.pop('vel', None)
            if sat is None:
                sat, vel = orbit()
            img, gv = view('LOW', ('ORBIT',), RED, sat=sat, vel=vel, **kw)
            m = ring_mask(img['overlay'])
            T = truth_points(gv, pose['cam'], sat, vel)
            H, W = m.shape
            T = T[(T[:, 0] > 2) & (T[:, 0] < W - 3) & (T[:, 1] > 2) & (T[:, 1] < H - 3)]
            P = _pts(m) + 0.5                    # pixel centres
            # ignore the anti-aliased limb, where a ring sliding behind the Earth
            # at a grazing angle legitimately shows through partly covered pixels
            limb = edge(disc_mask(view('LOW', (), RED, sat=sat, vel=vel, **kw)[0]['overlay']))
            L = _pts(limb) + 0.5

            def away(A):
                keep = np.ones(len(A), bool)
                for k in range(0, len(A), 2048):
                    keep[k:k + 2048] = ((A[k:k + 2048, None] - L[None]) ** 2).sum(-1).min(1) >= 4.0 ** 2
                return A[keep]
            def dmin(A, B):
                out = np.empty(len(A))
                for k in range(0, len(A), 2048):
                    out[k:k + 2048] = np.sqrt(((A[k:k + 2048, None] - B[None]) ** 2).sum(-1).min(1))
                return out
            off = dmin(away(P), T).max()         # drawn pixel far from the true ring
            miss = dmin(away(T), P).max()        # true visible point with no ring pixel near
            rep[(name, 'persp' if persp else 'ortho')] = (round(off, 2), round(miss, 2))
            assert off <= 2.5 and miss <= 1.5, (name, persp, off, miss)
    return rep


def test_transmittance_lut_accuracy():
    """Plan: LUT within 1 % of direct NumPy integration after 8-bit encoding."""
    rg, rt = luts.planet(3.0)
    img = luts.bake_transmittance(3.0)
    rng = np.random.RandomState(0)
    r = rg + rng.rand(20000) ** 2 * luts.ATMO['top_km']
    mu = rng.rand(20000) * 2 - 1
    up = mu > -np.sqrt(np.maximum(1 - (rg / r) ** 2, 0))     # rays that reach space
    T = luts.transmittance_direct(r[up], mu[up], rg, rt, steps=2000)
    err = np.abs(T - luts.lookup_transmittance(img, r[up], mu[up], rg, rt)).max()
    assert err < 0.01, err
    assert img.mode == 'RGB'
    vert = luts.transmittance_direct(np.array([rg]), np.array([1.0]), rg, rt, 2000)[0]
    assert vert[2] < vert[1] < vert[0] < 1.0                  # blue extinguished most
    return 'max abs error %.4f' % err


@gpu
def test_scatter_physics_sanity():
    """Day limb blue, backlit limb present, night disc dark, terminator not NaN."""
    rep = {}
    cam = Camera(az=0.55, el=0.42)
    M = cam.matrix()
    for name, deg in (('noon', 20), ('backlit', 175), ('night', 180)):
        a = math.radians(deg)
        sun = M[2] * math.cos(a) + M[0] * math.sin(a)
        img = view('LOW', None, cam=cam, sun=sun)[0]['overlay']
        h, w = img.shape[:2]
        c = img[h // 2, w // 2, :3]
        # a pixel just outside the right limb (disc radius = 0.44 * min(w,h) * RE / (RE+500))
        rpx = 0.44 * min(w, h) * RE / (RE + 500.0)
        # brightest pixel of the (atmo_scale-thin) shell, clear of the disc's AA edge
        row = img[h // 2, int(w / 2 + rpx * 1.008):int(w / 2 + rpx * 1.03), :3]
        limb = row[row.sum(-1).argmax()]
        rep[name] = (c.round(3).tolist(), limb.round(3).tolist())
    assert rep['noon'][1][2] > rep['noon'][1][0], rep           # day limb is blue
    assert max(rep['night'][0]) < 0.08, rep                       # night side is dark
    assert max(rep['backlit'][1]) > 0.1, rep                      # sunlit ring behind the Earth
    return rep


def test_noise_atlas_tiles():
    """The atlas reproduces the 30^3 volume at texel centres and wraps seamlessly."""
    img = luts.bake_noise()
    assert img.size == luts.NOISE_ATLAS and img.mode == 'RGB'
    vol = luts.noise_volume()
    rng = np.random.RandomState(1)
    ijk = rng.randint(0, luts.NOISE_TILE, (500, 3))
    got = luts.lookup_noise(img, ijk.astype(float))
    want = vol[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
    assert np.abs(got - want).max() < 1.0 / 255 + 1e-9, np.abs(got - want).max()
    # continuity across the tile seam on every axis
    worst = 0.0
    for ax in range(3):
        p = rng.rand(300, 3) * 30.0
        a, b = p.copy(), p.copy()
        a[:, ax], b[:, ax] = 29.999, 30.001
        worst = max(worst, np.abs(luts.lookup_noise(img, a) - luts.lookup_noise(img, b)).max())
    assert worst < 0.02, worst
    return 'seam jump %.4f' % worst


def test_weather_map():
    """Weather bake: right size, plausible global coverage, clear deserts, cloudy storm tracks."""
    img = luts.bake_weather()
    assert img.size == luts.WEATHER_SIZE and img.mode == 'RGB'
    a = np.asarray(img, float) / 255.0
    W, H = luts.WEATHER_SIZE
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    cov = a[..., 0]
    mean = cov.mean()
    sahara = cov[(lat > 18) & (lat < 28)][:, int(W * (190 / 360.0)):int(W * (205 / 360.0))].mean()
    storm = cov[(np.abs(lat) > 45) & (np.abs(lat) < 60)].mean()
    assert 0.25 < mean < 0.6, mean
    assert sahara < 0.15, sahara
    assert storm > mean, (storm, mean)
    assert np.array_equal(np.asarray(luts.bake_weather()), np.asarray(img))   # deterministic
    return dict(mean=round(mean, 3), sahara=round(sahara, 3), storm_tracks=round(storm, 3))


@gpu
def test_clouds_two_pass_matches_single():
    """Volumetric clouds: half-res machinery at scale ~1 composites to the single pass."""
    single = view('MED', render_scale=1.0)[0]['overlay']
    two = view('MED', render_scale=0.999999)[0]
    comp = two['overlay'] + two['radiance'] * (1.0 - two['overlay'][..., 3:4])
    h = min(comp.shape[0], single.shape[0])
    w = min(comp.shape[1], single.shape[1])
    d = np.abs(comp[:h, :w, :3] - single[:h, :w, :3]).max(-1)
    frac = float((d > 12 / 255.0).mean())
    assert frac < 0.02, frac
    return 'pixels off by >12/255: %.3f%%' % (100 * frac)


@gpu
def test_clouds_visible_every_preset():
    """With clouds on, the sunlit disc gets brighter (white cloud over dark ocean); off, no change."""
    rep = {}
    for q in ('LOW', 'MED', 'HIGH', 'ULTRA'):
        on = view(q, clouds=0.55)[0]
        off = view(q, clouds=0.0)[0]
        k = 'radiance' if 'radiance' in on else 'overlay'
        a, b = on[k][..., :3].mean(), off[k][..., :3].mean()
        rep[q] = (round(a, 4), round(b, 4))
        assert a > b * 1.15, rep
    return rep


def test_weather_model_short_run():
    """Offline weather model: stable, bounded winds, frames in range (coarse grid, 3 days)."""
    from orbitgfx import weather
    land = np.zeros((64, 128))
    land[20:40, 30:50] = 1.0
    c = weather.Climate(128, 64, decl_deg=-9.0, land=land)
    for _ in range(48 * 3):
        c.step()
    cov, typ, th = c.frame()
    for a in (c.model.psi1, c.model.psi2, c.qv, c.cl, c.cc, cov):
        assert np.isfinite(a).all()
    umax = float(np.abs(c.model.u1).max()) * 73.7
    assert umax < 150.0, umax
    for a in (cov, typ, th):
        assert a.min() >= 0.0 and a.max() <= 1.0
    img = weather.frame_image(cov, typ, th, size=(64, 32))
    assert img.size == (64, 32) and img.mode == 'RGB'
    return dict(max_u1_ms=round(umax, 1), cov_mean=round(float(cov.mean()), 3))


def test_weather_film_playback():
    """Film frame choice: follows the sun's position (UTC) and cloud_hours, continuous in time."""
    from orbitgfx.views import WeatherFilm
    if not WeatherFilm.available():
        return 'no film installed (skipped)'
    f = WeatherFilm()
    assert f.n >= 24 and abs(f.hours - 1.0) < 1e-9
    # sun overhead at longitude 180 (Earth rotation 0) = 00:00 UTC
    k0, k1, a = f.position(0.0, (-1.0, 0.0, 0.0), 0.0)
    assert (k0, round(a, 6)) == (0, 0.0), (k0, a)
    # 6 hours later the sun is overhead at 90 E
    th = 2 * math.pi * 0.25
    k0, k1, a = f.position(6.0, (math.cos(math.pi / 2 + th), math.sin(math.pi / 2 + th), 0.0), th)
    assert k0 == 6 and abs(a) < 1e-6, (k0, a)
    # continuous: stepping 10 minutes advances the film by 1/6 frame, wrapping at the loop end
    prev = None
    for i in range(int(f.n * 6) + 12):
        hours = i / 6.0
        th = 2 * math.pi * hours / 24.0
        sun = (-1.0, 0.0, 0.0)                              # fixed in space, the Earth turns
        k0, k1, a = f.position(hours, sun, th)
        pos = k0 + a
        if prev is not None:
            d = (pos - prev) % f.n
            assert abs(d - 1 / 6.0) < 1e-6, (i, prev, pos)
        prev = pos
    tex = f.texture(3)
    assert tex is f.texture(3)                              # cached
    return '%d frames, %.0f h each' % (f.n, f.hours)


@gpu
def test_clouds_stable_over_time():
    """No screen-space or temporal jitter: a still view renders the same clouds frame to frame
    (only night-side lightning may flicker)."""
    rep = {}
    for q in ('MED', 'HIGH', 'ULTRA'):
        _, gv = view(q)
        a = scene._render_sprite(gv.rad_node, 3.0).astype(float)
        b = scene._render_sprite(gv.rad_node, 3.37).astype(float)
        frac = float((np.abs(a - b).max(-1) > 2).mean())
        rep[q] = round(100 * frac, 4)
        assert frac < 0.001, rep
    return rep


@gpu
def test_city_lights_hidden_under_cloud():
    """Night side (Australia after sunset): the overlay's city lights are dimmed by the
    cloud the line of sight crosses."""
    hours = 30.0
    theta = 1.3 + hours * 3600 * 7.2921e-5
    a = theta + math.radians(140.0)
    sun = np.array([math.cos(a - math.radians(100)), math.sin(a - math.radians(100)), math.sin(math.radians(-9))])
    sun /= np.linalg.norm(sun)
    cam = Camera(az=a, el=math.radians(-30.0), zoom=2.2)

    from orbitgfx.sandbox import previews as P

    def warm(clouds, q):
        host = P.Sheet()
        host._fw_set(size=scene.Size(*P.PANEL), bounds=scene.Rect(0, 0, *P.PANEL))
        gv = GlobeView(host, P._texture(), quality=q, fill=0.44, inset=0.0, clouds=clouds)
        gv.layout((0.0, 0.0) + P.PANEL)
        gv.update(cam, sun, theta, cloud_hours=hours)
        im = scene._render_sprite(gv.node, 40.0).astype(float) / 255.0
        return float((np.clip(im[..., 0] - im[..., 2], 0, None) * (im[..., 0] > 0.25)).sum())
    rep = {}
    for q in ('MED', 'HIGH'):
        clear = warm(0.0, q)
        assert clear > 100.0, (q, clear)                     # lights are in view at all
        rep[q] = round(warm(0.55, q) / clear, 3)
        assert rep[q] < 0.8, rep
    return rep

def _sunvis(p, sun):
    """numpy port of the PENUMBRA sunVis() (Earth radii)."""
    r = np.linalg.norm(p)
    s = float(p @ sun)
    if s >= 0.0:
        return np.ones(3)
    b = math.sqrt(max(r * r - s * s, 0.0))
    rho = math.asin(min(1.0 / r, 1.0))
    th = math.acos(max(-1.0, min(1.0, -s / r)))
    x = max(-1.0, min(1.0, (th - rho) / 0.004654))
    geo = 0.5 + (x * math.sqrt(1.0 - x * x) + math.asin(x)) / math.pi
    h = max((b - 1.0) * 6378.137, 0.0)
    tau = np.array((3.29, 7.68, 18.7)) * math.exp(-h / 8.0) + 0.97 * math.exp(-h / 1.2)
    return geo * np.exp(-tau)


def _eclipse_setup(u_entry=0.9):
    from orbitgfx.sandbox import previews as P
    p, v = P._orbit(u_entry)
    r = np.linalg.norm(p) / RE
    a = math.pi - math.asin(1.0 / r)
    sun = p / np.linalg.norm(p) * math.cos(a) - v / np.linalg.norm(v) * math.sin(a)
    return P, sun / np.linalg.norm(sun), math.sqrt(398600.4418 / (RE + 500.0) ** 3)


def test_penumbra_sunlight():
    """Sunlight on a 500 km spacecraft through eclipse entry: full sun, then a
    reddening, dimming penumbra (refraction-free Beer-Lambert through the limb)
    lasting seconds, then umbra. Monotonic in every channel."""
    P, sun, n = _eclipse_setup()
    ts = np.linspace(-30.0, 10.0, 81)
    vis = np.array([_sunvis(P._orbit(0.9 + t * n)[0] / RE, sun) for t in ts])
    assert np.all(np.diff(vis, axis=0) <= 1e-9), 'not monotonic'
    assert vis[0].min() > 0.99 and vis[-1].max() < 1e-3, (vis[0], vis[-1])
    red = vis[:, 0] / np.maximum(vis[:, 2], 1e-9)
    mid = (vis[:, 0] > 0.05) & (vis[:, 0] < 0.8)
    assert mid.any() and red[mid].min() > 2.0, red[mid]          # sunset-coloured
    dur = float(np.ptp(ts[(vis.mean(1) < 0.95) & (vis.mean(1) > 0.02)]))
    assert 3.0 < dur < 25.0, dur
    return {'penumbra_s': round(dur, 1), 'red/blue mid': round(float(np.median(red[mid])), 1)}


@gpu
def test_eclipse_marker_colour():
    """Rendered spacecraft glow through the entry: warm, then red, then dark."""
    P, sun, n = _eclipse_setup()
    out = []
    for t in (-18.0, -8.0, -4.0, 0.0, 4.0, 24.0):
        sat, vel = P._orbit(0.9 + t * n)
        cam = Camera(az=math.atan2(sat[1], sat[0]) - 0.35 / 2.6,
                     el=math.asin(sat[2] / np.linalg.norm(sat)) + 0.30 / 2.6, zoom=2.6)
        host = P.Sheet()
        host._fw_set(size=scene.Size(*P.PANEL), bounds=scene.Rect(0, 0, *P.PANEL))
        gv = GlobeView(host, P._texture(), quality='HIGH', fill=0.44, inset=0.0)
        gv.layout((0.0, 0.0) + P.PANEL)
        gv.update(cam, sun, 1.3, sat=sat, vel=vel, trail_rad=0.0, cloud_hours=3.0)
        x, y, _ = gv.project(sat, cam)
        im = scene._render_sprite(gv.node, 40.0).astype(float) / 255.0
        H = im.shape[0]
        s = im.shape[1] / P.PANEL[0]
        cx, cy = int(x * s), int(H - y * s)
        ring = im[cy - 12:cy + 13, cx - 12:cx + 13].reshape(-1, im.shape[2])[:, :3]
        out.append(ring.mean(0))
    lit, dark = out[0], out[-1]
    red = [o[0] / max(o[2], 1e-3) for o in out]
    assert max(red[1:-1]) > red[-1] * 1.3, red                  # warm on the way in, violet once dark
    assert dark.sum() < lit.sum(), (lit, dark)
    return [tuple(round(float(c), 3) for c in o) for o in out]


@gpu(what='render')
def test_astronaut_view():
    """Astronaut camera riding behind the spacecraft: finite image, sky above, the
    horizon an arc that is highest in the middle (seen from 500 km)."""
    P, sun, n = _eclipse_setup()
    sat, vel = P._orbit(0.9 - 1.2)
    cam = Camera()
    cam.follow = cam.astro = True
    cam.step(1 / 60.0, sat, vel)
    assert cam.eye is not None
    host = P.Sheet()
    host._fw_set(size=scene.Size(*P.PANEL), bounds=scene.Rect(0, 0, *P.PANEL))
    gv = GlobeView(host, P._texture(), quality='HIGH', fill=0.44, inset=0.0)
    gv.layout((0.0, 0.0) + P.PANEL)
    gv.update(cam, sun, 1.3, sat=sat, vel=vel, cloud_hours=3.0)
    import skia
    snap = scene.render(host, None, u_time=40.0)
    im = snap.toarray(colorType=skia.kRGBA_8888_ColorType).astype(float) / 255.0
    assert np.isfinite(im).all()
    lum = im[..., :3].mean(-1)
    H, W = lum.shape
    rows = []
    for fx in (0.08, 0.5, 0.92):
        col = lum[:, int(fx * W) - 3:int(fx * W) + 4].mean(1)
        run = np.convolve((col > 0.2).astype(float), np.ones(24), 'valid') >= 24
        rows.append(int(np.argmax(run)))                       # top of the disc (stars ignored)
    assert 0.3 * H < rows[1] < 0.8 * H, rows
    assert rows[1] < rows[0] and rows[1] < rows[2], rows       # dome, not bowl
    cam.drag(5.0, 0.0, 0.0)                                    # touching leaves the mode
    assert cam.eye is None and not cam.astro
    return rows


def test_surface_textures():
    """Phase 5 baked surface: sea ice in the Arctic in March and none in the tropics,
    height peaks in the Himalaya, Black Marble lights on Europe and not at sea,
    land matches the Blue Marble colour (desert bright, ocean dark)."""
    from PIL import Image
    d = os.path.join(PKG, 'data', 'surface')
    col = np.asarray(Image.open(os.path.join(d, 'color.jpg')), float) / 255.0
    ear = np.asarray(Image.open(os.path.join(d, 'earth.png')), float) / 255.0
    aux = np.asarray(Image.open(os.path.join(d, 'aux.png')), float) / 255.0
    H, W = aux.shape[:2]

    def at(a, lat, lon, r=2):
        H, W = a.shape[:2]
        y, x = int((90 - lat) / 180 * H), int((lon + 180) / 360 * W) % W
        return a[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].mean((0, 1))
    rep = {'ice_pole': round(float(at(aux, 88, 0)[1]), 2), 'ice_hudson': round(float(at(aux, 60, -86)[1]), 2),
           'ice_tropics': round(float(aux[int(H * 0.3):int(H * 0.7), :, 1].max()), 2),
           'everest_km': round(float(9.0 * at(aux, 28.0, 86.9)[0] ** 2), 1),
           'lights_paris': round(float(at(ear, 48.9, 2.35)[2]), 2), 'lights_atlantic': float(at(ear, 30, -40)[2]),
           'sahara': round(float(at(col, 23, 10).mean()), 2), 'ocean': round(float(at(col, -30, -120).mean()), 2)}
    assert rep['ice_pole'] > 0.8 and rep['ice_hudson'] > 0.8 and rep['ice_tropics'] == 0.0, rep
    assert 5.0 < rep['everest_km'] < 9.5, rep
    assert rep['lights_paris'] > 0.3 and rep['lights_atlantic'] == 0.0, rep
    assert rep['sahara'] > 0.5 and rep['ocean'] < 0.1, rep
    He, We = ear.shape[:2]
    wlat = np.cos(np.radians(90.0 - (np.arange(He) + 0.5) / He * 180.0))[:, None]
    rep['land_frac'] = round(float((ear[..., 0] * wlat).sum() / (wlat.sum() * We)), 3)
    assert We == 2 * He and We >= 2048, (We, He)
    assert 0.26 < rep['land_frac'] < 0.32, rep                                    # Earth: 29 %
    return rep


def _sunvis_refr(p, sun):
    """numpy port of sunVis() with REFRACTION (the limb as a lens)."""
    bend = lambda h: 0.0203 * math.exp(-h / 7.3)
    r = np.linalg.norm(p)
    s = float(p @ sun)
    if s >= 0.0:
        return np.ones(3)
    b = math.sqrt(max(r * r - s * s, 0.0))
    rho = math.asin(min(1.0 / r, 1.0))
    th = math.acos(max(-1.0, min(1.0, -s / r)))
    L = math.sqrt(max(r * r - 1.0, 0.0)) * 6378.137
    d = rho - th
    h = max((b - 1.0) * 6378.137, 0.0)
    if h < 120.0:
        lo, hi = 0.0, 120.0
        for _ in range(12):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if bend(m) - m / L > d else (lo, m)
        h = 0.5 * (lo + hi)
    x = max(-1.0, min(1.0, (bend(h) - d) / 0.004654))
    geo = (0.5 + (x * math.sqrt(1.0 - x * x) + math.asin(x)) / math.pi) / (1.0 + L * bend(h) / 7.3)
    tau = np.array((3.29, 7.68, 18.7)) * math.exp(-h / 8.0) + 0.97 * math.exp(-h / 1.2)
    return geo * np.exp(-tau)


def test_refraction():
    """REFRACTION: the setting sun from the ground is lifted ~0.48 deg at true elevation 0
    (Saemundsson), and from orbit the limb acts as a lens: after geometric shadow entry
    a dim red sunlight lingers for several seconds, continuously, and the light fades
    monotonically."""
    def refr_deg(e, h=0.0):
        a = math.radians(max(e, -2.0) + 10.3 / (max(e, -2.0) + 5.11))
        f = min(1.0, max(0.0, (e + 2.0) / 1.0))
        f = f * f * (3 - 2 * f)
        return 1.02 / 60.0 / math.tan(a) * math.exp(-h / 8.0) * f
    rep = {'horizon_deg': round(refr_deg(0.0), 3), 'at_10deg': round(refr_deg(10.0), 3),
           'at_10km': round(refr_deg(0.0, 10.0), 3)}
    assert 0.45 < rep['horizon_deg'] < 0.65 and rep['at_10deg'] < 0.12 and rep['at_10km'] < 0.2, rep
    P, sun, n = _eclipse_setup()
    ts = np.linspace(-30.0, 30.0, 121)
    v0 = np.array([_sunvis(P._orbit(0.9 + t * n)[0] / RE, sun) for t in ts])
    v1 = np.array([_sunvis_refr(P._orbit(0.9 + t * n)[0] / RE, sun) for t in ts])
    assert np.all(np.diff(v1, axis=0) <= 1e-6), 'not monotonic'
    assert np.abs(np.diff(v1, axis=0)).max() < 0.1, 'jumps'
    last0 = float(ts[v0[:, 0] > 1e-3].max())
    last1 = float(ts[v1[:, 0] > 1e-3].max())
    rep['red_light_lasts_s'] = (round(last0, 1), round(last1, 1))
    assert last1 > last0 + 5.0, rep
    return rep


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    run_all(globals())
