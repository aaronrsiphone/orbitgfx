"""End-to-end checks for orbitgfx + the sandbox, using examples/mission_control.py.

    python -m orbitgfx.tests.test_mission       # or: pytest --pyargs orbitgfx.tests
Writes PNGs to orbitgfx/tests/out/. Pixel regression runs if tests/baseline/*.png exist.
Desktop only (needs the sandbox's moderngl + skia).
"""
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if __package__ in (None, ''):          # also runnable as a plain script
    sys.path.insert(0, os.path.dirname(PKG))
from orbitgfx.sandbox.runner import Runner  # noqa: E402
import scene  # noqa: E402  (the sandbox shim, installed by the runner)
from orbitgfx.tests._gpu import gpu, run_all  # noqa: E402

APP = os.path.join(PKG, 'examples', 'mission_control.py')
OUT = os.path.join(HERE, 'out')
BASE = os.path.join(HERE, 'baseline')
EPOCH = 1792188000.0          # 2026-10-16 ~22:00 UTC, fixed for reproducibility


def fixed_epoch(mod):
    mod.EPOCH_UNIX = EPOCH


def spherical_stations(mod):
    """1.x station geometry (sphere, radial vertical): the baseline frames were rendered
    with it, and their HUD text and pass times depend on it."""
    from orbitgfx.astro import RE, llxyz
    fixed_epoch(mod)
    mod.geodetic_to_ecef = lambda lat, lon, alt_km=0.0: np.array(llxyz(lon, lat))
    mod.geodetic_up = lambda lat, lon: np.array(llxyz(lon, lat)) / RE


def make(before_setup=fixed_epoch, **kw):
    os.makedirs(OUT, exist_ok=True)
    return Runner(APP, 'Mission', size=(390, 844), before_setup=before_setup, **kw)


def hold_at(s, met):
    s.met0 = s.met = met
    s.speed_idx = 0
    s.wall0 = __import__('time').monotonic()
    s.last_hud = -1.0


def test_physics():
    with make() as r:
        s = r.scene
        assert s.edrift < 1e-7 and s.hdrift < 1e-7
        n = 2 * math.pi / s.period
        a = 6378.137 + 500.0
        theory = -1.5 * n * 1.08262668e-3 * (6378.137 / a) ** 2 * math.cos(math.radians(51.6))
        theory = math.degrees(theory) * 86400
        assert abs(s.raan_rate - theory) < 0.02, (s.raan_rate, theory)
        assert s.passes and s.eclipses


def test_one_update_per_frame():
    """Pythonista calls update() once before every draw(); draw() must not call it again."""
    with make() as r:
        s = r.scene
        calls = []
        orig = s.update
        s.update = lambda: (calls.append(1), orig())[1]
        r.advance(0.5)
        assert len(calls) == r.frames, (len(calls), r.frames)
        del calls[:]
        s.draw()
        assert not calls, 'draw() called update()'


def test_ground_stations_wgs84():
    """Stations sit on the WGS84 ellipsoid; pass elevations use its normal."""
    from orbitgfx.astro import OMEGA_E, elevation, geodetic_to_ecef, geodetic_up
    with make() as r:
        s = r.scene
        for st in s.stations:
            assert np.allclose(st['ecef'], geodetic_to_ecef(st['lat'], st['lon']))
            assert np.allclose(st['up'], geodetic_up(st['lat'], st['lon']))
        st = s.stations[0]
        assert np.allclose(st['el'], elevation(s.ecef - st['ecef'], st['up']))
        # live link selection and the precomputed passes agree mid-pass
        a, b, _, k = s.passes[0]
        tm = 0.5 * (a + b)
        p, v = s._sample(tm)
        link = s._link(p, v, s.theta0 + OMEGA_E * tm)
        assert link is not None and link[1] > 5.0, link


@gpu(what='render')
def test_frame_loop_and_touches():
    """The exact interactions that crashed on device before."""
    with make() as r:
        s = r.scene
        r.advance(0.5)                                  # dt is now framework-owned and nonzero
        gx, gy, gw, gh = s.globe
        mx, my, mw, mh = s.map
        g = (gx + gw / 2, gy + gh / 2)
        speed0 = s.speed_idx
        r.tap(*g); r.advance(0.4)                       # deferred single tap fires
        assert s.speed_idx == (speed0 + 1) % 5
        r.double_tap(*g); r.advance(0.4)
        assert s.cam.follow and s.speed_idx == (speed0 + 1) % 5
        az0 = s.cam.az
        r.drag([(g[0] - 60, g[1]), (g[0] + 60, g[1])], duration=0.2)
        assert not s.cam.follow and s.cam.az != az0
        r.advance(0.5)                                  # inertia + decay without errors
        r.pinch(g, 60, 150)
        assert 2.0 < s.cam.zoom <= 5.0, s.cam.zoom
        hold_at(s, s.met); r.frame()
        met0 = s.met
        r.drag([(mx + 40, my + mh / 2), (mx + mw - 40, my + mh / 2)], duration=0.3)
        assert s.met > met0 + 0.5 * s.period
        assert not s.touches                            # framework dict cleaned up
        r.advance(0.2)
        r.render(os.path.join(OUT, 'after_gestures.png'))


def test_sandbox_blocks_scene_collisions():
    import orbitgfx.safety as safety

    class Bad(scene.Scene):
        def setup(self):
            self.dt = 0.5
            self.touches = {}

    hits = {h[1] for h in safety.collisions(Bad)}
    assert {'dt', 'touches'} <= hits, hits
    try:
        Bad().setup()
    except scene.SandboxError:
        pass
    else:
        raise AssertionError('sandbox should block self.dt = ...')


def _legacy_variants():
    from orbitgfx import FEATURES, build_globe_shader
    for feats in (FEATURES, (), ('ATMOSPHERE', 'CLOUDS'), [f for f in FEATURES if f != 'ORBIT']):
        yield build_globe_shader(feats, palette={'CYAN': (1, 0, 0)})


def test_shader_variants_lint():
    """Static text checks only; compiling is test_shader_variants_compile."""
    from orbitgfx import lint_glsl
    for src in _legacy_variants():
        assert not lint_glsl(src), lint_glsl(src)
    assert lint_glsl('void main(){ gl_FragColor = vec4(pow(x - 1.0, 2.0)); }')


def test_lint_pow_bases():
    """pow(x, y) is exp2(y log2 x): the base must be provably >= 0, and when it can be
    0 the exponent must be positive. 1.0 - z is NOT safe, even for a cosine: rounding
    (sqrt(1.0) on some GPUs, a normalised dot) can make it -1e-7, and one NaN pixel."""
    from orbitgfx import lint_glsl
    flagged = ('float y = pow(1.0 - z, 2.5);',
               'float y = pow(max(x, -1.0), 2.0);',            # max with a negative floor
               'float y = pow(max(x, 0.0), e);',               # 0 ^ (e <= 0) undefined
               'float y = pow(max(a, 0.0), 0.0);',
               'float y = pow(nh, 40.0);',                     # unknown variable
               'float nh = max(d, 0.0); nh -= 0.5; float y = pow(nh, 4.0);',
               'float y = pow(x - 1.0, 2.0);')
    clean = ('float y = pow(max(1.0 - z, 0.0), 2.5);',
             'vec3 y = pow(max(c, vec3(0.0)), vec3(2.2));',
             'vec3 y = pow(clamp(l, 0.0, 1.0), vec3(1.0 / 2.2));',
             'vec3 y = pow(max(T, vec3(1e-6)), vec3(clamp(g / h, 0.0, 1.0)));',  # base > 0
             'float nh = max(dot(a, b), 0.0); float y = pow(nh, 40.0);',
             'float y = pow(abs(sin(t)), 3.0);',
             'float y = 1.0;  // a comment is not code: pow(-1.0, 0.5)')
    for body in flagged:
        w = [x for x in lint_glsl('precision highp float;\n%s\ngl_FragColor' % body) if 'pow' in x]
        assert w, body
    for body in clean:
        w = [x for x in lint_glsl('precision highp float;\n%s\ngl_FragColor' % body) if 'pow' in x]
        assert not w, (body, w)


@gpu
def test_shader_variants_compile():
    """Desktop GLSL 120 compile (not Pythonista's compiler: that needs the device)."""
    for src in _legacy_variants():
        assert scene.Shader(src).compiled


def _diff(a, b):
    from PIL import Image
    A = np.asarray(Image.open(a).convert('RGB'), int)
    B = np.asarray(Image.open(b).convert('RGB'), int)
    d = np.abs(A - B).max(axis=2)
    return d, float((d > 8).mean())


@gpu(what='render')
def test_regression_against_baseline():
    if not os.path.isdir(BASE):
        return
    import time as _t
    with make(spherical_stations) as r:
        s = r.scene
        sys.modules[type(s).__module__].GLOBE_SUN_GLYPH = True     # the 1.x frames had it
        shots = {}
        t = s.passes[0][0] + 120
        hold_at(s, 1151); s.update(); s._hud()
        shots['f1.png'] = 1151
        r.render(os.path.join(OUT, 'f1.png'), u_time=1151)
        hold_at(s, t); s.update(); s._hud()
        r.render(os.path.join(OUT, 'f2.png'), u_time=t)
        e = s.eclipses[0][0] + 200
        hold_at(s, e); s.update(); s._hud()
        r.render(os.path.join(OUT, 'f3.png'), u_time=e)
        hold_at(s, t)
        from orbitgfx.astro import sun_dir
        sun = sun_dir(s.jd0 + t / 86400.)
        s.cam.az, s.cam.el, s.cam.zoom = math.atan2(sun[1], sun[0]) + 0.9, 0.35, 1.0
        s.update(); s._hud()
        r.render(os.path.join(OUT, 'day.png'), u_time=t)
        s.cam.follow, s.cam.zoom = True, 2.6
        for _ in range(400):
            s.last_update = _t.monotonic() - 0.05
            s.update()
        s.cam.follow = False
        s._hud()
        r.render(os.path.join(OUT, 'track.png'), u_time=t)
    report = []
    for name in ('f1.png', 'f2.png', 'f3.png', 'day.png', 'track.png'):
        b = os.path.join(BASE, name)
        if os.path.exists(b):
            d, frac = _diff(os.path.join(OUT, name), b)
            report.append((name, int(d.max()), frac))
            assert frac < 0.002, (name, frac)
    return report


if __name__ == '__main__':
    warnings.simplefilter('always')
    run_all(globals())
