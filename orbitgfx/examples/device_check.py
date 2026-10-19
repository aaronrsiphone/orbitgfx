"""orbitgfx on-device verification and performance check (Pythonista, iPhone / iPad).

Runs everything the desktop sandbox cannot: Apple's shader compiler, the iOS GPU's
float maths, real frame times, memory and heat. About 8 minutes (QUICK: about 3).

    1. Plug the device in, close other apps, Low Power Mode off, brightness any.
    2. Run this file in Pythonista. Don't touch the screen until it says DONE.
    3. Tap the screen at DONE to share orbitgfx_device_report.zip (AirDrop, Files,
       Mail...). It is also saved in Pythonista's top-level folder, and a short
       summary is copied to the clipboard.
    4. Also copy anything red in Pythonista's console (shader compile errors are
       printed there, not to the report).

If it freezes for more than a minute or Pythonista quits, force-quit, run it again:
it skips the step that failed and says so in the report.

Smoke run first (examples/device_smoke.py, or SMOKE = True below, about a minute):
every module parsed and imported, every shader variant compiled, every stage and
every device call (capture, memory, heat, zip, share) exercised once with no
timing, so a crash shows up in a minute instead of halfway through. It writes
orbitgfx_device_smoke.zip, separately from the full report.

Stages (each writes into report.json as it finishes):
  info          device, iOS, Pythonista, screen
  imports       every orbitgfx .py parsed; every device module imported
  pacing        display frame interval; update() vs draw() calls per frame
  capture       a way to read the screen back, and its colour response
  compile       every preset and feature set through the real shader compiler,
                with NANCHECK (NaN / inf pixels are painted magenta)
  nan_scan      every ray preset x camera pose x sun geometry, NaN pixels counted
  gpu_math      altitudeKm, refrDegAt, refrSin, asin, pow(negative) evaluated on the
                GPU and compared with the CPU (astro.refraction_deg)
  screens       fixed poses saved as PNG for comparison with the desktop renders
  bench         frame time per preset x view, headroom, build and first-frame time
  refraction    cost of REFRACTION (sunAt on every ground pixel) on vs off
  weather       frame hitches while the weather film changes frames
  lifecycle     update() / draw() counts over the whole run

Desktop side: python -m orbitgfx.tools.device_report orbitgfx_device_report.zip
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

try:
    import orbitgfx  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import orbitgfx  # noqa: F401
import scene
from orbitgfx import GlobeView, Camera, cached_earth_image, pil_texture, safety
from orbitgfx.astro import RE
try:
    from orbitgfx.astro import refraction_deg
except ImportError:                # an older build (2.0.0a16): the GPU-maths stage is skipped
    refraction_deg = None
from orbitgfx.examples.phase_check import mission_panel, orbit
from orbitgfx.examples import sunset as SS
from orbitgfx.views import QUALITY

QUICK = False                      # True: fewer poses and shorter timings (~3 minutes)
SMOKE = False                      # True: every code path once, no timing (~1 minute)
ONLY = None                        # e.g. ('bench', 'refraction'): run just these stages (+ info, imports, pacing, capture)
VERSION = 1                        # report format
TOP_INSET = 64
BG = (0.023, 0.039, 0.078)
RAY = ('LOW', 'MED', 'HIGH', 'ULTRA')
STATIONS = [((RE * 0.62, RE * 0.55, RE * 0.56), True), ((-RE * 0.4, RE * 0.75, RE * 0.52), False)]
SUNS = (('noon', 20.0), ('terminator', 90.0), ('backlit', 172.0), ('night', 130.0))
VIEWS = ('overview', 'chase', 'chase/P', 'astro')
FEATURE_SETS = (('all', None), ('none', ()), ('orbit', ('ORBIT', 'SATELLITE')),
                ('atmo_tonemap', ('ATMOSPHERE', 'TONEMAP')), ('orbit_refr', ('ORBIT', 'REFRACTION')),
                ('atmo_refr', ('ATMOSPHERE', 'REFRACTION')))
HEADROOM_STEPS = (1, 2, 3, 5)


def W(full, smoke=0.05):
    """A wait or timing length: `full` normally, `smoke` in a smoke run."""
    return smoke if SMOKE else full


def pick(full, smoke):
    """A list of cases: all of `full` normally, the representative `smoke` ones in a smoke run."""
    return smoke if SMOKE else full
SITE = 3                           # sunset.py's Bismarck Sea thunderstorms
SCREEN_SCALE = 1.5                 # pixels per point of the saved screens (zip size)


OUT_BASE = None                    # folder for the report; None: Pythonista's top level


def out_dir():
    docs = os.path.expanduser('~/Documents')
    base = OUT_BASE or (docs if os.path.isdir(docs) else os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(base, 'orbitgfx_device_smoke' if SMOKE else 'orbitgfx_device_report')
    os.makedirs(os.path.join(d, 'screens'), exist_ok=True)
    return d


# ------------------------------------------------------------------ poses
def cam_state(cam):
    return {'az': cam.az, 'el': cam.el, 'zoom': cam.zoom, 'fov_deg': cam.fov_deg,
            'distance': cam.distance,
            'eye': None if cam.eye is None else [float(v) for v in cam.eye],
            'look': None if cam.look is None else [float(v) for v in cam.look]}


def make_cam(st):
    cam = Camera(az=st['az'], el=st['el'], zoom=st['zoom'], fov_deg=st['fov_deg'],
                 distance=st['distance'])
    if st['eye'] is not None:
        cam.eye, cam.look = np.array(st['eye']), np.array(st['look'])
    return cam


def orbit_pose(view='overview', sun='noon', cloud_hours=2.0):
    """A mission-control-like pose: phase_check.py's views and sun geometries."""
    sat, vel = orbit(0.0)
    cam = Camera(az=0.55, el=0.42)
    if view != 'overview':
        cam.az = math.atan2(sat[1], sat[0]) - 0.35 / 2.6
        cam.el = math.asin(sat[2] / np.linalg.norm(sat)) + 0.30 / 2.6
        cam.zoom = 2.6
    if view == 'astro':
        cam.set_astronaut(sat, vel)
    M = Camera(az=cam.az, el=cam.el).matrix()
    a = math.radians(dict(SUNS)[sun])
    s = M[2] * math.cos(a) + M[1] * math.sin(a) * 0.35 + M[0] * math.sin(a)
    return {'kind': 'orbit', 'view': view, 'sun_name': sun, 'cam': cam_state(cam),
            'sun': [float(v) for v in s / np.linalg.norm(s)], 'theta': 1.3,
            'sat': [float(v) for v in sat], 'vel': [float(v) for v in vel], 'trail': 0.8,
            'cloud_hours': cloud_hours}


def ground_pose(elev=2.0, site=SITE):
    """sunset.py's camera at a site, the sun at true elevation `elev` (deg), setting."""
    name, lat, lon, k = SS.SITES[site]
    up = SS.unit(lat, lon)
    lo, hi = 0.0, math.pi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        e = math.degrees(math.asin(float(np.dot(up, SS.sun_dir(math.radians(lon) - mid)))))
        lo, hi = (mid, hi) if e > elev else (lo, mid)
    sub = math.radians(lon) - lo
    sun = SS.sun_dir(sub)
    t_utc = ((math.pi - sub) / (2.0 * math.pi)) % 1.0
    e_ = np.cross((0.0, 0.0, 1.0), up)
    e_ /= np.linalg.norm(e_)
    n_ = np.cross(up, e_)
    az = math.atan2(float(np.dot(sun, e_)), float(np.dot(sun, n_)))
    pitch = math.radians(6.0)
    fwd = (n_ * math.cos(az) + e_ * math.sin(az)) * math.cos(pitch) + up * math.sin(pitch)
    cam = Camera(fov_deg=SS.FOV)
    cam.eye = up * (1.0 + SS.EYE_KM / RE)
    cam.look = cam.eye + fwd
    return {'kind': 'ground', 'view': 'ground', 'site': name, 'elev': elev, 'cam': cam_state(cam),
            'sun': [float(v) for v in sun], 'theta': 0.0, 'sat': None, 'vel': None, 'trail': 0.0,
            'cloud_hours': (k // 24 + t_utc) * 24.0}


def orbit_spec(quality, features=None, perspective=False, debug=()):
    return {'quality': quality, 'features': None if features is None else list(features),
            'rect': 'panel', 'debug': list(debug),
            'kw': {'fill': 0.44, 'ref_radius_km': RE + 500.0, 'perspective': perspective}}


def ground_spec(quality, refraction=True, debug=()):
    feats = [f for f in SS.FEATURES if refraction or f != 'REFRACTION']
    return {'quality': quality, 'features': feats, 'rect': 'full', 'debug': list(debug),
            'kw': {'perspective': True, 'atmo_scale': 1.0, 'cloud_scale': 1.0, 'inset': 0.0,
                   'exposure': SS.EXPOSURE, 'defines': dict(SS.GROUND)}}


def spec_for(pose, quality, **kw):
    if pose['kind'] == 'ground':
        return ground_spec(quality, **kw)
    return orbit_spec(quality, perspective=pose['view'] in ('chase/P', 'astro'), **kw)


def build_globe(host, tex, spec, rect, z=-1.0):
    kw = dict(spec['kw'])
    g = GlobeView(host, tex, quality=spec['quality'], features=spec['features'],
                  debug=tuple(spec['debug']), z_position=z, **kw)
    g.layout(rect)
    return g


def apply_pose(globe, pose):
    cam = make_cam(pose['cam'])
    sat = None if pose['sat'] is None else np.array(pose['sat'])
    vel = None if pose['vel'] is None else np.array(pose['vel'])
    globe.update(cam, np.array(pose['sun']), pose['theta'], sat=sat, vel=vel,
                 trail_rad=pose['trail'], stations=STATIONS if pose['kind'] == 'orbit' else (),
                 cloud_hours=pose['cloud_hours'])


# ------------------------------------------------------------------ GPU maths probes
def refraction_lib():
    """The shader's units + refraction + sunAt GLSL (as tests/test_raycore.py uses it)."""
    import re
    from orbitgfx.shaders import build_ray_shader
    src = build_ray_shader(('SCATTER', 'REFRACTION'), None, 'single')
    body = re.search(r'// Units:.*?\nvec3 sunAt\(.*?\n\}\n', src, re.S).group(0)
    return 'uniform vec4 u_atm;\nuniform vec3 u_sun;\n' + body


def probes():
    """(name, GLSL expression of x, x0, x1, lo, hi, km per unit, reference(x) or None, tol).

    Tolerances: the refraction model itself to 2e-4 deg; trig round trips to 2e-3 deg
    (fp32 sin/asin are good to ~1e-5 rad; mediump would be off by ~0.05 deg)."""
    from orbitgfx.luts import planet
    km1, km15 = RE, planet(1.5, RE)[0]
    out = [('altitudeKm s1.5', 'altitudeKm(1.0 + x / u_atm.y)', 0.0, 60.0, 0.0, 64.0, km15,
            lambda x: x, 0.01),
           ('asin(sin) deg', 'degrees(asin(clamp(sin(radians(x)), -1.0, 1.0)))', -5.0, 10.0,
            -6.0, 11.0, km1, lambda x: x, 2e-3),
           # up to 8 deg only: refrSin fades the lift out between 8.6 and 14.5 deg, which
           # the plain Saemundsson reference below does not
           ('apparent elev h0', 'degrees(asin(refrSin(1.0, sin(radians(x)))))', -3.0, 8.0,
            -3.0, 9.0, km1, lambda x: x + refraction_deg(x, 0.0, km1), 2e-3),
           # informational: what pow() of a negative base gives here (README rule 1;
           # Mesa: NaN for 5.0 and 2.5, but x * x for 2.0, which compilers rewrite),
           # whether 1 - z can round below 0 (sqrt(1.0), a normalised dot), and whether
           # a shader can see a NaN at all (NANCHECK's x == x; fast-math may fold it)
           ('pow(x, 5)', 'pow(x, 5.0)', -1.0, 1.0, -1.0, 1.0, km1, None, None),
           ('pow(x, 2)', 'pow(x, 2.0)', -1.0, 1.0, 0.0, 1.0, km1, None, None),
           ('1 - sqrt(1 - x^2)', '1.0 - sqrt(max(1.0 - x * x, 0.0))', -1e-3, 1e-3, -1e-6, 1e-6,
            km1, None, None),
           ('1 - |normalize(v)|^2', '1.0 - dot(normalize(vec3(x, 0.31, 0.73)), normalize(vec3(x, 0.31, 0.73)))',
            -1.0, 1.0, -1e-6, 1e-6, km1, None, None),
           ('sees NaN of sqrt(x)', '(sqrt(x) == sqrt(x)) ? 1.0 : 0.0', -1.0, 1.0, 0.0, 1.0, km1, None, None),
           ('sees NaN, abs test', '(abs(sqrt(x)) < 1e30) ? 1.0 : 0.0', -1.0, 1.0, 0.0, 1.0, km1, None, None)]
    for h in (0.0, 0.4, 2.0, 10.0, 30.0):
        out.append(('refrDegAt h%g' % h, 'refrDegAt(1.0 + %r / u_atm.y, x)' % h, -5.0, 10.0,
                    0.0, 1.6, km1, lambda x, h=h: refraction_deg(x, h, km1), 2e-4))
    out.append(('refrDegAt h10 s1.5', 'refrDegAt(1.0 + 10.0 / u_atm.y, x)', -5.0, 10.0, 0.0, 1.6,
                km15, lambda x: refraction_deg(x, 10.0, km15), 2e-4))
    return out


PROBE_N = 128


def probe_shader(lib, expr):
    return ('precision highp float;\nvarying vec2 v_tex_coord;\n#define SCATTER\n#define REFRACTION\n' + lib +
            'uniform float u_x0;\nuniform float u_x1;\nuniform float u_lo;\nuniform float u_hi;\n'
            'void main() {\n'
            '    float i = floor(v_tex_coord.x * %.1f);\n'
            '    float x = mix(u_x0, u_x1, i / %.1f);\n'
            '    float v = %s;\n'
            '    float u = clamp((v - u_lo) / (u_hi - u_lo), 0.0, 1.0);\n'
            '    float a = floor(u * 255.0);\n'
            '    gl_FragColor = vec4(a / 255.0, u * 255.0 - a, 0.0, 1.0);\n'
            '}\n' % (PROBE_N, PROBE_N - 1, expr))


RAMP_SHADER = ('precision highp float;\nvarying vec2 v_tex_coord;\n'
               'void main() {\n'
               '    float i = floor(v_tex_coord.x * 256.0) / 255.0;\n'
               '    gl_FragColor = vec4(i, i, i, 1.0);\n'
               '}\n')
SOLID = (0.25, 0.5, 0.75)
SOLID_SHADER = ('precision highp float;\nvoid main() { gl_FragColor = vec4(%r, %r, %r, 1.0); }\n' % SOLID)


# ------------------------------------------------------------------ screen capture
class Capture(object):
    """Reads the rendered scene back as an RGBA uint8 array (rows top-down).

    Device: SKView.textureFromNode (the whole SpriteKit tree, shaders included), else
    UIView.drawViewHierarchyInRect. Sandbox: scene.render. The first one that returns
    the expected colour for a solid test sprite is used.
    """

    def __init__(self, host, skip=()):
        self.host = host
        self.skip = set(skip)
        self.method = None
        self.scale = None

    def candidates(self):
        if hasattr(scene, 'SandboxError'):
            return ['sandbox']
        return [m for m in ('textureFromNode', 'drawViewHierarchy', 'windowSnapshot') if m not in self.skip]

    def grab(self, method=None):
        m = method or self.method
        a = getattr(self, '_' + m)()
        self.scale = a.shape[1] / float(self.host.size[0])
        return a

    def _sandbox(self):
        import skia
        snap = scene.render(self.host, None)
        self.host._draws_by_capture += 1
        return np.ascontiguousarray(snap.toarray(colorType=skia.kRGBA_8888_ColorType))

    @staticmethod
    def _png_to_array(uiimage):
        import ctypes
        import io
        from PIL import Image
        from objc_util import ObjCInstance
        lib = ctypes.CDLL(None)
        lib.UIImagePNGRepresentation.restype = ctypes.c_void_p
        lib.UIImagePNGRepresentation.argtypes = [ctypes.c_void_p]
        data = ObjCInstance(lib.UIImagePNGRepresentation(uiimage.ptr))
        raw = ctypes.string_at(data.bytes(), data.length())
        return np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'))

    def _roots(self):
        """Candidate UIViews to search from: the scene's view (as a ui.View and as its raw
        pointer), then the app's key window."""
        from objc_util import ObjCClass, ObjCInstance
        roots = []
        v = getattr(self.host, 'view', None)
        for make in (lambda: ObjCInstance(v), lambda: ObjCInstance(v._objc_ptr)):
            try:
                roots.append(make())
            except Exception:
                pass
        try:
            app = ObjCClass('UIApplication').sharedApplication()
            win = app.keyWindow()
            if win is None:
                win = app.windows()[0]
            roots.append(win)
        except Exception:
            pass
        return roots

    def hierarchy(self, limit=80):
        """Class names of the view tree (up from the scene view, then down), for the report."""
        out, seen = [], set()
        for root in self._roots():
            top = root
            for _ in range(20):
                sup = top.superview() if hasattr(top, 'superview') else None
                if sup is None:
                    break
                top = sup
            todo = [(top, 0)]
            while todo and len(out) < limit:
                v, d = todo.pop(0)
                if int(v.ptr.value if hasattr(v.ptr, 'value') else v.ptr) in seen:
                    continue
                seen.add(int(v.ptr.value if hasattr(v.ptr, 'value') else v.ptr))
                out.append('%s%s' % ('  ' * d, str(v._get_objc_classname())
                                     if hasattr(v, '_get_objc_classname') else type(v).__name__))
                if d < 6:
                    todo = [(c, d + 1) for c in v.subviews()] + todo
        return out

    def _skview(self):
        """The SKView: searched in the scene view's subtree, then up its superviews and
        down from the top, then from the key window (Pythonista 3.4 / recent iOS does
        not always put it below scene.view)."""
        from objc_util import ObjCClass
        SKView = ObjCClass('SKView')
        for root in self._roots():
            top = root
            for _ in range(20):
                sup = top.superview()
                if sup is None:
                    break
                top = sup
            for start in (root, top):
                todo = [start]
                while todo:
                    v = todo.pop(0)
                    if v.isKindOfClass_(SKView):
                        return v
                    todo.extend(v.subviews())
        raise RuntimeError('no SKView in the view hierarchy:\n' + '\n'.join(self.hierarchy()))

    def _top_view(self):
        """The topmost UIView above the scene view (the window)."""
        roots = self._roots()
        if not roots:
            raise RuntimeError('no view or window to snapshot')
        top = roots[-1]
        for _ in range(20):
            sup = top.superview()
            if sup is None:
                break
            top = sup
        return top

    def _snapshot(self, view):
        import ctypes
        from objc_util import CGSize, ObjCInstance
        b = view.bounds()
        lib = ctypes.CDLL(None)
        lib.UIGraphicsBeginImageContextWithOptions.argtypes = [CGSize, ctypes.c_bool, ctypes.c_double]
        lib.UIGraphicsGetImageFromCurrentImageContext.restype = ctypes.c_void_p
        lib.UIGraphicsBeginImageContextWithOptions(CGSize(b.size.width, b.size.height), True, 0.0)
        try:
            view.drawViewHierarchyInRect_afterScreenUpdates_(b, True)
            img = ObjCInstance(lib.UIGraphicsGetImageFromCurrentImageContext())
        finally:
            lib.UIGraphicsEndImageContext()
        if img is None:
            raise RuntimeError('snapshot returned nil')
        return self._png_to_array(img)

    def _textureFromNode(self):
        from objc_util import ObjCClass
        sk = self._skview()
        tex = sk.textureFromNode_(sk.scene())
        if tex is None:
            raise RuntimeError('textureFromNode returned nil')
        img = ObjCClass('UIImage').imageWithCGImage_(tex.CGImage())
        return self._png_to_array(img)

    def _drawViewHierarchy(self):
        return self._snapshot(self._skview())

    def _windowSnapshot(self):
        """The whole window, status bar area included: scene points map 1:1 only if the
        scene is full screen (the default of scene.run), which the solid-colour test checks."""
        return self._snapshot(self._top_view())

    def px(self, x, y):
        """Scene point (y up) -> pixel (col, row)."""
        return int((x) * self.scale), int((self.host.size[1] - y) * self.scale)

    def crop(self, a, rect):
        x, y, w, h = rect
        c0, r1 = self.px(x, y)
        c1, r0 = self.px(x + w, y + h)
        return a[max(r0, 0):r1, max(c0, 0):c1]


# ------------------------------------------------------------------ device metrics
def objc(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def device_info():
    import platform
    info = {'orbitgfx': orbitgfx.__version__, 'python': platform.python_version(),
            'machine': platform.machine(), 'platform': platform.platform(),
            'numpy': np.__version__, 'on_device': bool(getattr(orbitgfx, 'ON_DEVICE', False)),
            'screen_scale': objc(lambda: float(scene.get_screen_scale()))}
    try:
        from objc_util import ObjCClass
        dev = ObjCClass('UIDevice').currentDevice()
        pi = ObjCClass('NSProcessInfo').processInfo()
        bundle = ObjCClass('NSBundle').mainBundle()
        info.update(
            ios=str(dev.systemVersion()), model=str(dev.model()),
            pythonista=str(bundle.objectForInfoDictionaryKey_('CFBundleShortVersionString')),
            pythonista_build=str(bundle.objectForInfoDictionaryKey_('CFBundleVersion')),
            physical_memory_gb=round(pi.physicalMemory() / 2 ** 30, 2),
            cpu_cores=int(pi.activeProcessorCount()), low_power=bool(pi.isLowPowerModeEnabled()))
        screen = ObjCClass('UIScreen').mainScreen()
        info['max_fps'] = objc(lambda: int(screen.maximumFramesPerSecond()))
        dev.setBatteryMonitoringEnabled_(True)
        info['battery'] = objc(lambda: round(float(dev.batteryLevel()), 2))
        info['battery_state'] = objc(lambda: int(dev.batteryState()))   # 2 charging, 3 full
    except Exception as e:
        info['objc_error'] = repr(e)
    return info


def vitals():
    """Thermal state (0 nominal .. 3 critical), memory available before jetsam, peak RSS."""
    v = {'t': round(time.time(), 1)}
    try:
        from objc_util import ObjCClass
        v['thermal'] = int(ObjCClass('NSProcessInfo').processInfo().thermalState())
    except Exception:
        v['thermal'] = None
    try:
        import ctypes
        lib = ctypes.CDLL(None)
        lib.os_proc_available_memory.restype = ctypes.c_size_t
        v['avail_mb'] = round(lib.os_proc_available_memory() / 2 ** 20, 1)
    except Exception:
        v['avail_mb'] = None
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        v['peak_rss_mb'] = round(rss / 2 ** 20 if rss > 2 ** 24 else rss / 1024.0, 1)
    except Exception:
        v['peak_rss_mb'] = None
    return v


def frame_stats(dts):
    d = np.sort(np.asarray(dts, float) * 1000.0)
    if len(d) == 0:
        return {'n': 0}
    keep = d[:max(1, int(len(d) * 0.95))]
    return {'n': int(len(d)), 'mean95_ms': round(float(keep.mean()), 2),
            'median_ms': round(float(np.median(d)), 2), 'p95_ms': round(float(np.percentile(d, 95)), 2),
            'p99_ms': round(float(np.percentile(d, 99)), 2), 'max_ms': round(float(d[-1]), 2)}


def pixel_stats(a):
    """Panel pixels: background, NaN magenta, raw data texture (shader fallback?), colour."""
    rgb = a[..., :3].astype(int)
    n = float(max(1, rgb.shape[0] * rgb.shape[1]))
    bg = np.array([int(round(c * 255)) for c in BG])
    is_bg = (np.abs(rgb - bg).max(-1) <= 3)
    mag = (rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] > 200)
    red = (rgb[..., 0] > 200) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    return {'bg': round(float(is_bg.sum() / n), 4), 'nan': round(float(mag.sum() / n), 5),
            'raw_red': round(float(red.sum() / n), 4),
            'mean': [round(float(v), 1) for v in rgb.reshape(-1, 3).mean(0)],
            'std': round(float(rgb.std()), 1)}


class Guard(object):
    """Crash guard: marks what is running; a mark left by a crashed run means skip it."""

    def __init__(self, path):
        self.path = path
        try:
            with open(path) as fh:
                self.prev = json.load(fh)
        except Exception:
            self.prev = {}
        self.now = {}
        self.crashed = sorted(self.prev)
        self._write()

    def _write(self):
        try:
            with open(self.path, 'w') as fh:
                json.dump(self.now, fh)
        except Exception:
            pass

    def mark(self, key):
        self.now[key] = time.time()
        self._write()

    def clear(self, key):
        self.now.pop(key, None)
        self._write()


def summarize(R):
    """The short text summary of a report (clipboard, summary.txt, device_report)."""
    st = R['stages']
    info = st.get('info', {})
    lines = ['ORBITGFX DEVICE %s  orbitgfx %s  %s iOS %s  Pythonista %s' % (
        'SMOKE RUN (no timing)' if R.get('smoke') else 'CHECK', info.get('orbitgfx'),
        info.get('machine'), info.get('ios'), info.get('pythonista'))]
    im = st.get('imports', {})
    if im:
        lines.append('imports: %s parsed, %d imported, failed: %s' % (
            im.get('parsed'), len(im.get('imported', [])), ', '.join(sorted(im.get('failed', {}))) or 'none'))
    pc = st.get('pacing', {})
    lines.append('frame %s ms  updates/draws %s/%s  capture %s' % (
        pc.get('target_ms'), pc.get('updates'), pc.get('draws'), st.get('capture', {}).get('method')))
    comp = st.get('compile', {}).get('checks', [])
    bad = [c['name'] for c in comp if 'error' in c or c.get('pixels', {}).get('bg', 0) > 0.97
           or c.get('pixels', {}).get('nan', 0) > 0]
    lines.append('compile %d checked, suspicious: %s' % (len(comp), ', '.join(bad) or 'none'))
    lines.append('NaN worst %s   GPU maths ok %s' % (st.get('nan_scan', {}).get('worst_nan'),
                                                   st.get('gpu_math', {}).get('all_ok')))
    for r in st.get('bench', {}).get('runs', []):
        s = r['stats']
        lines.append('%-6s %-8s mean %5s p95 %5s build %ss head x%s' % (
            r['quality'], r['view'], s.get('mean95_ms'), s.get('p95_ms'), s.get('build_s'),
            r.get('headroom', '-')))
    for p in st.get('refraction', {}).get('pairs', []):
        on = [x.get('mean95_ms') for x in p.get('on', [])]
        off = [x.get('mean95_ms') for x in p.get('off', [])]
        lines.append('refraction %s on %s off %s ms' % (p['case'], on, off))
    v = R['vitals']
    if v:
        lines.append('thermal %s -> %s  avail %s MB min  peak rss %s MB' % (
            v[0].get('thermal'), v[-1].get('thermal'),
            min([x['avail_mb'] for x in v if x.get('avail_mb') is not None] or [None]),
            v[-1].get('peak_rss_mb')))
    if R['errors']:
        lines.append('errors in: %s' % ', '.join(sorted(R['errors'])))
    if R['skipped_after_crash']:
        lines.append('skipped after a crash: %s' % ', '.join(R['skipped_after_crash']))
    return '\n'.join(lines)


# ------------------------------------------------------------------ the check
class DeviceCheck(scene.Scene):

    def setup(self):
        safety.assert_clean(type(self))
        self.background_color = BG
        w, h = self.size
        self.panel = mission_panel(w, h)
        self.full = (0.0, 0.0, w, h)
        from PIL import Image
        self.tex = pil_texture(cached_earth_image(__import__('orbitgfx.coastlines', fromlist=['x'])
                                                  .coastlines_xyz(RE)))
        self.white = pil_texture(Image.new('RGB', (4, 4), (255, 255, 255)))
        self.hdr = scene.LabelNode('orbitgfx device check', ('Menlo-Bold', 12), parent=self,
                                   color='white', position=(w / 2, h - TOP_INSET - 12))
        self.sub = scene.LabelNode("don't touch the screen", ('Menlo', 10), parent=self,
                                   color=(0.7, 0.8, 0.9), position=(w / 2, h - TOP_INSET - 30))
        self.hdr.z_position = self.sub.z_position = 10
        self.dir = out_dir()
        self.guard = Guard(os.path.join(self.dir, '.running.json'))
        self.report = {'format': VERSION, 'quick': QUICK, 'smoke': SMOKE, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'scene_size': [w, h], 'panel': list(self.panel),
                       'skipped_after_crash': self.guard.crashed, 'stages': {}, 'vitals': [],
                       'errors': {}}
        self.cap = Capture(self, skip=[k.split(':', 1)[1] for k in self.guard.crashed
                                       if k.startswith('capture:')])
        self.globes = []
        self.pose = None
        self.probe = None
        self.n_update = self.n_draw = 0
        self._draws_by_capture = 0
        self.prev = None
        self.samples = None
        self.frames = []
        self.steps = self.run()
        self.resume_at = 0.0
        self.done = False
        self.zip_path = None

    # ------------------------------------------------------------ helpers
    def status(self, a, b=None):
        self.hdr.text = a
        if b is not None:
            self.sub.text = b

    def clear(self):
        for g in self.globes:
            g.remove()
        self.globes = []
        if self.probe is not None:
            self.probe.remove_from_parent()
            self.probe = None

    def build(self, spec, pose, copies=0):
        self.clear()
        rect = self.panel if spec['rect'] == 'panel' else self.full
        t0 = time.perf_counter()
        self.globes = [build_globe(self, self.tex, spec, rect)]
        for k in range(copies):
            self.globes.append(build_globe(self, self.tex, spec, rect, z=-0.9 + 0.01 * k))
        self.pose = pose
        for g in self.globes:
            apply_pose(g, pose)
        return time.perf_counter() - t0

    def measure(self, settle, seconds):
        """Generator: wait `settle`, then collect frame times for `seconds`."""
        self.samples = []
        yield settle
        first = list(self.samples)
        self.samples = []
        yield seconds
        out = frame_stats(self.samples)
        out['settle_max_ms'] = round(max(first) * 1000.0, 1) if first else None
        self.samples = None
        self.last_stats = out

    def save(self):
        self.report['vitals'].append(vitals())
        try:
            with open(os.path.join(self.dir, 'report.json'), 'w') as fh:
                json.dump(self.report, fh, indent=1, sort_keys=True)
        except Exception as e:
            print('report write failed:', e)

    def shot(self, name, rect, full_res=False):
        """Capture, crop to rect, save (PNG at SCREEN_SCALE for screens, 1x JPEG otherwise)."""
        a = self.cap.grab()
        c = self.cap.crop(a, rect)
        try:
            from PIL import Image
            im = Image.fromarray(np.ascontiguousarray(c)).convert('RGB')
            k = (SCREEN_SCALE if full_res else 1.0) / self.cap.scale
            if k < 1.0:
                im = im.resize((max(1, int(round(im.width * k))), max(1, int(round(im.height * k)))),
                               Image.LANCZOS)
            if full_res:
                im.save(os.path.join(self.dir, 'screens', name + '.png'))
            else:
                im.save(os.path.join(self.dir, 'screens', name + '.jpg'), quality=85)
        except Exception as e:
            self.report['errors'].setdefault('image_save', repr(e))
        return c

    # ------------------------------------------------------------ frame loop
    def update(self):
        self.n_update += 1
        now = time.perf_counter()
        if self.prev is not None:
            d = now - self.prev
            self.frames = (self.frames + [d])[-120:]
            if self.samples is not None:
                self.samples.append(d)
        self.prev = now
        if self.pose is not None:
            p = self.pose
            if p.get('warp'):
                p['cloud_hours'] += p['warp'] * (self.frames[-1] if self.frames else 1 / 60.0) / 3600.0
            for g in self.globes:
                apply_pose(g, p)
        if self.done:
            return
        if now >= self.resume_at:
            try:
                self.resume_at = now + (next(self.steps) or 0.0)
            except StopIteration:
                self.done = True
                self.finish()

    def draw(self):
        self.n_draw += 1                       # classic layer: nothing drawn, only counted

    def touch_began(self, touch):
        if self.done and self.zip_path:
            try:
                import console
                console.open_in(self.zip_path)
            except Exception as e:
                self.sub.text = 'share failed: %r' % e

    # ------------------------------------------------------------ stages
    def run(self):
        stages = [('info', self.st_info), ('imports', self.st_imports), ('pacing', self.st_pacing),
                  ('capture', self.st_capture),
                  ('compile', self.st_compile), ('nan_scan', self.st_nan), ('gpu_math', self.st_math),
                  ('screens', self.st_screens), ('bench', self.st_bench),
                  ('refraction', self.st_refraction), ('weather', self.st_weather)]
        self.save()
        for name, fn in stages:
            if ONLY is not None and name not in ONLY and name not in ('info', 'imports', 'pacing', 'capture'):
                continue
            key = 'stage:' + name
            if key in self.guard.crashed:
                self.report['stages'][name] = {'skipped': 'crashed or hung on the previous run'}
                continue
            if name == 'gpu_math' and refraction_deg is None:
                self.report['stages'][name] = {'skipped': 'this build has no astro.refraction_deg'}
                continue
            if name in ('compile', 'nan_scan', 'gpu_math', 'screens') and not self.cap.method:
                self.report['stages'][name] = {'skipped': 'no working screen capture'}
                continue
            self.guard.mark(key)
            self.status('%s ...' % name, "don't touch the screen")
            t0 = time.time()
            self.report['stages'][name] = res = {}
            try:
                yield from fn(res)
            except Exception:
                self.report['errors'][name] = traceback.format_exc()
            res['seconds'] = round(time.time() - t0, 1)
            self.clear()
            self.pose = None
            self.guard.clear(key)
            self.save()
            yield W(0.2)
        self.report['stages']['lifecycle'] = {
            'updates': self.n_update, 'draws': self.n_draw - self._draws_by_capture,
            'draws_by_capture': self._draws_by_capture}

    def st_info(self, res):
        res.update(device_info())
        res['panel_pt'] = list(self.panel)
        yield 0.0

    def st_imports(self, res):
        """Parse every .py in the package (syntax errors anywhere), then import every
        module that is meant to load on the device (the sandbox, the tests and the
        desktop report reader are not)."""
        import importlib
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res['parsed'], res['imported'], res['failed'] = 0, [], {}
        mods = []
        for root, dirs, files in os.walk(pkg):
            dirs[:] = sorted(d for d in dirs if not d.startswith(('.', '__')))
            for f in sorted(files):
                if not f.endswith('.py'):
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, os.path.dirname(pkg))
                try:
                    with open(path, 'rb') as fh:
                        compile(fh.read(), path, 'exec')
                    res['parsed'] += 1
                except SyntaxError:
                    res['failed'][rel] = traceback.format_exc(limit=1)
                    continue
                name = rel[:-3].replace(os.sep, '.')
                if name.endswith('.__init__'):
                    name = name[:-9]
                parts = name.split('.')
                if 'sandbox' in parts or 'tests' in parts or name.endswith('tools.device_report'):
                    continue
                mods.append(name)
        for name in mods:
            self.status('imports', name)
            try:
                importlib.import_module(name)
                res['imported'].append(name)
            except Exception:
                res['failed'][name] = traceback.format_exc(limit=3)
            yield 0.0
        res['ok'] = not res['failed']

    def st_pacing(self, res):
        u0, d0 = self.n_update, self.n_draw
        yield from self.measure(W(0.5, 0.1), W(3.0, 0.3))
        res['frames'] = self.last_stats
        res['updates'] = self.n_update - u0
        res['draws'] = self.n_draw - d0 - self._draws_by_capture
        med = self.last_stats.get('median_ms') or 16.7
        self.target_ms = med
        res['target_ms'] = med
        res['slip_ms'] = round(med * 1.08, 2)

    def st_capture(self, res):
        res['tried'] = {}
        sp = scene.SpriteNode(self.white, parent=self, position=(self.size[0] / 2, self.size[1] / 2),
                              size=(60, 60))
        sp.shader = scene.Shader(SOLID_SHADER)
        yield W(0.3, 0.1)
        want = np.array([round(c * 255) for c in SOLID])
        for m in self.cap.candidates():
            key = 'capture:' + m
            self.guard.mark(key)
            try:
                t0 = time.perf_counter()
                a = self.cap.grab(m)
                dt = time.perf_counter() - t0
                c, r = self.cap.px(self.size[0] / 2, self.size[1] / 2)
                got = a[r, c, :3].astype(int)
                ok = bool(np.abs(got - want).max() <= 4)
                res['tried'][m] = {'ok': ok, 'got': got.tolist(), 'want': want.tolist(),
                                   'ms': round(dt * 1000, 1), 'shape': list(a.shape)}
                if ok and not self.cap.method:
                    self.cap.method = m
                if self.cap.method and not SMOKE:      # a smoke run tries every method
                    self.guard.clear(key)
                    break
            except Exception:
                res['tried'][m] = {'ok': False, 'error': traceback.format_exc(limit=3)}
            self.guard.clear(key)
            yield W(0.1)
        sp.remove_from_parent()
        res['method'] = self.cap.method
        try:
            res['hierarchy'] = self.cap.hierarchy()
        except Exception:
            res['hierarchy'] = traceback.format_exc(limit=2)
        if not self.cap.method:
            return
        # colour response: a 256-level ramp in R and G, read back
        self.probe = scene.SpriteNode(self.white, parent=self, size=(256, 4),
                                      position=(self.size[0] / 2, self.size[1] / 2))
        self.probe.shader = scene.Shader(RAMP_SHADER)
        yield W(0.3, 0.1)
        a = self.cap.grab()
        lvl = self.read_strip(a, 256)
        self.lut_r = lvl[:, 0].astype(int)
        self.lut_g = lvl[:, 1].astype(int)
        res['ramp_r_err'] = int(np.abs(self.lut_r - np.arange(256)).max())
        res['ramp_g_err'] = int(np.abs(self.lut_g - np.arange(256)).max())
        res['ramp_monotonic'] = bool(np.all(np.diff(self.lut_r) >= 0) and np.all(np.diff(self.lut_g) >= 0))
        res['ramp_r_sample'] = self.lut_r[::32].tolist()

    def read_strip(self, a, n):
        """Values of an n-sample probe strip centred on screen (one per point)."""
        w, h = self.size
        x0 = w / 2 - self.probe.size[0] / 2
        out = []
        for i in range(n):
            c, r = self.cap.px(x0 + (i + 0.5) * self.probe.size[0] / n, h / 2)
            out.append(a[r, c, :3])
        return np.array(out)

    def decode(self, v):
        """Undo the capture's colour response, then the 16-bit R/G encoding -> 0..1."""
        def inv(lut, m):
            return float(np.argmin(np.abs(lut - int(m))))
        r = inv(self.lut_r, v[0]) if self.lut_r is not None else float(v[0])
        g = inv(self.lut_g, v[1]) if self.lut_g is not None else float(v[1])
        return (r + g / 255.0) / 255.0

    def check_specs(self):
        specs = [('LEGACY', orbit_spec('LEGACY'), orbit_pose())]
        for q in RAY:
            specs.append((q, orbit_spec(q, debug=('NANCHECK',)), orbit_pose()))
            specs.append((q + ' persp', orbit_spec(q, perspective=True, debug=('NANCHECK',)),
                          orbit_pose('chase/P')))
        for base in ('LOW', 'MED'):                 # LOW: one pass, MED: radiance + overlay
            for name, feats in FEATURE_SETS[:3 if QUICK else None]:
                specs.append(('%s feats=%s' % (base, name), orbit_spec(base, feats, debug=('NANCHECK',)),
                              orbit_pose()))
        for q in ('MED',) if QUICK else ('MED', 'HIGH', 'ULTRA'):
            for refr in (True, False):
                specs.append(('ground %s refr=%d' % (q, refr), ground_spec(q, refr, debug=('NANCHECK',)),
                              ground_pose(2.0)))
        return specs

    def st_compile(self, res):
        res['checks'] = []
        for name, spec, pose in self.check_specs():
            self.status('compile %s' % name)
            item = {'name': name, 'spec': spec, 'pose': pose}
            try:
                item['build_s'] = round(self.build(spec, pose), 3)
                yield from self.measure(0.0, W(0.6, 0.1))
                item['first_frames'] = self.last_stats
                rect = self.panel if spec['rect'] == 'panel' else self.full
                c = self.shot('compile_' + name.replace(' ', '_').replace('=', ''), rect)
                item['pixels'] = pixel_stats(c)
            except Exception:
                item['error'] = traceback.format_exc(limit=4)
            res['checks'].append(item)
            self.save()

    def st_nan(self, res):
        res['checks'] = []
        poses = [orbit_pose('overview', s) for s, _ in SUNS] + [orbit_pose(v) for v in VIEWS[1:]]
        poses += [ground_pose(e) for e in (5.0, 0.2, -0.4, -3.0)]
        if QUICK:
            poses = poses[::2]
        poses = pick(poses, [orbit_pose('overview', 'noon'), orbit_pose('astro'), ground_pose(-0.4)])
        for q in pick(RAY, ('MED',)):
            for pose in poses:
                spec = spec_for(pose, q, debug=('NANCHECK',))
                self.status('NaN scan %s' % q, '%s %s' % (pose['view'], pose.get('sun_name', pose.get('elev'))))
                self.build(spec, pose)
                yield W(0.35, 0.1)
                rect = self.panel if spec['rect'] == 'panel' else self.full
                st = pixel_stats(self.cap.crop(self.cap.grab(), rect))
                res['checks'].append({'quality': q, 'view': pose['view'],
                                      'sun': pose.get('sun_name', pose.get('elev')), 'nan': st['nan'],
                                      'bg': st['bg']})
        res['worst_nan'] = max([c['nan'] for c in res['checks']] or [0.0])

    def st_math(self, res):
        lib = refraction_lib()
        res['probes'] = []
        w, h = self.size
        if getattr(self, 'lut_r', None) is None:
            self.lut_r = self.lut_g = None
        for name, expr, x0, x1, lo, hi, km, ref, tol in probes():
            self.status('GPU maths', name)
            item = {'name': name, 'expr': expr}
            try:
                if self.probe is not None:
                    self.probe.remove_from_parent()
                self.probe = scene.SpriteNode(self.white, parent=self, size=(PROBE_N, 4),
                                              position=(w / 2, h / 2))
                sh = scene.Shader(probe_shader(lib, expr))
                self.probe.shader = sh
                for k, v in (('u_x0', x0), ('u_x1', x1), ('u_lo', lo), ('u_hi', hi)):
                    sh.set_uniform(k, float(v))
                sh.set_uniform('u_atm', (1.0 + 100.0 / km, float(km), 1.0, 0.0))
                sh.set_uniform('u_sun', (1.0, 0.0, 0.0))
                yield W(0.25, 0.1)
                vals = self.read_strip(self.cap.grab(), PROBE_N)
                xs = np.linspace(x0, x1, PROBE_N)
                got = np.array([lo + (hi - lo) * self.decode(v) for v in vals])
                q = (hi - lo) / 65025.0
                item['quantum'] = q
                if ref is not None:
                    want = np.array([ref(x) for x in xs])
                    err = np.abs(got - want)
                    item['max_err'] = float(err.max())
                    item['at_x'] = float(xs[int(np.argmax(err))])
                    item['tol'] = max(tol, 4 * q)
                    item['ok'] = bool(err.max() < item['tol'])
                else:
                    item['values'] = [round(float(v), 4) for v in got[::8]]
                    item['xs'] = [round(float(x), 3) for x in xs[::8]]
                item['sample'] = [round(float(v), 5) for v in got[::16]]
            except Exception:
                item['error'] = traceback.format_exc(limit=4)
            res['probes'].append(item)
        res['all_ok'] = all(p.get('ok', True) and 'error' not in p for p in res['probes'])
        nan = {p['name']: p.get('values') for p in res['probes'] if p['name'].startswith('sees NaN')}
        # x < 0 is the first half of each strip: 0 there means the NaN was seen
        res['nan_visible'] = {k: (v is not None and min(v[:len(v) // 2]) < 0.5) for k, v in nan.items()}

    def st_screens(self, res):
        res['shots'] = []
        poses = [orbit_pose('overview', 'noon'), orbit_pose('overview', 'terminator'), orbit_pose('chase')]
        if QUICK:
            poses = poses[:2]
        jobs = [(q, p) for q in ('LEGACY',) + RAY for p in poses
                if not (q == 'LEGACY' and p['view'] != 'overview')]
        jobs += [(q, ground_pose(e)) for q in ('MED', 'ULTRA') for e in (0.5, -0.3)]
        jobs = pick(jobs, [('LOW', orbit_pose('overview', 'noon')), ('MED', ground_pose(0.5))])
        for q, pose in jobs:
            spec = spec_for(pose, q)
            name = 'screen_%s_%s_%s' % (q, pose['view'].replace('/', ''),
                                        pose.get('sun_name', 'e%+.1f' % pose.get('elev', 0)))
            self.status('screens', name)
            self.build(spec, pose)
            yield W(0.6, 0.15)
            rect = self.panel if spec['rect'] == 'panel' else self.full
            t = float(self.t)
            self.shot(name, rect, full_res=True)
            res['shots'].append({'file': 'screens/%s.png' % name, 'spec': spec, 'pose': pose,
                                 'rect': list(rect), 'scene_t': round(t, 2), 'scale': self.cap.scale,
                                 'saved_scale': min(SCREEN_SCALE, self.cap.scale)})
            self.save()

    def timed(self, spec, pose, seconds, copies=0):
        """Generator: build, settle, time; self.last_stats gets build/first-frame too."""
        b = self.build(spec, pose, copies)
        yield from self.measure(W(0.5), seconds)
        self.last_stats['build_s'] = round(b, 3)

    def st_bench(self, res):
        res['runs'] = []
        slip = getattr(self, 'target_ms', 16.7) * 1.08
        secs = W(1.5 if QUICK else 2.5, 0.2)
        for view in pick(VIEWS, ('overview', 'astro')):
            for q in pick(('LEGACY',) + RAY, ('LEGACY', 'LOW')):
                if q == 'LEGACY' and view in ('chase/P', 'astro'):
                    continue
                pose = orbit_pose(view)
                spec = spec_for(pose, q)
                self.status('bench %s %s' % (q, view))
                yield from self.timed(spec, pose, secs)
                run = {'quality': q, 'view': view, 'stats': self.last_stats}
                head = 0
                if self.last_stats.get('mean95_ms', 99) < slip and q != 'LEGACY':
                    for k in pick(HEADROOM_STEPS, HEADROOM_STEPS[:1]):
                        self.status('headroom %s %s +%d' % (q, view, k))
                        yield from self.timed(spec, pose, W(1.2, 0.1), copies=k)
                        if self.last_stats.get('mean95_ms', 99) > slip:
                            break
                        head = k
                    run['headroom'] = head + 1
                res['runs'].append(run)
                self.save()
        for q in pick(('MED', 'HIGH', 'ULTRA'), ('MED',)):
            pose = ground_pose(2.0)
            self.status('bench ground %s' % q)
            yield from self.timed(ground_spec(q), pose, secs)
            res['runs'].append({'quality': q, 'view': 'ground', 'stats': self.last_stats})
        res['slip_ms'] = round(slip, 2)

    def st_refraction(self, res):
        """GPU-bound comparisons, so frame time tracks shader cost."""
        res['pairs'] = []
        secs = W(2.0 if QUICK else 3.0, 0.1)
        cases = [('ground ULTRA', lambda r: ground_spec('ULTRA', r), ground_pose(2.0), 0),
                 ('ground MED', lambda r: ground_spec('MED', r), ground_pose(2.0), 0),
                 ('orbit HIGH x4', lambda r: orbit_spec('HIGH', [f for f in self.preset_feats('HIGH')
                                                                 if r or f != 'REFRACTION']),
                  orbit_pose('overview', 'terminator'), 3)]
        for name, mk, pose, copies in cases:
            pair = {'case': name}
            for r in (True, False, True):               # on, off, on again (drift check)
                self.status('refraction cost', '%s %s' % (name, 'on' if r else 'off'))
                yield from self.timed(mk(r), pose, secs, copies)
                pair.setdefault('on' if r else 'off', []).append(self.last_stats)
            res['pairs'].append(pair)
            self.save()

    @staticmethod
    def preset_feats(q):
        from orbitgfx.shaders import FEATURES
        return list(FEATURES) + list(QUALITY[q].get('features', ()))

    def st_weather(self, res):
        res['runs'] = []
        for warp in pick((0.0, 3600.0, 14400.0), (0.0, 14400.0)):   # cloud seconds per second
            pose = orbit_pose('overview', 'noon')
            pose['warp'] = warp
            self.status('weather film', 'x%d' % warp)
            yield from self.timed(orbit_spec('MED'), pose, W(3.0 if QUICK else 6.0, 0.3))
            film = getattr(self.globes[0], 'film', None)
            res['runs'].append({'warp': warp, 'stats': self.last_stats,
                                'film_frames_cached': len(getattr(film, '_cache', []) or [])})
            self.save()

    # ------------------------------------------------------------ the end
    def finish(self):
        self.report['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save()
        text = summarize(self.report)
        print(text)
        try:
            with open(os.path.join(self.dir, 'summary.txt'), 'w') as fh:
                fh.write(text + '\n')
        except Exception:
            pass
        try:
            import zipfile
            self.zip_path = self.dir + '.zip'
            with zipfile.ZipFile(self.zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(self.dir):
                    for f in files:
                        if not f.startswith('.'):
                            p = os.path.join(root, f)
                            z.write(p, os.path.relpath(p, os.path.dirname(self.dir)))
        except Exception as e:
            self.zip_path = None
            self.report['errors']['zip'] = repr(e)
        try:
            import clipboard
            clipboard.set(text)
        except Exception:
            pass
        try:
            os.remove(self.guard.path)
        except Exception:
            pass
        self.status('DONE - tap to share the report', os.path.basename(self.zip_path or self.dir))
        y = self.size[1] - TOP_INSET - 50
        for ln in text.splitlines()[:40]:
            lab = scene.LabelNode(ln, ('Menlo', 7), parent=self, color=(0.8, 0.9, 1.0), position=(6, y))
            lab.anchor_point = (0, 0.5)
            y -= 9


if __name__ == '__main__':
    scene.run(DeviceCheck(), orientation=getattr(scene, 'PORTRAIT', 0), show_fps=False)
