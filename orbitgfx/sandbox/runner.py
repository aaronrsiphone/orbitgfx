"""Drive a Pythonista Scene on the desktop: deterministic clock, frames, touches, PNGs.

    from orbitgfx.sandbox import Runner
    r = Runner(orbitgfx.EXAMPLES_DIR + '/mission_control.py', 'Mission', size=(390, 844))
    r.advance(2.0)                    # 2 s of frames at 60 fps (update + draw)
    r.tap(195, 600)                   # goes through Pythonista's touch dispatch
    r.drag([(100, 600), (220, 640)])  # single-finger drag
    r.pinch((195, 600), 60, 140)      # two-finger pinch out
    r.render('frame.png')

The wall clock is virtual: time.monotonic() / time.time() return runner time
while a Runner is alive, so animations and tap timing are reproducible.
"""
import importlib.util
import os
import sys
import time as _time

HERE = os.path.dirname(os.path.abspath(__file__))          # .../orbitgfx/sandbox
SHIM = os.path.join(HERE, 'shim')                            # emulated scene / ui
PKG_PARENT = os.path.dirname(os.path.dirname(HERE))          # folder holding orbitgfx/


def _install_paths(app_dir=None):
    """shim first (scene, ui), then orbitgfx's parent, then the app's folder."""
    for p in (SHIM, PKG_PARENT):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, PKG_PARENT)
    sys.path.insert(0, SHIM)
    if app_dir and app_dir not in sys.path:
        sys.path.insert(2, app_dir)



_install_paths()
import scene  # noqa: E402  (the sandbox scene)


class VirtualClock(object):
    def __init__(self, start_monotonic=1000.0, start_unix=None):
        self.mono = start_monotonic
        self.unix = _time.time() if start_unix is None else start_unix
        self._orig = None

    def install(self):
        self._orig = (_time.monotonic, _time.time, _time.perf_counter)
        _time.monotonic = lambda: self.mono
        _time.perf_counter = lambda: self.mono
        _time.time = lambda: self.unix

    def uninstall(self):
        if self._orig:
            _time.monotonic, _time.time, _time.perf_counter = self._orig
            self._orig = None

    def tick(self, dt):
        self.mono += dt
        self.unix += dt


def load_scene_class(path_or_module, cls_name):
    if hasattr(path_or_module, cls_name):
        return getattr(path_or_module, cls_name)
    path = os.path.abspath(path_or_module)
    _install_paths(os.path.dirname(path))
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


class Runner(object):

    def __init__(self, app, cls_name='Mission', size=(390, 844), scale=3.0, fps=60.0,
                 unix_time=None, setup=True, before_setup=None):
        """app: path to a .py file (or a module) defining `cls_name`.
        before_setup(module_or_cls): hook to tweak module constants (e.g. a fixed epoch).
        """
        scene.SCALE = scale
        self.fps = fps
        self.clock = VirtualClock(start_unix=unix_time)
        self.clock.install()
        self.cls = load_scene_class(app, cls_name)
        if before_setup:
            before_setup(sys.modules[self.cls.__module__])
        self.scene = self.cls()
        self.scene._fw_set(size=scene.Size(*size), bounds=scene.Rect(0, 0, *size))
        self._next_id = 1
        self.frames = 0
        if setup:
            self.scene.setup()

    def close(self):
        self.clock.uninstall()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- time
    def frame(self, dt=None, draw=True):
        """One Pythonista frame: advance t/dt, update(), draw()."""
        dt = 1.0 / self.fps if dt is None else dt
        self.clock.tick(dt)
        s = self.scene
        s._fw_set(dt=dt, t=s.t + dt)
        s.update()
        if draw:
            s.draw()               # classic calls go to a null canvas (counted only)
        self.frames += 1

    def advance(self, seconds, draw=True):
        n = max(1, int(round(seconds * self.fps)))
        for _ in range(n):
            self.frame(draw=draw)

    def resize(self, w, h):
        self.scene._fw_set(size=scene.Size(w, h), bounds=scene.Rect(0, 0, w, h))
        self.scene.did_change_size()

    # ---------------------------------------------------------------- touches
    def _touch(self, kind, tid, x, y, px, py):
        t = scene.Touch(x, y, px, py, tid)
        getattr(self.scene, '_touch_' + kind)(t)

    def tap(self, x, y, hold=0.05):
        tid = self._next_id; self._next_id += 1
        self._touch('began', tid, x, y, x, y)
        self.advance(hold)
        self._touch('ended', tid, x, y, x, y)
        self.frame()

    def double_tap(self, x, y, gap=0.12):
        self.tap(x, y)
        self.advance(gap)
        self.tap(x, y)

    def drag(self, points, duration=0.3, release=True):
        """Single finger along a polyline of scene points."""
        tid = self._next_id; self._next_id += 1
        pts = [tuple(p) for p in points]
        steps = max(2, int(duration * self.fps))
        path = []
        for i in range(steps + 1):
            u = i / float(steps) * (len(pts) - 1)
            k = min(int(u), len(pts) - 2)
            f = u - k
            path.append((pts[k][0] + (pts[k + 1][0] - pts[k][0]) * f,
                         pts[k][1] + (pts[k + 1][1] - pts[k][1]) * f))
        self._touch('began', tid, path[0][0], path[0][1], path[0][0], path[0][1])
        for a, b in zip(path, path[1:]):
            self.frame()
            self._touch('moved', tid, b[0], b[1], a[0], a[1])
        if release:
            self._touch('ended', tid, path[-1][0], path[-1][1], path[-1][0], path[-1][1])
            self.frame()

    def pinch(self, center, d0, d1, duration=0.3):
        """Two fingers on a horizontal line through `center`, from spread d0 to d1."""
        cx, cy = center
        a, b = self._next_id, self._next_id + 1
        self._next_id += 2
        self._touch('began', a, cx - d0 / 2, cy, cx - d0 / 2, cy)
        self._touch('began', b, cx + d0 / 2, cy, cx + d0 / 2, cy)
        steps = max(2, int(duration * self.fps))
        prev = d0
        for i in range(1, steps + 1):
            d = d0 + (d1 - d0) * i / float(steps)
            self.frame()
            self._touch('moved', a, cx - d / 2, cy, cx - prev / 2, cy)
            self._touch('moved', b, cx + d / 2, cy, cx + prev / 2, cy)
            prev = d
        self._touch('ended', a, cx - d1 / 2, cy, cx - d1 / 2, cy)
        self._touch('ended', b, cx + d1 / 2, cy, cx + d1 / 2, cy)
        self.frame()

    # ---------------------------------------------------------------- output
    def render(self, path, u_time=None):
        """Draw the current state to a PNG at SCALE (default 3x, like an iPhone)."""
        return scene.render(self.scene, path, u_time=u_time)
