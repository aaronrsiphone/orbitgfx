"""orbitgfx 2.0 on-device check (phase 1+): timing per preset, then interactive.

1. For ~3 minutes it times every preset in three views (overview, chase 2.6x,
   chase in perspective). Where a preset holds 60 fps it then stacks extra
   copies of the globe until frames slip, to measure the headroom: "x3" means
   three globes still fit in a 60 fps frame, so one globe uses at most a third
   of the GPU. Then it copies a results block to the clipboard and saves
   phase_check_results.txt next to this script. Don't touch the screen.
2. Then it is interactive:  tap = next quality,  double tap = ortho / perspective /
   astronaut (riding behind the spacecraft; the orbit is 4x real time here),
   drag = rotate,  pinch = zoom,  tap the top line = next sun geometry,
   tap below the globe = weather time warp (30x, 600x, 3600x: one hour per second).
   The top line shows quality, projection, sun and live frame time.
"""
import math
import os
import sys
import time

import numpy as np

try:
    import orbitgfx  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import orbitgfx  # noqa: F401
import scene
from orbitgfx import GlobeView, Camera, Gestures, cached_earth_image, pil_texture, safety
from orbitgfx.astro import RE

HERE = os.path.dirname(os.path.abspath(__file__))
TOP_INSET = 64                    # clear of the Dynamic Island
QUALITIES = ('LEGACY', 'LOW', 'MED', 'HIGH', 'ULTRA')
HEADROOM_STEPS = (1, 2, 3, 5, 7)   # extra globes tried, stop at the first that drops frames
SLIP_MS = 17.5
WARPS = (30.0, 600.0, 3600.0)
SUNS = (('noon', 20.0), ('terminator', 90.0), ('backlit', 172.0), ('night', 130.0))
VIEWS = ('overview', 'chase', 'chase/P', 'astro')
BG = (0.023, 0.039, 0.078)


def orbit(t, inc=51.6, raan=0.4, alt=500.0):
    """Circular orbit at time t (s): ECI km, km/s."""
    r = RE + alt
    n = math.sqrt(398600.4418 / r ** 3)
    u = 0.9 + n * t
    i = math.radians(inc)
    p = np.array([math.cos(u), math.sin(u), 0.0]) * r
    v = np.array([-math.sin(u), math.cos(u), 0.0]) * n * r
    Rx = np.array([[1, 0, 0], [0, math.cos(i), -math.sin(i)], [0, math.sin(i), math.cos(i)]])
    Rz = np.array([[math.cos(raan), -math.sin(raan), 0], [math.sin(raan), math.cos(raan), 0], [0, 0, 1]])
    M = Rz @ Rx
    return M @ p, M @ v


def mission_panel(w, h):
    """The portrait globe panel of examples/mission_control.py."""
    m, head, foot = 10.0, 34.0, 22.0
    fs = min(1.1, max(0.7, min(w, 520.0) / 400.0))
    tele_h = 5 * 14 * fs + 32
    top = h - head
    mw = w - 2 * m
    mh = mw / 2
    if mh > (top - foot) * 0.40:
        mh = (top - foot) * 0.40
    gy = foot + mh + 6 + tele_h + 6
    return (m, gy, w - 2 * m, max(80.0, top - gy))


class Check(scene.Scene):

    def setup(self):
        safety.assert_clean(type(self))
        self.background_color = BG
        w, h = self.size
        self.panel = mission_panel(w, h)
        self.tex = pil_texture(cached_earth_image(__import__('orbitgfx.coastlines', fromlist=['x'])
                                                  .coastlines_xyz(RE)))
        self.cam = Camera(az=0.55, el=0.42)
        self.qi, self.si, self.persp = 2, 0, False
        self.globe = None
        self.extra = []
        from orbitgfx import luts
        self.bakes = []
        for name, fn in (('weather', luts.weather_lut), ('noise3d', luts.noise_lut)):
            t0 = time.perf_counter()
            fn()
            self.bakes.append('%s %.2f s' % (name, time.perf_counter() - t0))
        self.build()
        self.hdr = scene.LabelNode('', ('Menlo-Bold', 12), parent=self, color='white',
                                   position=(w / 2, h - TOP_INSET - 12))
        self.sub = scene.LabelNode('', ('Menlo', 10), parent=self, color=(0.7, 0.8, 0.9),
                                   position=(w / 2, h - TOP_INSET - 30))
        self.res_labels = []
        self.wi = 0
        self.gest = Gestures(lambda: (('top', (0, h - TOP_INSET - 45, w, 45)), ('globe', self.panel),
                                      ('warp', (0, 0, w, self.panel[1]))),
                             double_tap_regions=('globe',), pinch_regions=('globe',))
        self.gest.on_press = lambda r, now: self.cam.grab(now)
        self.gest.on_drag = lambda r, dx, dy, now: self.cam.drag(dx, dy, now)
        self.gest.on_release = lambda r, now, tap: self.cam.release(now)
        self.gest.on_pinch_begin = lambda r: self.cam.pinch_begin()
        self.gest.on_pinch = lambda r, k: self.cam.pinch(k)
        self.gest.on_pinch_end = lambda r: self.cam.pinch_end()
        self.gest.on_tap = self.on_tap
        self.gest.on_double_tap = self.on_double
        self.sim_t = 0.0
        self.orb_t = 0.0
        self.prev = None
        self.frames = []
        self.bench = self.run_bench()
        self.bench_done = False
        self.resume_at = 0.0
        self.samples = None
        self.results = []

    # ------------------------------------------------------------ globe
    def build(self, copies=0):
        for g in [self.globe] + self.extra:
            if g is not None:
                g.remove()
        self.extra = []
        self.globe = GlobeView(self, self.tex, quality=QUALITIES[self.qi], fill=0.44,
                               ref_radius_km=RE + 500.0, perspective=self.persp)
        self.globe.layout(self.panel)
        for k in range(copies):                 # load test: identical globes drawn on top
            g = GlobeView(self, self.tex, quality=QUALITIES[self.qi], fill=0.44,
                          ref_radius_km=RE + 500.0, perspective=self.persp, z_position=-0.9 + 0.01 * k)
            g.layout(self.panel)
            self.extra.append(g)

    def sun(self):
        M = self.cam.matrix()
        a = math.radians(SUNS[self.si][1])
        s = M[2] * math.cos(a) + M[1] * math.sin(a) * 0.35 + M[0] * math.sin(a)
        return s / np.linalg.norm(s)

    def set_view(self, view):
        sat, _ = orbit(self.orb_t)
        self.cam.follow = self.cam.astro = view == 'astro'
        if view == 'overview':
            self.cam.az, self.cam.el, self.cam.zoom = 0.55, 0.42, 1.0
        else:
            self.cam.az = math.atan2(sat[1], sat[0]) - 0.35 / 2.6
            self.cam.el = math.asin(sat[2] / np.linalg.norm(sat)) + 0.30 / 2.6
            self.cam.zoom = 2.6
        self.persp = view.endswith('/P') or view == 'astro'

    # ------------------------------------------------------------ input
    def on_tap(self, region):
        if not self.bench_done:
            return
        if region == 'top':
            self.si = (self.si + 1) % len(SUNS)
        elif region == 'warp':
            self.wi = (self.wi + 1) % len(WARPS)
        else:
            self.qi = (self.qi + 1) % len(QUALITIES)
            self.build()

    def on_double(self, region):
        if self.bench_done:
            # ortho -> perspective -> astronaut -> ortho
            if self.cam.astro:
                self.cam.follow = self.cam.astro = False
                self.persp = False
            elif self.persp:
                self.cam.follow = self.cam.astro = True
            else:
                self.persp = True
            self.build()

    def touch_began(self, t): self.gest.began(t)
    def touch_moved(self, t): self.gest.moved(t)
    def touch_ended(self, t): self.gest.ended(t)

    # ------------------------------------------------------------ frame
    def update(self):
        now = time.perf_counter()
        if self.prev is not None:
            d = now - self.prev
            self.frames = (self.frames + [d])[-120:]
            if self.samples is not None:
                self.samples.append(d)
        self.prev = now
        self.gest.poll()
        self.cam.held = self.gest.active
        self.sim_t += 1 / 60.0 * (WARPS[self.wi] if self.bench_done else 30.0)
        # the spacecraft clock: weather-warp speed, but 4x real time in the astronaut view
        self.orb_t += 1 / 60.0 * (4.0 if self.cam.astro else WARPS[self.wi] if self.bench_done else 30.0)
        sat, vel = orbit(self.orb_t)
        self.cam.step(1 / 60.0, sat, vel)
        st = [((RE * 0.62, RE * 0.55, RE * 0.56), True), ((-RE * 0.4, RE * 0.75, RE * 0.52), False)]
        for g in [self.globe] + self.extra:
            g.update(self.cam, self.sun(), 1.3 + self.sim_t * 7.29e-5, sat=sat, vel=vel,
                     trail_rad=0.8, stations=st, cloud_hours=self.sim_t / 3600.0)
        if not self.bench_done:
            if now >= self.resume_at:
                try:
                    self.resume_at = now + (next(self.bench) or 0.0)
                except StopIteration:
                    self.bench_done = True
                    self.finish()
            return
        f = np.asarray(self.frames) * 1000.0
        if self.cam.astro and QUALITIES[self.qi] == 'LEGACY':
            self.cam.follow = self.cam.astro = False
        proj = 'astro' if self.cam.astro else 'persp' if self.persp else 'ortho'
        self.hdr.text = '%s  %s  sun:%s  x%d' % (QUALITIES[self.qi], proj,
                                                 SUNS[self.si][0], WARPS[self.wi])
        if len(f) > 10:
            self.sub.text = 'frame %.1f ms  p95 %.1f  (tap / 2x tap / tap here)' % (
                f.mean(), np.percentile(f, 95))

    def run_bench(self):
        for view in VIEWS:
            for qi, q in enumerate(QUALITIES):
                if q == 'LEGACY' and (view.endswith('/P') or view == 'astro'):
                    continue
                self.set_view(view)
                self.qi = qi
                self.build()
                self.hdr.text = 'timing %s  %s' % (q, view)
                self.sub.text = "don't touch the screen"
                yield 0.5
                self.samples = []
                yield 2.5
                d = np.sort(np.asarray(self.samples) * 1000.0)
                self.samples = None
                keep = d[:max(1, int(len(d) * 0.95))]
                mean = float(keep.mean())
                head = 0
                if mean < SLIP_MS and q != 'LEGACY':
                    for k in HEADROOM_STEPS:
                        self.build(copies=k)
                        self.hdr.text = 'headroom %s  %s  +%d globes' % (q, view, k)
                        yield 0.6
                        self.samples = []
                        yield 1.4
                        dk = np.sort(np.asarray(self.samples) * 1000.0)
                        self.samples = None
                        if float(dk[:max(1, int(len(dk) * 0.95))].mean()) > SLIP_MS:
                            break
                        head = k
                    self.build()
                self.results.append((q, view, mean, float(np.percentile(d, 95)), len(d),
                                     ('x%d' % (head + 1)) if mean < SLIP_MS and q != 'LEGACY' else '-'))
        self.set_view('overview')
        self.persp = False
        self.qi = 2
        self.build()

    def finish(self):
        lines = ['ORBITGFX PHASE CHECK v3  orbitgfx %s' % orbitgfx.__version__,
                 'panel %.0fx%.0f pt @%g   bake: %s' % (self.panel[2], self.panel[3],
                                                        scene.get_screen_scale(), ', '.join(self.bakes))]
        lines += ['%-6s %-9s mean %5.1f ms  p95 %5.1f  (%d fr)  headroom %s' % r for r in self.results]
        if (HEADROOM_STEPS[-1] + 1) and any(r[5] == 'x%d' % (HEADROOM_STEPS[-1] + 1) for r in self.results):
            lines.append('(x%d = the most tried)' % (HEADROOM_STEPS[-1] + 1))
        text = '\n'.join(lines)
        print(text)
        try:
            with open(os.path.join(HERE, 'phase_check_results.txt'), 'w') as fh:
                fh.write(text + '\n')
        except Exception:
            pass
        try:
            import clipboard
            clipboard.set(text)
        except Exception:
            pass
        y = self.panel[1] - 14
        for ln in lines:
            lab = scene.LabelNode(ln, ('Menlo', 8), parent=self, color=(0.8, 0.9, 1.0),
                                  position=(8, y))
            lab.anchor_point = (0, 0.5)
            self.res_labels.append(lab)
            y -= 11


if __name__ == '__main__':
    scene.run(Check(), orientation=getattr(scene, 'PORTRAIT', 0), show_fps=True)
