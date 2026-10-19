"""Minimal orbitgfx globe (the README example): drag, pinch, 600x time lapse."""
import os, sys, time
try:                                   # installed (e.g. Pythonista site-packages)
    import orbitgfx  # noqa: F401
except ImportError:                    # running from inside an unzipped orbitgfx folder
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import orbitgfx  # noqa: F401      (on a desktop this also installs the scene shim)
import scene
from orbitgfx import GlobeView, Camera, Gestures, cached_earth_image, pil_texture, safety
from orbitgfx.astro import gmst, julian_date, sun_dir, OMEGA_E

class Demo(scene.Scene):
    def setup(self):
        safety.assert_clean(type(self))          # catches self.dt / self.touches bugs
        from orbitgfx.coastlines import coastlines_xyz
        tex = pil_texture(cached_earth_image(coastlines_xyz(6378.137)))
        self.cam = Camera(el=0.4)
        self.globe = GlobeView(self, tex, features=('STARS', 'ATMOSPHERE', 'CLOUDS',
                                                   'LIGHTS', 'AURORA', 'BIOMES', 'GLINT'))
        self.globe.layout((10, 10, self.size.w - 20, self.size.h - 20))
        self.gest = Gestures(lambda: [('globe', self.globe.rect)], pinch_regions=('globe',))
        self.gest.on_press = lambda r, now: self.cam.grab(now)
        self.gest.on_drag = lambda r, dx, dy, now: self.cam.drag(dx, dy, now)
        self.gest.on_release = lambda r, now, tap: self.cam.release(now)
        self.gest.on_pinch_begin = lambda r: self.cam.pinch_begin()
        self.gest.on_pinch = lambda r, k: self.cam.pinch(k)
        self.jd0 = julian_date()
        self.step_s = 1 / 60                      # NOT self.dt: Scene owns dt

    def update(self):
        self.gest.poll()
        self.cam.held = self.gest.active
        self.cam.step(self.step_s)
        hours = self.t * 600 / 3600               # 600x time lapse
        jd = self.jd0 + hours / 24
        self.globe.update(self.cam, sun_dir(jd), gmst(jd), cloud_hours=hours)

    def touch_began(self, t): self.gest.began(t)
    def touch_moved(self, t): self.gest.moved(t)
    def touch_ended(self, t): self.gest.ended(t)

if __name__ == '__main__':
    scene.run(Demo())
