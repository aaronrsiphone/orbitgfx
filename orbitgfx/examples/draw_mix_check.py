"""Device check: a ray-core globe in a scene that also draws in draw().

mission_control.py draws its panels, stars and sun glyphs in draw() (immediate
mode). On the device the MED / HIGH globe then came out as a filled gold ellipse.
This shows one globe under the same conditions and lets you flip each suspect:

    tap the top third     next quality (LOW, MED, HIGH, ULTRA)
    tap the middle        immediate-mode drawing in draw() on / off
    tap the bottom third  primer sprite on / off (the 2.0.0a9 fix)

The top line shows the state. Report which combinations show the gold ellipse.
"""
import math
import os
import sys

import numpy as np

try:
    import orbitgfx  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import orbitgfx  # noqa: F401
import scene
from orbitgfx import GlobeView, Camera, cached_earth_image, pil_texture
from orbitgfx import views
from orbitgfx.astro import RE

QUALITIES = ('LOW', 'MED', 'HIGH', 'ULTRA')
GOLD = (1.0, 0.78, 0.34)


class Check(scene.Scene):

    def setup(self):
        self.background_color = (0.023, 0.039, 0.078)
        w, h = self.size
        self.tex = pil_texture(cached_earth_image(__import__('orbitgfx.coastlines', fromlist=['x'])
                                                  .coastlines_xyz(RE)))
        self.cam = Camera(az=0.55, el=0.42)
        self.qi, self.immediate, self.primer = 1, True, True
        self.globe = None
        self.lab = scene.LabelNode('', ('Menlo-Bold', 12), parent=self, color='white',
                                   position=(w / 2, h - 80))
        self.build()

    def build(self):
        if self.globe is not None:
            self.globe.remove()
        views.PRIMER = self.primer
        w, h = self.size
        self.globe = GlobeView(self, self.tex, quality=QUALITIES[self.qi], fill=0.44)
        self.globe.layout((10, h * 0.25, w - 20, h * 0.5))
        self.lab.text = '%s   draw(): %s   primer: %s' % (
            QUALITIES[self.qi], 'on' if self.immediate else 'off', 'on' if self.primer else 'off')

    def touch_ended(self, t):
        y = t.location.y / self.size.h
        if y > 2 / 3.0:
            self.qi = (self.qi + 1) % len(QUALITIES)
        elif y > 1 / 3.0:
            self.immediate = not self.immediate
        else:
            self.primer = not self.primer
        self.build()

    def update(self):
        self.cam.step(1 / 60.0)
        M = self.cam.matrix()
        sun = M[2] * 0.5 + M[0] * 0.85
        sun = sun / np.linalg.norm(sun)
        a = self.t * 0.2
        sat = np.array((math.cos(a), math.sin(a), 0.3)) * (RE + 500.0)
        vel = np.array((-math.sin(a), math.cos(a), 0.0)) * 7.6
        self.globe.update(self.cam, sun, 1.3, sat=sat, vel=vel, trail_rad=0.8, cloud_hours=3.0)

    def draw(self):
        if not self.immediate:
            return
        w, h = self.size
        scene.stroke(0.3, 0.5, 0.7); scene.stroke_weight(1); scene.no_fill()
        scene.rect(10, h * 0.25, w - 20, h * 0.5)
        scene.no_stroke(); scene.fill(*GOLD)
        scene.ellipse(30, 40, 16, 16)                # like mission_control's sun glyph


if __name__ == '__main__':
    scene.run(Check(), show_fps=True)
