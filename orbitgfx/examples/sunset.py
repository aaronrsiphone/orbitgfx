"""Sunset from the ground, under different kinds of cloud (orbitgfx 2.0 ray core).

The camera stands on the surface and looks toward the setting sun. The clouds are
the ones the precomputed weather film has at that place and hour, so every site
shows a different kind of sky. Scale is not to scale: clouds are drawn in the
same shell and noise as from orbit, the sun's disc is twice its size.

    tap the top     next site
    tap the middle  atmospheric refraction on / off
    tap the bottom  pause / 60x / 240x
    drag            look around       double tap   ULTRA / HIGH / MED

With refraction on (the default) the air lifts the setting sun by about half a
degree and squashes it into an oval, and sunlight lasts a couple of minutes longer:
the ground, sea glint and cloud shadows are lit by the lifted sun, and clouds high up
keep their light after the ground's sunset (their light grazed the dense low air).
The caption shows the true elevation and, with refraction, where the sun is seen.
Toggling rebuilds the globe: REFRACTION is a compile-time shader feature.
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
from orbitgfx import GlobeView, Camera, Gestures, cached_earth_image, pil_texture, safety
from orbitgfx.astro import RE, refraction_deg

TOP_INSET = 64
EYE_KM = 0.4                       # standing on a hill
FOV = 75.0
EXPOSURE = 0.8                     # the sky from below is far brighter than Earth from orbit
START_ELEV = 7.0                   # sun elevation when a site starts (degrees)
END_ELEV = -9.0                    # then it loops
SPEEDS = (0.0, 60.0, 240.0)        # simulated seconds per second
DECL = math.radians(-9.0)          # the weather film's season
OMEGA = 2.0 * math.pi / 86164.0
QUALITIES = ('ULTRA', 'HIGH', 'MED')
# cloud noise 8x finer than the orbit view: cells of ~10 km, detail ~1 km, so clouds
# have shapes you can see from under them (the "not to scale" part)
GROUND = {'LIMB_GAIN': '1.0', 'NOISE_F': '640.0', 'NOISE_FD': '5200.0', 'NOISE_V': '72.0',
          'CI_F': '440.0', 'CI_WF': '240.0', 'BASE_AMT': '0.85',
          'GROUNDCAM': '1'}      # the eye is inside the air: compiles the ground-camera code
# (name, latitude, longitude, weather-film frame): picked from the film where the
# cloud the camera looks at, just toward the sun, is of that kind at sunset
SITES = (
    ('Canary Current: stratocumulus deck', 23.0, -18.9, 43),
    ('Off northern Chile: open-cell stratocumulus', -22.0, -71.3, 47),
    ('Philippine Sea: trade cumulus', 14.0, 147.7, 32),
    ('Bismarck Sea: thunderstorms', -7.0, 151.1, 56),
    ('North Pacific: frontal overcast', 38.0, -142.1, 27),
    ('Amazon: high cirrus', -10.0, -58.4, 46),
    ('Arabia: clear desert sky', 20.0, 41.7, 15),
)
FEATURES = ('STARS', 'CLOUDS', 'LIGHTS', 'GLINT', 'BIOMES', 'TONEMAP', 'SCATTER', 'WEATHER', 'VCLOUDS',
            'SURFACE', 'RELIEF', 'PENUMBRA', 'GGX', 'SKYREFLECT', 'REFRACTION')


def unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array((math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)))


def sun_dir(sub_lon):
    return np.array((math.cos(DECL) * math.cos(sub_lon), math.cos(DECL) * math.sin(sub_lon), math.sin(DECL)))


class Sunset(scene.Scene):

    def setup(self):
        safety.assert_clean(type(self))
        self.background_color = (0.0, 0.0, 0.0)
        w, h = self.size
        self.tex = pil_texture(cached_earth_image(__import__('orbitgfx.coastlines', fromlist=['x'])
                                                  .coastlines_xyz(RE)))
        self.cam = Camera(fov_deg=FOV)
        self.qi, self.si, self.rate_i = 0, 0, 1
        self.refraction = True
        self.yaw, self.pitch = 0.0, math.radians(6.0)
        self.globe = None
        self.build()
        self.title = scene.LabelNode('', ('Menlo-Bold', 13), parent=self, color='white',
                                     position=(w / 2, h - TOP_INSET - 10))
        self.sub = scene.LabelNode('', ('Menlo', 11), parent=self, color=(0.85, 0.9, 1.0),
                                   position=(w / 2, h - TOP_INSET - 28))
        self.gest = Gestures(lambda: (('top', (0, h * 0.75, w, h * 0.25)), ('mid', (0, h * 0.2, w, h * 0.55)),
                                      ('bottom', (0, 0, w, h * 0.2))),
                             double_tap_regions=('mid',))
        self.gest.on_drag = self.on_drag
        self.gest.on_tap = self.on_tap
        self.gest.on_double_tap = self.on_double
        self.start_site()

    def build(self):
        if self.globe is not None:
            self.globe.remove()
        w, h = self.size
        feats = FEATURES if self.refraction else tuple(f for f in FEATURES if f != 'REFRACTION')
        self.globe = GlobeView(self, self.tex, quality=QUALITIES[self.qi], features=feats,
                               perspective=True, atmo_scale=1.0, cloud_scale=1.0, inset=0.0, exposure=EXPOSURE,
                               defines=GROUND)
        self.globe.layout((0.0, 0.0, w, h))

    def start_site(self):
        name, lat, lon, k = SITES[self.si]
        self.up = unit(lat, lon)
        # the hour angle where the sun stands START_ELEV degrees up, going down
        lo, hi = 0.0, math.pi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            e = math.degrees(math.asin(float(np.dot(self.up, sun_dir(math.radians(lon) - mid)))))
            lo, hi = (mid, hi) if e > START_ELEV else (lo, mid)
        self.sub0 = math.radians(lon) - lo
        t_utc = ((math.pi - self.sub0) / (2.0 * math.pi)) % 1.0
        self.hours0 = (k // 24 + t_utc) * 24.0
        self.sim_s = 0.0
        self.yaw = 0.0

    # ------------------------------------------------------------ input
    def on_tap(self, region):
        if region == 'top':
            self.si = (self.si + 1) % len(SITES)
            self.start_site()
        elif region == 'bottom':
            self.rate_i = (self.rate_i + 1) % len(SPEEDS)
        elif region == 'mid':
            self.refraction = not self.refraction
            self.build()

    def on_double(self, region):
        self.qi = (self.qi + 1) % len(QUALITIES)
        self.build()

    def on_drag(self, region, dx, dy, now):
        self.yaw -= dx * 0.004
        self.pitch = min(math.radians(80.0), max(math.radians(-20.0), self.pitch - dy * 0.004))

    def touch_began(self, t): self.gest.began(t)
    def touch_moved(self, t): self.gest.moved(t)
    def touch_ended(self, t): self.gest.ended(t)

    # ------------------------------------------------------------ frame
    def update(self):
        self.gest.poll()
        self.sim_s += self.dt * SPEEDS[self.rate_i]
        sub = self.sub0 - OMEGA * self.sim_s
        sun = sun_dir(sub)
        elev = math.degrees(math.asin(float(np.dot(self.up, sun))))
        if elev < END_ELEV:
            self.sim_s = 0.0
            sub, sun = self.sub0, sun_dir(self.sub0)
        # local frame: east, north, up; look toward the sun's azimuth + yaw, pitched up
        e = np.cross((0.0, 0.0, 1.0), self.up)
        e /= np.linalg.norm(e)
        n = np.cross(self.up, e)
        az = math.atan2(float(np.dot(sun, e)), float(np.dot(sun, n))) + self.yaw
        fwd = (n * math.cos(az) + e * math.sin(az)) * math.cos(self.pitch) + self.up * math.sin(self.pitch)
        eye = self.up * (1.0 + EYE_KM / RE)
        self.cam.eye, self.cam.look = eye, eye + fwd
        self.cam.fov_deg = FOV
        self.globe.update(self.cam, sun, 0.0, cloud_hours=self.hours0 + self.sim_s / 3600.0)
        name = SITES[self.si][0]
        self.title.text = name
        seen = ' (seen %+.1f)' % (elev + refraction_deg(elev, EYE_KM)) if self.refraction else ''
        self.sub.text = 'sun %+.1f deg%s  %s  %s  refraction %s' % (
            elev, seen, ('paused', '60x', '240x')[self.rate_i], QUALITIES[self.qi],
            'on' if self.refraction else 'off')


if __name__ == '__main__':
    scene.run(Sunset(), orientation=getattr(scene, 'PORTRAIT', 0), show_fps=True)
