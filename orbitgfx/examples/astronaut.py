import math
import numpy as np
import scene
from orbitgfx import GlobeView, Camera, cached_earth_image, pil_texture
from orbitgfx.astro import gmst, julian_date, sun_dir
from orbitgfx.coastlines import coastlines_xyz

R = 6378.137 + 420.0                                  # km: an ISS-like orbit
N = math.sqrt(398600.4418 / R ** 3)                   # mean motion, rad/s
INC = math.radians(51.6)


def orbit(t):
    """Position (km) and velocity (km/s) on a circular orbit, ECI."""
    u = N * t
    p = R * np.array((math.cos(u), math.sin(u) * math.cos(INC), math.sin(u) * math.sin(INC)))
    v = N * R * np.array((-math.sin(u), math.cos(u) * math.cos(INC), math.cos(u) * math.sin(INC)))
    return p, v


class Astronaut(scene.Scene):
    def setup(self):
        texture = pil_texture(cached_earth_image(coastlines_xyz(6378.137)))
        self.globe = GlobeView(self, texture, quality='HIGH', inset=0.0)
        self.globe.layout((0, 0, self.size.w, self.size.h))
        self.camera = Camera()
        self.jd0 = julian_date()

    def update(self):
        t = self.t * 10                               # 10x real time
        p, v = orbit(t)
        jd = self.jd0 + t / 86400
        self.camera.set_astronaut(p, v, down=0.3)     # ride behind it, looking down
        self.globe.update(self.camera, sun_dir(jd), gmst(jd), sat=p, vel=v,
                          cloud_hours=t / 3600)


scene.run(Astronaut())
