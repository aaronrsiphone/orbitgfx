import scene
from orbitgfx import GlobeView, Camera, cached_earth_image, pil_texture
from orbitgfx.astro import gmst, julian_date, sun_dir
from orbitgfx.coastlines import coastlines_xyz


class Earth(scene.Scene):
    def setup(self):
        texture = pil_texture(cached_earth_image(coastlines_xyz(6378.137)))
        self.globe = GlobeView(self, texture, quality='HIGH')
        self.globe.layout((0, 0, self.size.w, self.size.h))
        self.camera = Camera(az=-2.0, el=0.45)
        self.jd0 = julian_date()                      # now

    def update(self):
        hours = self.t * 600 / 3600                   # 600x time lapse
        jd = self.jd0 + hours / 24
        self.globe.update(self.camera, sun_dir(jd), gmst(jd), cloud_hours=hours)


scene.run(Earth())
