"""GPU views (SpriteNode + shader) and the orbit camera.

Pythonista draws the classic `draw()` layer FIRST and the node tree on top,
so a shader node covers anything drawn classically under it. GlobeView renders
everything inside its panel (Earth, orbit, spacecraft, links); outside the
globe its pixels are transparent, so classic panel art under it shows through.
"""
import math
import os
import warnings

import numpy as np
import scene

from .astro import RE, ecef_to_eci
from .shaders import (FEATURES, STATION_SLOTS, MAP_NIGHT_SHADER, build_globe_shader,
                      build_ray_shader)

# Quality presets (plan "Performance", adjusted after Phase 0 on an iPhone 16 Pro Max).
#   core:          'legacy' = the 1.x shader, 'ray' = the 2.0 ray core
#   render_scale:  resolution of the radiance pass (0.5 = EffectNode at half size)
#   the remaining keys are read by later phases (atmosphere, clouds, lighting)
# phase 4 ray-traced lighting on every ray preset: soft Earth shadow with the sunset
# colour, microfacet sun glint, lit spacecraft (sun + earthshine)
LIGHTING = ('PENUMBRA', 'GGX', 'CRAFTLIGHT', 'REFRACTION')
# phase 5: NASA Blue Marble surface + Black Marble lights + sea ice, terrain relief
TERRAIN = ('SURFACE', 'RELIEF')
QUALITY = {
    'LEGACY': dict(core='legacy', render_scale=1.0),
    'LOW':  dict(core='ray', render_scale=1.0, features=('SCATTER', 'WEATHER') + LIGHTING + TERRAIN, atmo_samples=6,
                 cloud_steps=0, light_steps=0, detail_noise=False, reflection=0, relief_steps=0),
    'MED':  dict(core='ray', render_scale=0.5, features=('SCATTER', 'WEATHER', 'VCLOUDS', 'SKYREFLECT') + LIGHTING
                 + TERRAIN,
                 atmo_samples=10,
                 cloud_steps=24, cloud_steps_base=8, cloud_steps_min=6, cloud_slices=10, light_steps=4, ms_octaves=2,
                 detail_noise=True, lightning=True, reflection=4, relief_steps=0),
    'HIGH': dict(core='ray', render_scale=0.5, features=('SCATTER', 'WEATHER', 'VCLOUDS', 'SKYREFLECT') + LIGHTING
                 + TERRAIN + ('RELIEF_SHADOW',),
                 atmo_samples=16,
                 cloud_steps=40, cloud_steps_base=12, cloud_steps_min=8, cloud_slices=14, light_steps=6, ms_octaves=3,
                 detail_noise=True, lightning=True, reflection=4, relief_steps=8),
    # experimental: HIGH clouds with more steps in a 0.75-resolution radiance pass
    'ULTRA': dict(core='ray', render_scale=0.75, features=('SCATTER', 'WEATHER', 'VCLOUDS', 'SKYREFLECT') + LIGHTING
                  + TERRAIN + ('RELIEF_SHADOW',),
                  atmo_samples=16,
                  cloud_steps=56, cloud_steps_base=16, cloud_steps_min=12, cloud_slices=18, light_steps=6, ms_octaves=3,
                  detail_noise=True, lightning=True, reflection=4, relief_steps=8),
}
DEFAULT_QUALITY = 'LEGACY'        # becomes 'MED' once the 2.0 look ships (phase 2+)
DEFAULT_FOV = 35.0                # degrees, for perspective=True
ORTHO_DISTANCE = 50.0             # Earth radii; where orthographic rays start
ATMO_SCALE = 1.5                  # on-screen atmosphere thickness x real (plan: atmo_scale; 1.5 picked on device)
LIMB_GAIN = 1.8                   # air light x (artistic: brighter limb at the satellite-image exposure)
HAZE = 0.10                       # fraction of air light kept looking straight down (satellite-image look)
SUN_IRRADIANCE = 16.0             # HDR units of the sun's irradiance
SCATTER_EXPOSURE = 0.3            # default exposure when the physical atmosphere is on (albedo ~ display)
MS_GAIN = 0.10                    # multiple-scattering ambient (plan suggested 0.25; 0.10 tuned in phase 2)
LAND_GAIN = 1.0                   # land albedo boost over the 1.x land colours
CLOUD_SCALE = 4.0                 # cloud shell (real 1-12 km) drawn this many times thicker
CLOUD_TAU = 10.0                  # optical depth through the whole shell at density 1
CLOUD_DRIFT = math.radians(6.0) / 24.0   # weather drifts east, rad per cloud hour
CLOUD_SEED = 3                    # weather map seed


_TEXTURES = {}


class WeatherFilm(object):
    """Hourly weather frames precomputed by orbitgfx.weather (tools/make_weather.py).

    Frames are Earth-fixed and start at 00:00 UTC, so the frame for a moment is
    chosen from where the sun is overhead (afternoon storms over land happen where
    it is afternoon in the render) and from cloud_hours for the day in the loop.
    """

    def __init__(self, path=None, keep=4):
        import json
        import os
        from .weather import FILM_DIR
        self.dir = path or FILM_DIR
        with open(os.path.join(self.dir, 'meta.json')) as fh:
            self.meta = json.load(fh)
        self.n = int(self.meta['frames'])
        self.hours = float(self.meta['frame_hours'])
        self.keep = keep
        self._cache = []                    # [(index, texture)], most recent last

    @staticmethod
    def available(path=None):
        import os
        from .weather import FILM_DIR
        return os.path.exists(os.path.join(path or FILM_DIR, 'meta.json'))

    def texture(self, k):
        import os
        import ui
        k %= self.n
        for i, (kk, tex) in enumerate(self._cache):
            if kk == k:
                self._cache.append(self._cache.pop(i))
                return tex
        with open(os.path.join(self.dir, 'w%03d.jpg' % k), 'rb') as fh:
            tex = scene.Texture(ui.Image.from_data(fh.read()))
        self._cache.append((k, tex))
        del self._cache[:-self.keep]
        return tex

    def position(self, cloud_hours, sun, theta):
        """(frame a, frame b, mix) for this moment."""
        c, s = math.cos(theta), math.sin(theta)
        x, y = c * sun[0] + s * sun[1], -s * sun[0] + c * sun[1]       # ECI -> Earth-fixed
        sub_lon = math.atan2(y, x)
        t_utc = ((math.pi - sub_lon) / (2 * math.pi)) % 1.0
        day = round(float(cloud_hours) / 24.0 - t_utc)
        f = (day + t_utc) * 24.0 / self.hours
        k0 = int(math.floor(f))
        return k0 % self.n, (k0 + 1) % self.n, f - k0


_FILMS = {}


def weather_film(path=None):
    key = path or ''
    if key not in _FILMS:
        _FILMS[key] = WeatherFilm(path)
    return _FILMS[key]


def _lut_tex(key, make):
    """One scene.Texture per baked table, shared by every GlobeView (memory, load time)."""
    tex = _TEXTURES.get(key)
    if tex is None:
        from .luts import lut_texture
        tex = _TEXTURES[key] = lut_texture(make())
    return tex


SURFACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'surface')


def _surface_tex(name):
    """Phase 5 surface textures (tools/make_surface.py), loaded once per process."""
    key = ('surface', name)
    tex = _TEXTURES.get(key)
    if tex is None:
        path = os.path.join(SURFACE_DIR, name)
        try:
            import ui
            img = ui.Image.named(path)           # straight from the file: no PNG re-encode
            if img is None:
                raise IOError(path)
            tex = scene.Texture(img)
        except Exception:
            from PIL import Image
            from .earth import pil_texture
            tex = pil_texture(Image.open(path).convert('RGB'))
        tex.filtering_mode = getattr(scene, 'FILTERING_LINEAR', 0)
        _TEXTURES[key] = tex
    return tex


def surface_size():
    """(width, height) of the baked surface data texture."""
    from PIL import Image
    with Image.open(os.path.join(SURFACE_DIR, 'earth.png')) as im:
        return im.size


def _vec(a):
    return tuple(float(x) for x in a)


PRIMER = True
_PRIMER_TEX = []


def _primer_texture():
    if not _PRIMER_TEX:
        from PIL import Image
        from .earth import pil_texture
        _PRIMER_TEX.append(pil_texture(Image.new('RGBA', (2, 2), (0, 0, 0, 255))))
    return _PRIMER_TEX[0]


def look_at(eye, look):
    """Rows right, up, toward the viewer (= -viewing direction) of a camera at
    `eye` looking at `look` (any length unit, ECI).

    Screen-up follows the local vertical at the eye. Looking straight along it
    (down at the Earth's centre, or straight up) leaves no horizon to level
    with, so up then comes from the coordinate axis least parallel to the view.
    """
    if look is None:
        raise ValueError('Camera.eye needs Camera.look')
    e = np.asarray(eye, float)
    f = e - np.asarray(look, float)
    n = float(np.linalg.norm(f))
    if not (math.isfinite(n) and n > 1e-12):
        raise ValueError('Camera.eye and Camera.look must be finite and distinct')
    f = f / n
    ne = float(np.linalg.norm(e))
    r = np.cross(e / ne, f) if ne > 1e-12 else np.zeros(3)
    if np.linalg.norm(r) < 1e-6:                    # radial view: no local horizon
        r = np.cross(np.eye(3)[int(np.argmin(np.abs(f)))], f)
    r = r / np.linalg.norm(r)
    return np.array((r, np.cross(f, r), f))


class Camera(object):
    """Orbit camera: azimuth / elevation about Earth's centre.

    matrix() rows are screen-right, screen-up, toward-viewer in ECI.
    Drag with inertia, pinch zoom, and an optional chase mode (`follow`).

    fov_deg:  None = orthographic (1.x behaviour); a number = pinhole camera with
              that vertical-or-horizontal field of view across the panel's short side.
    distance: pinhole eye distance from Earth's centre in Earth radii. None picks
              the distance at which the Earth looks as large as at this zoom in
              orthographic, so zoom keeps working.
    """

    def __init__(self, az=0.0, el=0.5, zoom=1.0, zoom_range=(0.7, 5.0),
                 drag_gain=(0.5, 0.4), friction=3.0, chase_offset=(-0.35, 0.30),
                 fov_deg=None, distance=None):
        self.az, self.el, self.zoom = az, el, zoom
        self.eye = None             # explicit eye (Earth radii, ECI): a free look-at camera
        self.look = None
        self.astro = False          # astronaut view: with follow, ride behind the spacecraft
        self.astro_fov = 70.0
        self.fov_deg = fov_deg
        self.distance = distance
        self.zoom_range = zoom_range
        self.drag_gain = drag_gain
        self.friction = friction
        self.chase_offset = chase_offset
        self.spin = [0.0, 0.0]
        self.follow = False
        self.held = False
        self._last_move = 0.0
        self._pinch0 = None
        self._astro_eye = False     # eye/look were placed by set_astronaut, not by the app

    def matrix(self):
        if self.eye is not None:
            return look_at(self.eye, self.look)
        ca, sa = math.cos(self.az), math.sin(self.az)
        ce, se = math.cos(self.el), math.sin(self.el)
        return np.array(((-sa, ca, 0.0), (-se * ca, -se * sa, ce), (ce * ca, ce * sa, se)))

    # gestures ---------------------------------------------------------
    def grab(self, now):
        self.held = True
        self.spin = [0.0, 0.0]
        self._last_move = now

    def drag(self, dx, dy, now):
        """dx, dy in points. Breaks out of chase mode."""
        self.follow = False
        if self.eye is not None:
            # leaving the astronaut view: continue from where it was looking
            e = np.asarray(self.eye, float)
            self.az, self.el = math.atan2(e[1], e[0]), math.asin(e[2] / np.linalg.norm(e))
            self.astro = False
            self.leave_astronaut()
        k = 1.0 / self.zoom
        daz = -math.radians(dx * self.drag_gain[0] * k)
        de = -math.radians(dy * self.drag_gain[1] * k)
        self.az += daz
        self.el = max(-1.45, min(1.45, self.el + de))
        dtm = max(1.0 / 120, now - self._last_move)
        self.spin = [0.6 * self.spin[0] + 0.4 * daz / dtm,
                     0.6 * self.spin[1] + 0.4 * de / dtm]
        self._last_move = now

    def release(self, now, fling=True):
        self.held = False
        if not fling or now - self._last_move > 0.08:
            self.spin = [0.0, 0.0]

    def pinch_begin(self):
        self._pinch0 = self.zoom

    def pinch(self, ratio):
        z0 = self.zoom if self._pinch0 is None else self._pinch0
        self.zoom = max(self.zoom_range[0], min(self.zoom_range[1], z0 * ratio))

    def pinch_end(self):
        self._pinch0 = None

    # per frame --------------------------------------------------------
    def set_astronaut(self, sat_km, vel, back=0.06, above=0.035, ahead=0.45, down=0.10):
        """Eye just behind and above the spacecraft, looking ahead and down at the
        horizon (Earth radii); the limb curves across the view like from the ISS."""
        p = np.asarray(sat_km, float) / RE
        r = p / np.linalg.norm(p)
        v = np.asarray(vel, float)
        v = v - r * np.dot(v, r)
        v /= max(np.linalg.norm(v), 1e-9)
        self.eye = p + r * above - v * back
        self.look = p + v * ahead - r * down
        self.fov_deg = self.astro_fov
        self._astro_eye = True

    def leave_astronaut(self):
        self.eye = self.look = None
        self.fov_deg = None
        self._astro_eye = False

    def step(self, dt, target=None, vel=None):
        """Advance inertia, or ease toward `target` (ECI vector) in chase mode; in
        astronaut mode (astro and follow, with vel) ride behind the spacecraft."""
        if self.follow and self.astro and target is not None and vel is not None:
            self.set_astronaut(target, vel)
            return
        if self._astro_eye:
            # only undo what astronaut mode placed: an app's own eye/look camera stays
            self.leave_astronaut()
        if self.follow and target is not None:
            p = np.asarray(target, float)
            taz = math.atan2(p[1], p[0]) + self.chase_offset[0] / self.zoom
            tel = max(-1.45, min(1.45, math.asin(p[2] / np.linalg.norm(p))
                                  + self.chase_offset[1] / self.zoom))
            k = 1.0 - math.exp(-dt * 2.5)
            self.az += ((taz - self.az + math.pi) % (2 * math.pi) - math.pi) * k
            self.el += (tel - self.el) * k
        elif not self.held and (self.spin[0] or self.spin[1]):
            self.az += self.spin[0] * dt
            self.el = max(-1.45, min(1.45, self.el + self.spin[1] * dt))
            decay = math.exp(-dt * self.friction)
            self.spin = [s * decay if abs(s * decay) > 1e-3 else 0.0 for s in self.spin]


class GlobeView(object):
    """Shader-rendered globe filling a rectangular panel.

    texture:   scene.Texture of an earth_image() (R land, G coast, B lights).
    features:  subset of shaders.FEATURES (default all). For LOW/MED/HIGH the
               default also adds the preset's physics ('SCATTER'); an explicit
               list is used as given (shaders.RAY_FEATURES lists every flag).
    palette:   overrides for shaders.PALETTE.
    fill:      fraction of min(panel w, h) used by the reference radius at zoom 1.
    ref_radius_km: radius that should fit the panel (e.g. RE + orbit altitude).
    quality:   'LEGACY' (the 1.x shader), 'LOW', 'MED' or 'HIGH' (see QUALITY).
    perspective: False, True (DEFAULT_FOV) or a field of view in degrees; used
               when camera.fov_deg is None.
    render_scale: override the preset's radiance-pass resolution (1.0 or 0.5).
    exposure:  tone-mapping exposure (SCATTER always tone-maps; TONEMAP adds it to
               the 1.x colours).
    atmo_scale: on-screen atmosphere thickness relative to the real one.
    cloud_scale: the cloud shell (real 1-12 km) is drawn this many times thicker.
    weather:   'film' plays the precomputed weather model (orbitgfx/data/weather),
               'baked' uses the static procedural map, 'auto' = film when present.
    haze:      0..1, air light kept over the disc looking straight down (1 = fully
               physical, hazy; low = Rayleigh-corrected like true-colour satellite
               images). The limb always gets the full atmosphere.
    defines:   extra or overriding shader #defines, e.g. {'CLOUD_GAIN': '1.5'} (tuning).
    debug:     extra shader defines for tests, e.g. ('NANCHECK',).
    """

    def __init__(self, parent, texture, features=None, palette=None, clouds=0.55,
                 fill=0.44, ref_radius_km=RE + 500.0, z_position=-1, inset=1.0,
                 quality=None, perspective=False, render_scale=None, exposure=None,
                 atmo_scale=ATMO_SCALE, haze=HAZE, cloud_scale=CLOUD_SCALE, weather='auto', defines=None, debug=()):
        self.quality = (quality or DEFAULT_QUALITY).upper()
        if self.quality not in QUALITY:
            raise ValueError('quality must be one of %s' % ', '.join(QUALITY))
        self.preset = dict(QUALITY[self.quality])
        self.ray = self.preset['core'] == 'ray'
        scale = self.preset['render_scale'] if render_scale is None else float(render_scale)
        if not self.ray or not hasattr(scene, 'EffectNode'):
            scale = 1.0
        self.render_scale = scale
        self.perspective = perspective
        self.features = None
        self.clouds = clouds
        self.fill = fill
        self.ref_radius_km = ref_radius_km
        self.inset = inset
        self.rect = (0.0, 0.0, 1.0, 1.0)
        self.effect = self.rad_node = self.rad_shader = None
        self.atmo_scale = float(atmo_scale)
        self.haze = float(haze)
        self.cloud_scale = float(cloud_scale)
        self.weather_tex = self.noise_tex = None
        self.trans_tex = None
        self._air_r2 = None             # squared top of the air, until the ground-camera warning is armed
        if self.ray:
            if features is None:            # preset look: all 1.x layers + the preset's physics
                feats = tuple(FEATURES) + tuple(self.preset.get('features', ()))
            else:                           # explicit list: exactly these flags
                feats = tuple(features)
            defs = {'ATMO_N': int(self.preset.get('atmo_samples', 10)), 'LAND_GAIN': repr(float(LAND_GAIN)),
                    'REFL_N': int(self.preset.get('reflection', 4) or 4),
                    'LIMB_GAIN': repr(float(LIMB_GAIN))}
            pr = self.preset
            if 'VCLOUDS' in feats:
                med = QUALITY['MED']               # explicit VCLOUDS on a preset without clouds
                g = lambda k: pr.get(k) or med[k]
                defs.update(CLOUD_N=int(g('cloud_steps')), CLOUD_NB=repr(float(g('cloud_steps_base'))),
                            CLOUD_NMIN=repr(float(g('cloud_steps_min'))), LIGHT_N=int(g('light_steps')),
                            MS_OCT=int(g('ms_octaves')), CLOUD_NS=int(g('cloud_slices')))
                for flag, key in (('DETAIL', 'detail_noise'), ('LIGHTNING', 'lightning')):
                    if pr.get(key):
                        defs[flag] = 1
            self.film = None
            if 'WEATHER' in feats or 'VCLOUDS' in feats:
                if weather == 'film' or (weather == 'auto' and WeatherFilm.available()):
                    self.film = weather_film()
                    defs['WEATHER2'] = 1
            self._film_k = None
            if set(feats) & {'SURFACE', 'RELIEF', 'RELIEF_SHADOW'}:
                sw, sh_ = surface_size()
                defs.update(TEX_W='%.1f' % sw, TEX_H='%.1f' % sh_)
            defs.update(defines or {})
            self.features = feats
        self.exposure = exposure if exposure is not None else (
            SCATTER_EXPOSURE if self.ray and 'SCATTER' in self.features else 1.0)
        if not self.ray:
            src = build_globe_shader(features, palette)
        elif scale >= 1.0:
            src = build_ray_shader(feats, palette, 'single', debug, defs)
        else:
            # radiance at reduced resolution: EffectNode scaled up, child sprite small
            self.effect = scene.EffectNode(parent=parent)
            self.effect.z_position = z_position - 0.01
            self.primer = None
            self.set_primer(PRIMER)
            self.rad_node = scene.SpriteNode(texture, parent=self.effect)
            self.rad_shader = scene.Shader(build_ray_shader(feats, palette, 'radiance', debug, defs))
            self.rad_node.shader = self.rad_shader
            src = build_ray_shader(feats, palette, 'overlay', debug, defs)
        self.node = scene.SpriteNode(texture, parent=parent)
        self.node.z_position = z_position
        self.shader = scene.Shader(src)
        self.node.shader = self.shader
        self.shaders = [sh for sh in (self.rad_shader, self.shader) if sh is not None]
        if self.ray and 'SCATTER' in self.features:
            from .luts import planet, transmittance_lut
            self.trans_tex = _lut_tex(('trans', self.atmo_scale), lambda: transmittance_lut(self.atmo_scale))
            rg, rt = planet(self.atmo_scale, RE)
            self._set('u_trans', self.trans_tex)
            self._set('u_atm', (rt / rg, rg, SUN_IRRADIANCE, MS_GAIN))
            # a camera inside this radius needs the ground-camera shader code (GROUNDCAM)
            self._air_r2 = None if 'GROUNDCAM' in defs else (rt / rg) ** 2
            self._set('u_haze', self.haze)
        if self.ray and set(self.features) & {'SURFACE', 'RELIEF', 'RELIEF_SHADOW'}:
            # the matching data texture (land, coast, lights) replaces the app's 1.x one
            self.node.texture = _surface_tex('earth.png')
            if self.rad_node is not None:
                self.rad_node.texture = self.node.texture
            rad = self.rad_shader or self.shader      # the overlay pass doesn't sample them
            rad.set_uniform('u_surf', _surface_tex('color.jpg'))
            rad.set_uniform('u_aux', _surface_tex('aux.png'))
        if self.ray and 'WEATHER' in self.features + (('WEATHER',) if 'VCLOUDS' in self.features else ()):
            from .luts import noise_lut, weather_lut
            self.weather_tex = _lut_tex(('weather', CLOUD_SEED), lambda: weather_lut(CLOUD_SEED))
            self._set('u_weather', self.weather_tex)
            h0, h1 = 1.0 * self.cloud_scale / RE, 12.0 * self.cloud_scale / RE
            self._cld = (h0, h1, CLOUD_TAU / (h1 - h0))
            if 'VCLOUDS' in self.features:
                self.noise_tex = _lut_tex(('noise',), noise_lut)
                self._set('u_noise', self.noise_tex)

    def set_primer(self, on):
        """A 1-point, near-invisible plain sprite drawn inside the EffectNode before the
        radiance sprite. On the device, a scene that also draws in draw() (immediate
        mode, e.g. scene.ellipse) can leave its own program and texture bound when the
        EffectNode renders; the EffectNode's only child then skips the rebind and is
        drawn as the last immediate-mode shape (a filled ellipse the size of the
        globe). The primer makes the offscreen pass switch program and texture every
        frame."""
        if self.effect is None:
            return
        if on and self.primer is None:
            self.primer = scene.SpriteNode(_primer_texture(), parent=self.effect)
            self.primer.size = (1.0, 1.0)
            self.primer.position = (0.0, 0.0)
            self.primer.alpha = 0.01
            self.primer.z_position = -1.0
        elif not on and self.primer is not None:
            self.primer.remove_from_parent()
            self.primer = None

    def remove(self):
        """Take this view's nodes out of the scene (e.g. to rebuild at another quality)."""
        for n in (self.node, self.effect):
            if n is not None:
                n.remove_from_parent()

    def _set(self, name, value):
        for sh in self.shaders:
            sh.set_uniform(name, value)

    set_uniform = _set

    def layout(self, rect):
        x, y, w, h = rect
        self.rect = rect
        i = self.inset
        W, H = w - 2 * i, h - 2 * i
        px = 1.0 / scene.get_screen_scale()
        self.node.size = (W, H)
        self.node.position = (x + w / 2, y + h / 2)
        self._set('u_size', (W, H))
        self._set('u_cloud', float(self.clouds))
        self.shader.set_uniform('u_pxs', px)
        if self.effect is not None:
            s = self.render_scale
            self.effect.position = (x + w / 2, y + h / 2)
            self.effect.x_scale = self.effect.y_scale = 1.0 / s
            self.rad_node.size = (W * s, H * s)
            self.rad_node.position = (0.0, 0.0)
            self.rad_node.z_position = 0.0
            self.rad_shader.set_uniform('u_pxs', px / s)

    @property
    def center(self):
        x, y, w, h = self.rect
        return x + w / 2, y + h / 2

    def earth_radius(self, zoom=1.0):
        """Earth radius on screen in points (the apparent radius, also in perspective)."""
        x, y, w, h = self.rect
        return min(w, h) * self.fill * RE / self.ref_radius_km * zoom

    def projection(self, camera):
        """(perspective?, focal length in points, eye distance in Earth radii)."""
        fov = camera.fov_deg
        if getattr(camera, 'eye', None) is not None and self.ray:
            x, y, w, h = self.rect
            f = 0.5 * min(w, h) / math.tan(math.radians(fov or DEFAULT_FOV) / 2.0)
            return True, f, float(np.linalg.norm(camera.eye))
        if fov is None and self.perspective:
            fov = DEFAULT_FOV if self.perspective is True else float(self.perspective)
        if fov is None or not self.ray:
            return False, 1.0, ORTHO_DISTANCE
        x, y, w, h = self.rect
        f = 0.5 * min(w, h) / math.tan(math.radians(fov) / 2.0)
        d = camera.distance
        if d is None:
            er = self.earth_radius(camera.zoom)
            d = math.sqrt((f / er) ** 2 + 1.0)
        return True, f, max(float(d), 1.0005)

    def project(self, p_km, camera):
        """ECI km -> (x, y, depth) in scene points; depth > 0 faces the viewer."""
        R = camera.matrix()
        p = np.asarray(p_km, float)
        q = p @ R.T
        cx, cy = self.center
        persp, f, d = self.projection(camera)
        if persp:
            v = p / RE - self.eye_position(camera, R, d)
            z = np.maximum(-(v @ R[2]), 1e-6)
            return cx + (v @ R[0]) * f / z, cy + (v @ R[1]) * f / z, q[..., 2]
        s = self.earth_radius(camera.zoom) / RE
        return cx + q[..., 0] * s, cy + q[..., 1] * s, q[..., 2]

    def eye_position(self, camera, R=None, d=None):
        """Camera eye in Earth radii (ECI)."""
        if getattr(camera, 'eye', None) is not None and self.ray:
            return np.asarray(camera.eye, float)
        if R is None:
            R = camera.matrix()
        if d is None:
            d = self.projection(camera)[2]
        return R[2] * d

    def update(self, camera, sun, theta, sat=None, vel=None, trail_rad=0.0,
               stations=(), cloud_hours=0.0):
        """Push one frame of state to the GPU.

        sun:      ECI unit vector.     theta: Earth rotation angle (rad).
        sat, vel: ECI km and km/s (None hides ORBIT/SATELLITE/FOOTPRINT data).
        trail_rad: arc length of the bright trail behind the spacecraft.
        stations: up to 6 (ecef_km, active) pairs; active draws the RF link.
        cloud_hours: drives cloud drift (e.g. mission time / 3600).
        """
        sh = self                      # GlobeView.set_uniform reaches every pass
        R = camera.matrix()
        if self.ray:
            persp, f, d = self.projection(camera)
            self._set('u_persp', 1.0 if persp else 0.0)
            self._set('u_focal', float(f))
            eye = self.eye_position(camera, R, d)
            self._set('u_campos', _vec(eye))
            if self._air_r2 is not None and persp and float(np.dot(eye, eye)) < self._air_r2:
                self._air_r2 = None
                warnings.warn("Camera.eye is inside the atmosphere but the view was built without "
                              "the 'GROUNDCAM' define: the sky will render wrongly. Pass "
                              "defines={'GROUNDCAM': '1'} (see examples/sunset.py).", stacklevel=2)
            self._set('u_exposure', float(self.exposure))
        sh.set_uniform('u_er', float(self.earth_radius(camera.zoom)))
        sh.set_uniform('u_right', _vec(R[0]))
        sh.set_uniform('u_up', _vec(R[1]))
        sh.set_uniform('u_fwd', _vec(R[2]))
        sh.set_uniform('u_sun', _vec(sun))
        sh.set_uniform('u_rot', (math.cos(theta), math.sin(theta)))
        if getattr(camera, 'eye', None) is not None:
            sat = vel = None           # riding the spacecraft: no ring or marker in view
        if sat is not None and vel is not None:
            p, v = np.asarray(sat, float), np.asarray(vel, float)
            e2 = np.cross(np.cross(p, v), p)
            e2 = e2 / np.linalg.norm(e2)
            sh.set_uniform('u_sat', _vec(p / RE))
            sh.set_uniform('u_e2', _vec(e2))
        else:
            sh.set_uniform('u_sat', (0.0, 0.0, 0.0))
            sh.set_uniform('u_e2', (0.0, 1.0, 0.0))
        sh.set_uniform('u_trail', float(trail_rad))
        sh.set_uniform('u_ctime', float(cloud_hours))
        if self.weather_tex is not None:
            film = getattr(self, 'film', None)
            if film is not None:
                k0, k1, mix = film.position(cloud_hours, sun, theta)
                if self._film_k != (k0, k1):
                    self._set('u_weather', film.texture(k0))
                    self._set('u_weather2', film.texture(k1))
                    self._film_k = (k0, k1)
                self._set('u_wmix', float(mix))
                a = theta                               # the film's winds move the clouds
            else:
                a = theta + float(cloud_hours) * CLOUD_DRIFT
            self._set('u_wrot', (math.cos(a), math.sin(a)))
            self._set('u_cld', self._cld + (float(cloud_hours),))
        sh.set_uniform('u_cam', (float(camera.az), float(camera.el)))
        stations = list(stations)[:STATION_SLOTS]
        for k in range(STATION_SLOTS):
            if k < len(stations):
                ecef, active = stations[k]
                s = ecef_to_eci(np.asarray(ecef, float) / RE, theta)
                sh.set_uniform('u_st%d' % k, _vec(s) + (1.0 if active else 0.0,))
            else:
                sh.set_uniform('u_st%d' % k, (0.0, 0.0, 0.0, 0.0))

    def sun_marker_position(self, sun, camera, margin=16.0):
        """Where to put a classic sun glyph on the panel edge, plus its alpha."""
        x, y, w, h = self.rect
        cx, cy = self.center
        sc = camera.matrix() @ np.asarray(sun, float)
        dlen = math.hypot(sc[0], sc[1])
        if dlen < 1e-3:
            return None
        rad = min(w, h) * 0.5 - margin
        px = min(x + w - 14, max(x + 14, cx + sc[0] / dlen * rad))
        py = min(y + h - 26, max(y + 14, cy + sc[1] / dlen * rad))
        alpha = .35 + .65 * max(0.0, float(sc[2]) + 0.3) / 1.3
        return px, py, alpha


class NightMapOverlay(object):
    """Exact day/night, twilight and city lights over an equirectangular map.

    Sits ABOVE the classic map layer (node tree draws last), so it also dims
    whatever you draw classically on the night side.
    """

    def __init__(self, parent, texture, z_position=-1):
        self.node = scene.SpriteNode(texture, parent=parent)
        self.node.z_position = z_position
        self.shader = scene.Shader(MAP_NIGHT_SHADER)
        self.node.shader = self.shader

    def layout(self, rect):
        x, y, w, h = rect
        self.node.size = (w, h)
        self.node.position = (x + w / 2, y + h / 2)

    def update(self, sun_ecef):
        self.shader.set_uniform('u_sun', _vec(sun_ecef))
