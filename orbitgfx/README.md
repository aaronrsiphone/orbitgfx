# orbitgfx

GPU Earth and orbit rendering for Pythonista's `scene` module, plus a desktop
sandbox that runs the same code (real GLSL, real touch dispatch) and writes
screenshots.

```
orbitgfx/                     drop this one folder into site-packages
  __init__.py                 import orbitgfx  (works on device and desktop)
  README.md, LICENSE
  views.py                    GlobeView, NightMapOverlay, Camera
  shaders.py                  globe shader template (feature flags + palette), map shader, lint_glsl
  earth.py                    bake / cache the Earth data texture (land, coast, city lights)
  gestures.py                 taps, double taps, drags, pinch, by screen region
  draw2d.py                   classic-layer helpers: panels, glow dots, sun glyph, MapRect
  astro.py                    frames, GMST, Sun, eclipse, small circles, Hermite sampling
  safety.py                   finds attributes that collide with Scene internals
  ivp.py                      NumPy-only RK45 solve_ivp (SciPy-compatible subset)
  coastlines.py               Natural Earth 1:110m coastlines loader
  data/                       ne_110m_coastline.geojson (public domain, bundled)
  examples/                   mission_control.py (the full app), globe_demo.py (40 lines)
  sandbox/                    desktop only, never imported on device
    runner.py, render.py      Runner API and the screenshot CLI
    shim/                     emulated scene + ui (put on sys.path only on desktop)
  tests/                      test_mission.py + baseline renders
```

Tuning the look, the physics and the cost of everything: see [TUNING.md](TUNING.md).

## On the device

Copy the `orbitgfx/` folder into Pythonista's `site-packages` (or next to your
script). Only Pythonista built-ins are used: `scene`, `ui`, NumPy, PIL.

- **Nothing else to install.** The RK45 solver (`orbitgfx.ivp`) and the
  coastline data (`orbitgfx.coastlines`, with the GeoJSON bundled in `data/`)
  ship in the package, so the examples run as they are, offline, from
  anywhere.
- **Restart after editing.** Pythonista keeps site-packages modules loaded
  between runs, so restart the app (or `importlib.reload`) after changing
  orbitgfx.
- **Texture cache.** The baked Earth texture is cached in
  `orbitgfx/.orbitgfx_cache/`. Delete that folder to rebuild it.

```python
import scene, time
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

scene.run(Demo())
```

### GlobeView

`update(camera, sun, theta, sat=None, vel=None, trail_rad=0, stations=(), cloud_hours=0)`

- `sun`: ECI unit vector. `theta`: Earth rotation angle (GMST).
- `sat`, `vel`: ECI km and km/s. They drive the orbit ring (solved per pixel),
  the trail, the footprint, and the spacecraft marker.
- `stations`: up to 6 `(ecef_km, active)` pairs. An active station gets a
  pulsing ring and an RF link line.
- `project(p_km, camera)` returns screen points, for labels or touch hit-tests.
- `sun_marker_position(sun, camera)` gives a spot on the panel edge for a
  classic sun glyph.

### Quality presets (2.0)

```python
GlobeView(self, tex, quality='MED')              # 'LEGACY' (default for now), 'LOW', 'MED', 'HIGH', 'ULTRA'
GlobeView(self, tex, quality='LOW', perspective=True)    # pinhole camera, 35 degree FOV
Camera(fov_deg=50, distance=1.2)                 # or set the projection on the camera
```

- `LEGACY` is the 1.x shader, pixel for pixel.
- `LOW`, `MED` and `HIGH` use the 2.0 ray core: one ray per pixel, exact
  sphere and orbit-ring intersections, orthographic or perspective.
- `MED` and `HIGH` draw the globe as two sprites: the radiance layer (surface,
  atmosphere, clouds) at half resolution inside an `EffectNode`, and the line
  work (ring, stations, spacecraft, graticule, stars) at full resolution on
  top. On an iPhone 16 Pro Max this costs a quarter of the full-resolution
  fragment work.
- `LOW`, `MED` and `HIGH` turn on `SCATTER`, a physically based atmosphere:
  single Rayleigh + Mie scattering with ozone, a cheap multiple-scattering
  term, the Earth's shadow inside the air, aerial perspective over the
  surface, ground lit by sunlight that has crossed the atmosphere, and night
  airglow (green 557.7 nm line with the red 630 nm band above it, only where
  that air is in the Earth's shadow), all HDR with ACES tone mapping
  (`exposure=`, default 0.3, which puts surface albedo roughly on display
  values). Sample counts per pixel: 6 / 10 / 16. The sunlight table is baked
  once by `luts.py` (0.1 s) and cached.
- `haze=` (default 0.10): the disc is shown like true-colour satellite imagery,
  which removes most of the air light ("Rayleigh correction"). `haze` is the
  fraction kept looking straight down; it ramps to the full physical value at
  the limb. `haze=1.0` is the fully physical, hazy view.
- `atmo_scale=` (default 1.5) draws the atmosphere 1.5x thicker than real so
  it reads at phone scale; the optics stay in real km.
- City lights and the coastline stroke are drawn at full resolution even in
  `MED` / `HIGH`.
- The `TONEMAP` feature tone-maps the 1.x colours without `SCATTER`.
- Clouds (2.0 phase 3) come from a baked global weather map (`luts.py`,
  1024 x 512: coverage, cloud type, thickness) with latitude bands, cyclones
  with spiral arms, tropical cyclones with eyes, clear subtropical deserts and
  marine stratocumulus decks. The map turns with the Earth and drifts east
  (`cloud_hours`); `clouds=` (default 0.55) shifts the global coverage, 0 = none.
  - `LOW` draws it as a lit 2D layer with fbm edges and ground shadows.
  - `MED`, `HIGH` and `ULTRA` ray-march a cloud shell (real 1-12 km, drawn
    `cloud_scale=` 4x thicker) through a tiling 3D noise atlas: wind-streaked
    ragged edges, stratus sheets vs lumpy convection, light marched toward the
    sun, multiple-scattering octaves, powder, two-lobe phase, sunlight coloured
    by the atmosphere at the cloud's real altitude (red tops at sunset, the
    Earth's shadow across the deck), cloud shadows on the ground, and
    lightning in night-side thunderstorms. On the disc the march samples fixed
    heights in the cloud shell (10 / 14 / 18 slices), so the clouds look the
    same from any camera and in every frame; above the limb it marches evenly
    (up to 24 / 40 / 56 steps). Light steps 4 / 6 / 6. Ground shadows integrate
    the same cloud column along the sun ray.
  - Cloud kinds: stratus sheets and lumpy convection (type channel), cellular
    stratocumulus in the low decks (closed cells where solid, open cells where
    broken), fair-weather cumulus puffs where convective cover is broken, tall
    thunderstorm towers, and a thin high cirrus layer (weather map channel B:
    anvil outflow and jet-stream cirrus) drawn as wind-streaked fibres.
  - City lights and coastlines are dimmed by the same cloud column the camera
    looks through, so thick cloud hides them at night.
  - `ULTRA` is experimental: HIGH clouds in a 0.75-resolution radiance pass.
- Weather film (2.0 phase 3b, default when present: `weather='auto'`): the
  clouds come from a precomputed run of a physical weather model
  (`orbitgfx/weather.py`, frames in `orbitgfx/data/weather/`, 4-day seamless
  loop, hourly, 3.6 MB). The phone only plays it back: each frame of the globe
  blends the two hourly frames around the moment, picked from where the sun is
  overhead (so afternoon storms over land happen where it is afternoon) and
  from `cloud_hours`.
  - Dynamics: the two-layer quasi-geostrophic (Phillips) model on the sphere.
    Jet streams, Rossby waves, cyclones, anticyclones and fronts come from
    baroclinic instability of an equator-to-pole temperature gradient
    (stronger in the winter hemisphere); surface friction gives trade winds
    and westerlies.
  - Moisture: sea surface temperature (warm pools, cold upwelling, warm
    boundary currents), evaporation, vapour carried by the wind,
    Clausius-Clapeyron saturation from the model's own warm and cold sectors,
    vertical motion from its temperature budget, frontal and convective cloud,
    stratocumulus over cold water, cloud decay in sinking air, afternoon
    convection over land from the sun.
  - Tropical cyclones: genesis over water above 27.8 C, steering by the model
    winds plus beta drift, intensity toward an SST potential intensity, decay
    over land; spiral bands, central overcast and an eye.
  - `weather='baked'` uses the static procedural map instead. Regenerate the
    film (about 5 minutes on a laptop, other seasons with `--decl`):
    `python -m orbitgfx.tools.make_weather --days 4 --decl -9`.
- Ray-traced lighting (2.0 phase 4, every ray preset):
  - `PENUMBRA`: sunlight at the spacecraft goes through the Sun's real disc
    (0.27 degrees) setting behind the Earth, then through the limb atmosphere
    (Rayleigh, Mie and ozone optical depth by tangent height). Entering eclipse
    at 500 km the marker, its glow and the ring near it turn orange and red
    for about 14 s before going dark, instead of switching off.
  - `CRAFTLIGHT`: the spacecraft is drawn as a small lit sphere (sun side,
    terminator, earthshine from the Earth below), with the outline kept so it
    always reads as a marker.
  - `REFRACTION` (every ray preset): the air bends light. Sunlight near the
    ground arrives from a sun lifted by the standard astronomical refraction
    (Saemundsson: 0.48 deg for a sun on the true horizon, 0.1 deg at 10 deg),
    so the day and the afterglow last a couple of minutes longer. From the ground
    the sun's disc is drawn lifted and squashed into an oval (the lower edge is
    lifted more). From orbit the limb is a lens: a ray grazing at height h is
    bent by about 1.2 deg x exp(-h / 7.3 km), so at eclipse entry dim red
    sunlight lingers ~15 s past geometric shadow, spread out and dimmed by the
    vertical squeeze. Terrestrial refraction (the sea horizon) is not modelled.
  - `GGX`: the sun glint on water is a microfacet highlight (GGX slopes for a
    wind-roughened sea, Smith masking, Schlick Fresnel).
  - `SKYREFLECT` (`MED` and up): the ocean reflects the sky along the mirror
    ray (a short atmosphere march, `reflection` samples per preset), with
    Fresnel, so water brightens and turns sky-coloured toward the limb and at
    sunset.
  - Astronaut camera: `cam.astro = True` with `cam.follow = True` and
    `cam.step(dt, sat_km, vel)` rides just behind and above the spacecraft
    looking ahead at the horizon (70 degree FOV pinhole; ray presets only).
    `cam.set_astronaut(sat_km, vel, back=, above=, ahead=, down=)` places it
    directly; any drag leaves the mode. `Camera.eye` / `Camera.look` (Earth
    radii, ECI) are a general look-at camera. The ring and marker are hidden
    while riding the spacecraft. In `examples/mission_control.py` (ray
    qualities) double tap cycles FREE, TRACK, ASTRO.
- Surface (2.0 phase 5, every ray preset): real imagery instead of the 1.x
  flat colours. `SURFACE` uses NASA Blue Marble Next Generation (March 2004)
  for land colour, NASA Black Marble 2016 for city lights, and a coastline and
  land mask traced from the same imagery (the 1.x texture is still used by
  `LEGACY`). Blue Marble shows no sea ice, so March sea ice is drawn in: the
  Arctic basin, Hudson and Baffin Bays, the Canadian archipelago, the East
  Greenland current, Barents/Kara, Bering and Okhotsk Seas, the northern Baltic,
  and the Antarctic pack near its March minimum. Ice has no sun glint.
  - `RELIEF`: slopes from a land height map tilt the ground toward or away
    from the sun (exaggerated 3x, `defines={'RELIEF_GAIN': '3.0'}`), so
    mountain ranges read at low sun.
  - `RELIEF_SHADOW` (`HIGH`, `ULTRA`): a short march toward the sun across the
    height map casts mountain shadows near the terminator.
  - Data: `orbitgfx/data/surface/` (colour and lights at 4096 x 2048, about
    10 km per texel; height and sea ice at 2048 x 1024; 2.7 MB), baked by
    `python -m orbitgfx.tools.make_surface BLUE BLACK HEIGHT` from the
    full-resolution GeoTIFFs (Blue Marble 5400 x 2700, Black Marble 13500 x 6750) of
    NASA Blue Marble and Black Marble (public domain, NASA Earth Observatory)
    and the land height map from three-globe's example assets (MIT).
- Scenes that also draw in `draw()` (immediate mode, like
  `examples/mission_control.py`): on the device the half-resolution radiance
  pass could come out as the last immediate-mode shape, a gold ellipse the size
  of the globe. GlobeView puts a 1-point primer sprite inside its EffectNode so
  the pass rebinds its shader every frame (`views.PRIMER`, on by default;
  confirmed on the device: only the EffectNode presets MED / HIGH / ULTRA were affected).
  `examples/draw_mix_check.py` flips quality, immediate drawing and the primer
  to test it.
- Ground camera (`examples/sunset.py`): an eye/look camera inside the air
  works too. The cloud march then starts at the cloud base with steps that grow
  with distance, a finer noise octave frays nearby clouds, air light in front of
  a cloud is the share of the path up to it, the sun's disc is drawn (2x size,
  reddened and hidden by cloud) and stars wait for dusk. The example stands at
  seven places picked from the weather film for their sky at sunset
  (stratocumulus deck, open cells, trade cumulus, thunderstorms, frontal
  overcast, cirrus, clear desert) with 8x finer cloud noise than the orbit view.
- `defines=` overrides shader constants for tuning, e.g.
  `GlobeView(..., defines={'CLOUD_GAIN': '2.5', 'WARP_AMT': '0.03'})`.
- Measured frame times (iPhone 16 Pro Max, 410 x 552 pt panel @3x, 2.0.0a9).
  Every row below holds 60 fps (16.6 ms) unless marked. Headroom = how many
  copies of the globe still fit in a 60 fps frame (x1 = no spare GPU).

  | preset | overview | chase 2.6x | chase, perspective | astronaut |
  | --- | --- | --- | --- | --- |
  | LEGACY | 60 fps | 60 fps | - | - |
  | LOW | x4 | x2 | x2 | x3 |
  | MED | x3 | x2 | x2 | x3 |
  | HIGH | x3 | x2 | x1 | x3 |
  | ULTRA | x2 | x1, p95 29 ms | 18.8 ms, p95 33 ms (drops frames) | x1 |

  With phase 5 (2.0.0a10: Blue Marble, relief, mountain shadows on HIGH) the
  table is unchanged except MED chase/P x2 -> x1 and HIGH astro x3 -> x2.

- `examples/phase_check.py` times every preset on the device, measures the
  headroom (how many globes still fit in a 60 fps frame) and lets you flip
  between them.

### Feature flags and palette

```python
from orbitgfx import FEATURES, PALETTE
# STARS ATMOSPHERE ORBIT CLOUDS LIGHTS AURORA GRATICULE FOOTPRINT STATIONS SATELLITE BIOMES GLINT
GlobeView(self, tex, features=[f for f in FEATURES if f != 'GRATICULE'],
          palette={'CYAN': (1.0, 0.4, 0.2), 'OCEAN': (0.02, 0.18, 0.30)})
```

Disabled features are compiled out with `#ifdef`, so they cost nothing on the
GPU.

### Layering rule

Pythonista draws the classic `draw()` layer first and the node tree on top.
Shader views are nodes, so:

- Put panel backgrounds and markers under the globe in `draw()`. The globe is
  transparent outside the Earth.
- `NightMapOverlay` sits above your classic map, so it also darkens anything
  drawn there on the night side.
- Labels need to sit above shader views: keep views at `z_position=-1`, which
  is the default.

## Sandbox (desktop)

Run these from the folder that contains `orbitgfx/`, or anywhere if it is
installed on your path:

```
pip install numpy pillow moderngl skia-python
python -m orbitgfx.sandbox.render orbitgfx/examples/mission_control.py Mission -o shot.png \
    --unix 1792188000 --exec "s.met0 = s.met = 4500; s.speed_idx = 0"
python -m orbitgfx.tests.test_mission
python -m orbitgfx.tests.test_raycore           # 2.0: ray core vs LEGACY, NaN scan, presets
python -m orbitgfx.sandbox.previews -o sheet.png  # presets x sun geometries contact sheet
```

```python
import orbitgfx                            # on a desktop this installs the scene/ui shim
from orbitgfx.sandbox import Runner
with Runner(orbitgfx.EXAMPLES_DIR + '/globe_demo.py', 'Demo') as r:
    r.advance(1.0); r.pinch((195, 420), 80, 180); r.render('zoomed.png')
```

- **Shaders run on real GLSL.** Your shader source runs on an actual OpenGL
  context (headless EGL; works with Mesa llvmpipe, no GPU needed), with
  Pythonista's built-ins wired up (`v_tex_coord`, `u_texture`, `u_time`,
  `u_sprite_size`, `u_scale`). Compile errors are raised when `Shader()` is
  created, and `lint_glsl` warnings are printed.
- **Classic drawing.** Skia renders `draw()` with y-up at 3x, matching an
  iPhone.
- **Framework bookkeeping is emulated.** The Runner updates `t`, `dt` and
  `touches` every frame, and touches go through the same dispatch code
  Pythonista uses (`old_touch.layer`).
- **Strict mode.** Assigning `self.dt`, `self.t`, `self.touches`, `self.size`
  and similar raises `SandboxError`. Both device crashes from the first
  version are caught here.
- **Virtual clock.** `time.monotonic()` and `time.time()` are virtual, so tap
  timing, inertia and animation are reproducible.
- **Runner API.** `Runner(...).advance(s)`, `.tap(x, y)`, `.double_tap()`,
  `.drag(points)`, `.pinch(center, d0, d1)`, `.resize(w, h)`,
  `.render(path, u_time=None)`.
- **Same data and solver as the device.** The sandbox uses the bundled
  solver and coastlines, so desktop renders match what the phone shows.

Not emulated: Actions, ShapeNode, EffectNode, sound. Fonts use DejaVu Sans
Mono instead of Menlo, so text widths differ slightly.

## Rules that keep GPU code working on iOS

1. **No negative bases in `pow`.** `pow(x, y)` with `x < 0` is undefined in
   GLSL ES, and Apple GPUs return NaN. Square with `x * x` instead.
   `lint_glsl` flags this.
2. **Precision line first.** Start every shader with `precision highp float;`
   and never include `#version`.
3. **Premultiplied alpha.** Write premultiplied colour to `gl_FragColor`.
4. **No uniform arrays.** Use one uniform per slot (`u_st0` to `u_st5`).
5. **RGB-only data textures.** iOS premultiplies alpha when loading an image,
   which would corrupt any data stored in the RGB channels.
6. **Leave Scene's attributes alone.** Never assign Scene-owned attributes
   (`dt`, `t`, `touches`, `size`, `bounds`, `view`, ...). Run
   `safety.assert_clean(MyScene)` at the start of `setup()`.
