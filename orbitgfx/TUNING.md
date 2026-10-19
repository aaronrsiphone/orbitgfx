# orbitgfx 2.0 tuning guide

How to change the look, the physics and the cost of everything orbitgfx draws: the
quality presets, the camera, the atmosphere, the surface, the water, the clouds, the
offline weather simulation, eclipse lighting and refraction, colours, and the example
apps. Every knob below exists in 2.0.0a17; defaults are the values shipped.

Contents

1. How tuning works
2. Quality and performance
3. Feature flags
4. Camera and projection
5. Exposure and tone
6. Atmosphere
7. Night side: airglow, aurora, city lights
8. Surface: imagery, sea ice, relief
9. Water: glint and sky reflection
10. Clouds
11. The weather simulation (offline)
12. Sunlight, eclipses and refraction
13. Colours and line work
14. The example apps
15. Recipes
16. Troubleshooting

---

## 1. How tuning works

There are five places to turn a knob, from quickest to deepest:

| Level | Where | Scope | Takes effect |
| --- | --- | --- | --- |
| Constructor arguments | `GlobeView(..., haze=0.3)` | one globe | when the view is built |
| Per-frame arguments | `globe.update(camera, sun, theta, ..., cloud_hours=)` | one frame | immediately |
| Module constants | `orbitgfx.views.ATMO_SCALE = 1.2` before building | every globe built afterwards | when a view is built |
| Shader defines | `GlobeView(..., defines={'CLOUD_GAIN': '2.5'})` | one globe | compiled into its shader |
| Offline data | `tools/make_weather.py`, `tools/make_surface.py` | every globe | after re-baking files |

Shader defines are compile-time constants (`#ifndef NAME / #define NAME default` in
`shaders.py`). Pass them as strings; floats need a decimal point (`'2.0'`, not `'2'`).
Defines override whatever the preset or `GlobeView` would have set.

```python
from orbitgfx import GlobeView
globe = GlobeView(scene, tex, quality='HIGH', haze=0.2, clouds=0.7,
                  defines={'CLOUD_GAIN': '2.4', 'RELIEF_GAIN': '4.0'})
```

**Pythonista caches imported modules.** After editing any file inside
`site-packages/orbitgfx`, force-quit Pythonista (swipe it away) before running again,
or your change will not load. Changing module constants from your own script at
runtime (`orbitgfx.views.HAZE = 0.3`) does not need a restart.

**Measure after every change that touches cost.** Run `examples/phase_check.py`: it
times every preset in four views and reports headroom (how many globes fit in a 60 fps
frame). Section 2 has the iPhone 16 Pro Max numbers.

---

## 2. Quality and performance

### Presets

`GlobeView(quality=...)` picks a preset from `orbitgfx.views.QUALITY`.

| Preset | Core | Radiance resolution | What you get |
| --- | --- | --- | --- |
| `LEGACY` | 1.x shader | full | the original look, pixel for pixel (the default for now) |
| `LOW` | ray | full, one pass | physical atmosphere, 2D lit clouds from the weather film, imagery, relief, eclipse lighting, refraction |
| `MED` | ray | half | + volumetric clouds, sky reflection on water |
| `HIGH` | ray | half | + more atmosphere/cloud/light samples, mountain shadows |
| `ULTRA` | ray | 0.75 | HIGH with even more cloud steps (experimental) |

Measured on iPhone 16 Pro Max (410 x 552 pt panel, 2.0.0a10), headroom = globes per
60 fps frame:

| Preset | overview | chase 2.6x | chase, perspective | astronaut |
| --- | --- | --- | --- | --- |
| LOW | x4 | x2 | x2 | x3 |
| MED | x3 | x2 | x1 | x3 |
| HIGH | x3 | x2 | x1 | x2 |
| ULTRA | x2 | x1 (p95 29 ms) | 19 ms, drops frames | x1 |

The ground camera in `examples/sunset.py` is the most expensive view (every sky pixel
marches cloud): about 20-40 fps on ULTRA.

### Preset keys

Each preset is a dict; copy one and change keys to make your own:

```python
from orbitgfx import views
views.QUALITY['MINE'] = dict(views.QUALITY['HIGH'], render_scale=0.6, cloud_steps=48)
globe = views.GlobeView(scene, tex, quality='MINE')
```

| Key | Meaning | LOW / MED / HIGH / ULTRA |
| --- | --- | --- |
| `core` | `'legacy'` or `'ray'` | ray |
| `render_scale` | resolution of the radiance pass (surface, air, clouds). Line work (ring, coasts, lights, stars, spacecraft) is always full resolution | 1.0 / 0.5 / 0.5 / 0.75 |
| `features` | feature flags added to the 1.x layers (section 3) | |
| `atmo_samples` | steps of the atmosphere integral per pixel (`ATMO_N`) | 6 / 10 / 16 / 16 |
| `cloud_steps` | max cloud steps for rays that graze the shell above the limb, and for the ground camera (`CLOUD_N`) | - / 24 / 40 / 56 |
| `cloud_steps_base` | steps per shell thickness for grazing rays (`CLOUD_NB`) | - / 8 / 12 / 16 |
| `cloud_steps_min` | minimum steps for grazing rays (`CLOUD_NMIN`) | - / 6 / 8 / 12 |
| `cloud_slices` | fixed-height slices for rays that look down through the clouds (`CLOUD_NS`) | - / 10 / 14 / 18 |
| `light_steps` | steps of the march toward the sun from each cloud sample (`LIGHT_N`) | - / 4 / 6 / 6 |
| `ms_octaves` | multiple-scattering octaves in clouds, 2 or 3 (`MS_OCT`) | - / 2 / 3 / 3 |
| `detail_noise` | erode cloud edges with the fine noise (`DETAIL`) | off / on / on / on |
| `lightning` | flashes in night-side thunderstorms (`LIGHTNING`) | off / on / on / on |
| `reflection` | atmosphere steps for the sky reflected by the sea (`REFL_N`) | 0 / 4 / 4 / 4 |
| `relief_steps` | steps of the mountain-shadow march (informational; the shader uses `RS_N`, 8) | 0 / 0 / 8 / 8 |

`GlobeView(render_scale=...)` overrides the preset's resolution for one view. Values
below 1 use an `EffectNode` (a smaller offscreen pass scaled up); 1.0 renders in one
pass.

### Where the time goes

Rough cost order on the GPU, most expensive first: cloud steps x light steps (MED and
up), atmosphere samples, sky reflection over water, mountain shadows (8 texture reads on
land), relief (2 reads), everything else. To claw back frame time, try in this order:
`render_scale` (0.5 -> 0.4), `cloud_steps`/`cloud_slices`, `light_steps` 6 -> 4,
`atmo_samples` 16 -> 10, drop `SKYREFLECT` or `RELIEF_SHADOW` from `features`.

---

## 3. Feature flags

`GlobeView(features=...)` with an explicit tuple uses exactly those flags. Without it,
a ray preset uses every 1.x layer plus its own list. Requirements are checked when the
shader is built (`WEATHER` and `SURFACE` need `SCATTER`; `VCLOUDS` adds `WEATHER`;
`RELIEF` adds `SURFACE`; `RELIEF_SHADOW` adds `RELIEF`).

### 1.x layers (all presets)

| Flag | Draws |
| --- | --- |
| `STARS` | twinkling star field (fixed on the sky in perspective) |
| `ATMOSPHERE` | the 1.x limb glow (replaced by `SCATTER` in ray presets) |
| `ORBIT` | the orbit ring and the bright trail behind the spacecraft |
| `CLOUDS` | clouds (1.x: noise bands; ray: from the weather film) |
| `LIGHTS` | city lights on the night side |
| `AURORA` | auroral ovals around the magnetic poles |
| `GRATICULE` | latitude/longitude lines |
| `FOOTPRINT` | the spacecraft's ground-coverage circle |
| `STATIONS` | ground stations and the RF link |
| `SATELLITE` | the spacecraft marker |
| `BIOMES` | 1.x land/sea colour by latitude (ignored where `SURFACE` paints imagery) |
| `GLINT` | sun glint on water |

### 2.0 flags (ray core only)

| Flag | What it does | Presets | Cost |
| --- | --- | --- | --- |
| `TONEMAP` | ACES tone mapping of the 1.x colours (without `SCATTER`) | - | tiny |
| `SCATTER` | physical atmosphere: Rayleigh + Mie + ozone single scattering, multiple-scattering ambient, Earth's shadow in the air, aerial perspective, airglow | all ray | medium |
| `WEATHER` | clouds from the weather film (2D lit layer on LOW) | all ray | small |
| `VCLOUDS` | volumetric cloud march (3D noise, light march, ground shadows, cirrus) | MED+ | large |
| `SKYREFLECT` | the sky reflected by the sea (Fresnel weighted) | MED+ | medium over water |
| `PENUMBRA` | soft eclipse: the Sun's disc setting behind the limb, reddened by the air | all ray | tiny |
| `GGX` | microfacet sun glint (rough sea, masking, Fresnel) | all ray | tiny |
| `CRAFTLIGHT` | spacecraft drawn as a sun-lit sphere with earthshine | all ray | tiny |
| `REFRACTION` | atmospheric refraction (section 12) | all ray | small |
| `SURFACE` | NASA Blue Marble land, Black Marble lights, sea ice, coastline from imagery | all ray | small |
| `RELIEF` | terrain shading from the height map | all ray | small |
| `RELIEF_SHADOW` | mountain shadows near the terminator | HIGH, ULTRA | medium on land |

Debug: `GlobeView(debug=('NANCHECK',))` paints any NaN/inf pixel magenta.

Example: MED without sky reflection and with mountain shadows:

```python
f = tuple(views.FEATURES) + tuple(x for x in views.QUALITY['MED']['features'] if x != 'SKYREFLECT') \
    + ('RELIEF_SHADOW',)
globe = GlobeView(scene, tex, quality='MED', features=f)
```

---

## 4. Camera and projection

`Camera(az=0, el=0.5, zoom=1, zoom_range=(0.7, 5), drag_gain=(0.5, 0.4), friction=3,
chase_offset=(-0.35, 0.30), fov_deg=None, distance=None)`

| Knob | Effect |
| --- | --- |
| `az`, `el` | where the camera looks from (radians, Earth-centred) |
| `zoom` | 1 = the default disc size (`GlobeView(fill=0.44)` of the panel), clamped to `zoom_range` |
| `drag_gain`, `friction` | how far a drag turns the globe and how fast a fling slows down |
| `follow`, `chase_offset` | chase mode: ease toward the spacecraft, offset by (az, el) / zoom |
| `fov_deg`, `distance` | perspective: field of view (degrees) or eye distance (Earth radii). With neither, a perspective view keeps the same disc size as orthographic at that zoom |
| `astro`, `astro_fov` | astronaut view (with `follow`): ride behind the spacecraft, 70 degree FOV |
| `eye`, `look` | free look-at camera, Earth radii, ECI. Works inside the atmosphere (ground cameras) |

`Camera.set_astronaut(sat_km, vel, back=0.06, above=0.035, ahead=0.45, down=0.10)`
places the astronaut camera directly; the four distances are in Earth radii (0.06 =
380 km behind, 0.035 = 220 km above, looking at a point 2900 km ahead and 640 km
down). Raise `down` to look more at the ground, lower it for more horizon and sky.

`GlobeView(perspective=True)` turns on the pinhole camera (35 degree FOV,
`DEFAULT_FOV`); `perspective=50` sets the FOV. Ortho rays start `ORTHO_DISTANCE` (50
Earth radii) out.

Ground camera (as in `sunset.py`):

```python
cam = Camera(fov_deg=75)
up = unit_vector(lat, lon)              # theta = 0 makes ECI = Earth-fixed
cam.eye = up * (1 + 0.002 / 6371 * 1000)  # 2 m above the sea
cam.look = cam.eye + forward_direction
```

A camera inside the atmosphere needs the ground-camera shader code, which is compiled
only with the `GROUNDCAM` define (it keeps the orbit shaders small and fast):
`GlobeView(..., defines={'GROUNDCAM': '1'})`; `sunset.GROUND` already has it. Without it
such a camera draws the sky wrongly and `GlobeView.update` warns once.

---

## 5. Exposure and tone

The ray core renders in HDR (sunlight = `SUN_IRRADIANCE`, 16 units) and maps to the
screen with ACES.

| Knob | Default | Effect |
| --- | --- | --- |
| `GlobeView(exposure=)` | 0.3 with `SCATTER` | overall brightness. 0.3 puts surface albedo on display values (satellite-image look). Ground views want more: `sunset.py` uses 0.8 |
| `views.SUN_IRRADIANCE` | 16 | sun strength relative to sky and lights; changes the balance between day side and night-side lights/airglow |
| `views.LIMB_GAIN` | 1.8 | brightens the air light (blue limb, sky) at the satellite exposure. Artistic; use 1.0 for physical (ground views do) |

Raising exposure brightens night-side lights and airglow too; lower `LIMB_GAIN` to keep
the limb from blowing out.

---

## 6. Atmosphere

The atmosphere is Bruneton/Hillaire-style single scattering with a baked transmittance
table (`luts.py`), real scale heights (Rayleigh 8 km, aerosol 1.2 km, ozone layer at
25 km), drawn thicker than real so it reads at phone scale.

Shader units: positions are in drawn Earth radii; one unit is `u_atm.y` =
6378.137 / `atmo_scale` km. Every height in km comes from `altitudeKm(r)` (refraction,
the sun's disc, the airglow), so a known physical altitude means the same air whatever
`atmo_scale` is. The orbit, spacecraft and their eclipse are true scale (`RE_KM`).

| Knob | Default | Effect |
| --- | --- | --- |
| `GlobeView(atmo_scale=)` / `views.ATMO_SCALE` | 1.5 | thickness of the drawn atmosphere x real. 1.0 is true scale (thin, almost invisible from orbit). Drawn as a real 100 km atmosphere on an `atmo_scale` x smaller planet: vertical extinction stays in real km, but grazing paths see the tighter curvature and cross less air (tangent transmittance at the ground, R/G/B: 0.106/0.0095/0.00005 at 1.0, 0.160/0.022/0.0003 at 1.5), so the limb is brighter and less red than real. A visual exaggeration; 1.0 is the physical geometry. Ground views use 1.0 |
| `GlobeView(haze=)` / `views.HAZE` | 0.10 | fraction of the air light kept when looking straight down. Satellite imagery removes most of it ("Rayleigh correction"); 1.0 is the fully physical, hazy blue view. It always ramps back to full toward the limb |
| `views.MS_GAIN` | 0.10 | multiple-scattering ambient: how much light the air gets from the rest of the sky. 0.25 washes the disc out; 0 makes twilight too dark |
| `atmo_samples` (preset) / `ATMO_N` define | 6-16 | accuracy of the integral; low values band at the terminator |
| `LIMB_GAIN` | see section 5 | |

The transmittance table is cached per `atmo_scale` value; a new value bakes in ~0.1 s.

Deeper (edit `shaders.py`): Rayleigh/aerosol/ozone coefficients `BETA_R`, `BETA_ME`,
`BETA_MS`, `BETA_O`; the Mie phase asymmetry (`phaseM`, g = 0.76); the density profile
in `density()`.

---

## 7. Night side: airglow, aurora, city lights

All three are fixed constants in `shaders.py` (search for the comment):

| Effect | Where | Knobs |
| --- | --- | --- |
| Green airglow (557.7 nm) | `airglow:` in `RAY_MAIN` | height 95 km, width 6 km, brightness 0.025; only where that air is in Earth's shadow |
| Red airglow (630 nm) | same block | height 165 km, width 60 km, brightness 0.007 |
| Aurora | `#ifdef AURORA` | oval at 0.36 rad from the magnetic pole `mp`, curtain noise, colours green -> violet, strength 0.6 |
| City lights | `#ifdef LIGHTS` | colour `LIGHT` in the palette, gain 1.35; drawn at full resolution in the overlay, dimmed by the cloud column in front of them |

City lights come from Black Marble when `SURFACE` is on. To brighten small towns,
re-bake with a lower exponent in `tools/make_surface.py`
(`np.clip(li / ref, 0, 1) ** 0.7`; 0.5 lifts more) or raise the 1.35 gain.

---

## 8. Surface: imagery, sea ice, relief

### Imagery

`SURFACE` draws `orbitgfx/data/surface/`:

| File | Size | Contents |
| --- | --- | --- |
| `color.jpg` | 4096 x 2048 | Blue Marble Next Generation (March 2004) land colour, sea ice painted on the water |
| `earth.png` | 4096 x 2048 | R land mask, G coastline stroke, B Black Marble 2016 lights (traced from the same imagery) |
| `aux.png` | 2048 x 1024 | R height, sqrt-encoded (h = 9 km x R^2, 0 = water), G sea-ice concentration |

Re-bake (about 25 s on a laptop):

```
python -m orbitgfx.tools.make_surface BLUE_MARBLE.tif BLACK_MARBLE.tif earth-topology.png \
       [--width 4096] [--aux-width 2048]
```

- Another month of Blue Marble (`world.topo.2004MM...`) changes snow cover and
  vegetation; keep sea ice consistent with it (below).
- `--width 2048` halves texture memory (2 x 32 MB -> 2 x 8 MB) and package size.
- The 1.x texture is still used by `LEGACY`.

| Knob | Default | Effect |
| --- | --- | --- |
| `SURF_GAIN` define | 0.8 | land albedo from the imagery colour (linearised with gamma 2.2). Higher: brighter land |
| `LAND_GAIN` / `views.LAND_GAIN` | 1.0 | land albedo for the procedural (non-`SURFACE`) colours |

### Sea ice

Blue Marble has no sea ice, so `make_surface.py` draws March extents from tables at the
top of the file:

- `N_EDGE`: (longitude, ice-edge latitude) pairs around the Arctic, interpolated.
- `S_EDGE`: the same around Antarctica (March is near its minimum).
- `BOXES`: (lat0, lat1, lon0, lon1) extra regions (Baltic gulfs, Gulf of St Lawrence,
  northern Caspian, Bohai).

For a September Arctic minimum, raise the `N_EDGE` latitudes to about 72-80 and remove
Hudson Bay, the Baltic and Okhotsk. The ragged edge comes from `jitter` (degrees of
wobble) and the leads (cracks) from `n4`; the ice colour from `ice_rgb`.

### Relief

`RELIEF` tilts the lighting by the slope of the height map; `RELIEF_SHADOW` (HIGH,
ULTRA) marches toward the sun across it.

| Knob | Default | Effect |
| --- | --- | --- |
| `RELIEF_GAIN` define | 3.0 | vertical exaggeration of both slopes and shadows. 1 is true scale (barely visible from orbit); 6 makes mountains look like crumpled paper |
| `RS_N` define | 8 | shadow march steps (4 km to 256 km, quadratic spacing) |
| shadow softness | `smoothstep(-1.5, 0.5, d)` in the shadow loop | km of penumbra |

The shadow march runs only on land with the sun below about 37 degrees.

---

## 9. Water: glint and sky reflection

| Knob | Default | Effect |
| --- | --- | --- |
| `GLINT_ALPHA` define | 0.15 | GGX roughness of the sea. 0.05 = glassy calm (small, intense sun spot); 0.3 = rough, wind-blown (large, faint glitter) |
| `REFL_N` define / `reflection` key | 4 | steps of the reflected-sky integral. 0 in a preset disables it |
| Fresnel | 0.02 at normal incidence (water) | in the `GGX` and `SKYREFLECT` blocks |

Water is wherever the land mask is 0; sea ice and cloud switch the glint off.

---

## 10. Clouds

### Coverage and timing

| Knob | Default | Effect |
| --- | --- | --- |
| `GlobeView(clouds=)` | 0.55 | 0.55 shows the weather film as recorded. Each 0.1 above or below shifts the coverage by 0.09; 0 removes all clouds (cirrus too) |
| `update(..., cloud_hours=)` | | the weather clock. The film is 96 hourly frames (4 days, looping); the frame is chosen from where the sun is overhead (UTC) plus `cloud_hours`, and neighbouring hours are blended |
| `GlobeView(weather=)` | `'auto'` | `'film'` (the precomputed weather), `'baked'` (a static procedural map from `luts.py`), `'auto'` = film when present |
| `views.CLOUD_DRIFT` | 6 deg/day | eastward drift of the baked map (not the film) |
| `views.CLOUD_SEED` | 3 | seed of the baked map |

### Volume and shape

The cloud shell spans 1-12 km. Coverage, type and cirrus come from the weather film;
the 3D shape from a tiling noise atlas (`luts.noise_lut`).

| Knob | Default | Effect |
| --- | --- | --- |
| `GlobeView(cloud_scale=)` / `views.CLOUD_SCALE` | 4 | shell drawn this many times thicker than real (4 = 4-48 km), so tops and shadows read from orbit. 1.0 for ground views |
| `views.CLOUD_TAU` | 10 | optical depth through the whole shell at density 1. Higher: whiter, more opaque cloud with sharper edges; lower: thin, grey, see-through |
| `NOISE_F` | 80 | base shape frequency, cells per Earth radius (80 = ~80 km cells) |
| `NOISE_FD` | 650 | detail (edge erosion) frequency (~10 km) |
| `NOISE_V` | 9 | how fast the shape changes with height (towers vs sheets) |
| `NOISE_W`, `WARP_AMT` | 140, 0.022 | domain warp of the weather lookup: ragged, wind-streaked edges. 0 gives the coarse weather pixels' straight edges |
| `BASE_AMT` | 0.6 | how much the base noise breaks up the coverage (0 = smooth blobs) |
| `ERODE` | 0.15 | how much detail noise eats cloud edges (cores stay solid) |
| `DETAIL_BILLOW` | 1.0 | 1 = wispy bases and billowy tops, 0 = the same texture throughout |
| `CELLS` | 1.0 | strength of the stratocumulus cell pattern (closed cells in solid decks, open cells where broken); 0 turns it off |

Cloud types come from the film's type channel: below about 0.2, stratiform decks
(cells); 0.5-0.8 with broken cover, fair-weather cumulus puffs; above 0.7, tall
convection and (with `lightning`) flashes at night.

### Lighting

| Knob | Default | Effect |
| --- | --- | --- |
| `CLOUD_GAIN` | 2.0 | brightness of sunlit cloud. 2.5 for brilliant tops; 1.5 for greyer |
| `light_steps` / `LIGHT_N` | 4-6 | accuracy of self-shadowing (dark bases, lit tops) |
| `ms_octaves` / `MS_OCT` | 2-3 | multiple-scattering octaves: softer, brighter cloud interiors |
| phase function | `hg(c, 0.80)` / `hg(c, -0.30)` in `cloudMarch` | forward silver lining vs. backscatter |
| powder | `kpow = 0.30 * ...` | darkened edges when looking away from the sun |
| ambient | `amb` in `cloudMarch` | sky light on shaded sides (0.08 blue) |

Clouds are lit by sunlight coloured by the air at their real 1-12 km altitude, so tops
turn red at sunset regardless of `cloud_scale`.

### Cirrus

| Knob | Default | Effect |
| --- | --- | --- |
| `CI_OPAC` | 0.7 | cirrus opacity |
| `CI_F` | 55 | fibre frequency (Earth radii) |
| `CI_STRETCH` | 3.5 | how much fibres are stretched along latitude circles |
| `CI_WF`, `CI_WARP` | 30, 18 | warp frequency and strength (curving, fraying strands) |
| `CI_LO` | 0.35 | fibre threshold: higher = sparser strands |

### Ground views

From under the clouds, features of 10-80 km look like smooth haze. `sunset.py` passes
noise 8x finer:

```python
GROUND = {'LIMB_GAIN': '1.0', 'NOISE_F': '640.0', 'NOISE_FD': '5200.0', 'NOISE_V': '72.0',
          'CI_F': '440.0', 'CI_WF': '240.0', 'BASE_AMT': '0.85'}
```

With the camera inside or under the shell the march uses `cloud_steps` steps spaced
quadratically (dense nearby) and adds a finer noise octave for clouds closer than about
a pixel per 1.5 km.

---

## 11. The weather simulation (offline)

`orbitgfx/weather.py` is a small general-circulation model that is run once, offline;
the phone only plays back its frames. Only its dynamical core is a governing-equation
model: a two-layer quasi-geostrophic midlatitude circulation. Moisture and clouds are
parameterised proxies calibrated to look right, tropical cyclones are scripted artwork,
and the film is a 4-day seamless loop of one season, not observations. The tiers and
what each is checked against are in VALIDATION.md.

### Making a film

```
python -m orbitgfx.tools.make_weather [--days 4] [--spin 30] [--grid 256] [--decl -9] [--out DIR]
```

| Option | Default | Effect |
| --- | --- | --- |
| `--days` | 4 | loop length; the first 12 hours are cross-faded with the hours after the end so it loops seamlessly. Each day adds 24 frames (~0.85 MB) |
| `--spin` | 30 | spin-up days before recording (the jets and storms need ~20 to develop) |
| `--grid` | 256 | longitude points (latitude = half). 256 = 1.4 degrees; 512 doubles resolution and takes ~8x longer |
| `--decl` | -9 | sun declination = season. -9: late February / mid-October; +23: northern summer; -23: northern winter; 0: equinox |

About 5 minutes on a laptop at the defaults (numpy only; it runs in Pythonista too, more
slowly). Frames are written to `data/weather/` and picked up the next time a globe is
built. If you change the season, `sunset.py`'s `DECL` and site frames should follow.

### Dynamics (`Model(...)`)

Pass these through `weather.Climate(..., **model_kw)` (edit `record()` in `weather.py`):

| Parameter | Default | Effect |
| --- | --- | --- |
| `ld_km` | 1000 | deformation radius: the size of storms. Smaller = more, smaller cyclones |
| `shear` | 1.6 rad/day | equator-to-pole temperature contrast (jet strength). Higher = stronger jets and more storms; above ~2.2 the model gets violent |
| `t_rad` | 20 days | radiative relaxation toward that contrast |
| `t_fric` | 4 days | surface friction toward the trade winds / westerlies / polar easterlies |
| `t_drag1` | 5 days | upper-layer drag (keeps jets near 60 m/s) |

The time step (30 minutes) and the semi-Lagrangian bicubic advection are set for
stability; 2-hour steps blow up.

### Moisture and clouds (module constants in `weather.py`)

| Constant | Default | Effect |
| --- | --- | --- |
| `KW` | 30 /day | large-scale condensation per unit ascent (fronts, comma clouds) |
| `KC` | 30 /day | convective condensation (ITCZ, afternoon storms) |
| `LS_DECAY` | 1.5 /day | frontal cloud lifetime |
| `CV_DECAY` | 1.2 /day | convective cloud and anvil lifetime |
| `CI_DECAY` | 0.7 /day | cirrus lifetime (~1.4 days) |
| `HADLEY` | 3.0 /day | extra decay in the sinking subtropical air (clear deserts and trade zones) |
| `A_LS`, `A_CV` | 2.0, 1.2 | cloud water -> optical cover |
| `CLEAR_PCT`, `OVERCAST_PCT` | 40, 88 | coverage calibration: the share of the globe drawn clear / fully overcast. Earth is ~67 % cloudy; raise `CLEAR_PCT` for a clearer planet |

In `Climate.__init__`: the sea-surface temperature profile (warm pool, upwelling -6 C,
western boundary currents +3 C), land wetness (subtropical deserts), and the air
temperature profile.

### Tropical cyclones (`Climate._cyclones`)

Genesis about 1.7 attempts per day over water warmer than 27.8 C; steering by the model
winds plus beta drift; intensity toward a potential intensity from SST
(`(sst - 25.5) / 2.5`), decay over land; drawn as spiral bands, a central dense overcast
and an eye. The storms are procedural cloud art: they are stamped into the convective
cloud field (`cc = max(cc, storm)`) and never feed back into the QG streamfunctions,
PV, winds or moisture. A hurricane here is not a resolved vortex, and QG is weakest
in the tropics anyway.

---

## 12. Sunlight, eclipses and refraction

### Eclipse (`PENUMBRA`)

The sunlight reaching a point in space is the visible part of the Sun's disc past the
limb (0.267 degree radius) times the transmission of the air the grazing ray crosses
(Rayleigh 3.29 / 7.68 / 18.7 and aerosol 0.97 optical depths at sea level, per RGB).
Entering shadow at 500 km, the light fades over ~14 s and goes red. The marker, its
glow and the ring near it are tinted by `eclipseTint` (mix toward `ECLC`, the eclipse
colour, and toward orange where red light remains).

### Refraction (`REFRACTION`)

| Where | Model | Knobs |
| --- | --- | --- |
| Sunlight in the air (ground, sea, clouds, sky) | at every lit point the sun's direction is lifted toward that point's zenith (`sunAt()`), and that one direction drives the transmittance, the Lambert term, relief and mountain shadows, the glint, cloud shadows and the day/night switch of the city lights. Above the point's horizon the lift is Saemundsson's formula (input: TRUE elevation): 0.48 deg for a sun on the true horizon, 0.09 deg at 10 deg (the familiar 0.57 deg is Bennett's value for an object seen on the horizon; the shader fades the lift out between 8.6 and 14.5 deg elevation, where it is invisible, so `refrSin()` returns the true sun above that), scaled by the air density at the point's physical height, `altitudeKm()` (8 km scale height). Below it (clouds and air above the ground's horizon after sunset) the light has grazed down to tangent height ht: lifted `limbBend(ht) - limbBend(h)` more, so high cloud keeps its light for longer | `refrDegAt()`, `sunAt()`; CPU twin `astro.refraction_deg()` |
| The sun's disc from the ground | lifted by the same amount and squashed vertically (the lower edge lifted more); its colour is the transmittance along the line of sight, which is already the refracted ray (not lifted a second time) | disc drawn 2x true size: `0.0090` rad radius in the disc code |
| Eclipses from orbit | the limb is a lens: a ray grazing at height h is bent by 0.0203 rad x exp(-h / 7.3 km) (1.16 deg at the surface); the arriving ray is found by bisection, and its light is spread by the vertical squeeze (refractive dimming). Dim red sunlight lingers ~15 s past geometric shadow | `limbBend()` |

Not modelled: terrestrial refraction (the sea horizon is not lifted), the green flash.
Approximations: scattering phase angles and the clouds' own light-marching rays use the
true sun direction (they differ by under 1.2 deg); the mission timeline's eclipse
windows are the unrefracted cylinder (the globe's PENUMBRA lighting is refracted).
`REFRACTION` is a compile-time feature: toggling it means rebuilding the GlobeView.

### Spacecraft (`CRAFTLIGHT`)

A 3-point sphere lit by the sun through `sunVis` (wrapped Lambert, `+ 0.25`), earthshine
from the Earth below (0.30, bluish) and a floor of 0.14 so it always reads as a marker.

---

## 13. Colours and line work

`GlobeView(palette={...})` overrides any of these (RGB, 0-1):

| Key | Default | Used for |
| --- | --- | --- |
| `BGC` | (0.023, 0.039, 0.078) | panel background (the trail fades into it) |
| `CYAN` | (0.34, 0.96, 0.82) | orbit ring and trail |
| `GOLD` | (1.0, 0.78, 0.36) | spacecraft, footprint, RF link |
| `ECLC` | (0.55, 0.50, 0.98) | orbit and marker in Earth's shadow |
| `TEXTC` | (0.66, 0.78, 0.87) | graticule, idle stations |
| `NIGHT` | (0.006, 0.014, 0.032) | unlit surface (1.x) |
| `DEEP`, `OCEAN` | | 1.x ocean colours |
| `COAST` | (0.34, 0.90, 0.76) | coastline glow |
| `LIGHT` | (1.0, 0.76, 0.42) | city lights |

`GlobeView(fill=0.44)` is the disc's share of the panel at zoom 1;
`ref_radius_km` the orbit radius used to fit the ring; `inset` the margin in points.

---

## 14. The example apps

### `examples/mission_control.py`

| Constant | Default | Effect |
| --- | --- | --- |
| `QUALITY` | `'LEGACY'` | preset of the globe |
| `PERSPECTIVE` | False | pinhole camera (ray presets) |
| `CLOUDS` | 0.55 | `GlobeView(clouds=)` |
| `SPEEDS` | 0, 60, 240, 960, 2400 | time warp steps (tap) |
| `TRAIL_SECONDS` | 1400 | length of the bright trail |
| `GLOBE_SUN_GLYPH` | False | the 1.x sun icon beside the globe |
| `ALTITUDE`, `INCLINATION`, `RAAN0` | 500 km, 51.6 deg | the orbit |
| station list | 6 stations | name, latitude, longitude |

Double tap the globe: FREE -> TRACK -> ASTRO (ray presets).

### `examples/sunset.py`

| Constant | Default | Effect |
| --- | --- | --- |
| `EYE_KM` | 0.4 | camera height. 0.002 = standing on a beach (the glint and the sun set together); higher = the sun's reflection reaches the horizon before the sun does |
| `FOV` | 75 | field of view (degrees) |
| `EXPOSURE` | 0.8 | ground-view exposure |
| `START_ELEV`, `END_ELEV` | 7, -9 | sun elevation range of the loop (degrees) |
| `SPEEDS` | 0, 60, 240 | time warp (tap the bottom) |
| `DECL` | -9 deg | must match the weather film's season |
| `QUALITIES` | ULTRA, HIGH, MED | double tap cycles |
| `GROUND` | see section 10 | defines for clouds seen from below |
| `SITES` | 7 places | (name, latitude, longitude, film frame). A site shows whatever cloud the film has there; to add one, find a frame where the cloud a degree or so toward the sunset is of the kind you want |

Tap the middle to toggle refraction; drag to look around.

### `examples/phase_check.py`

`QUALITIES`, `HEADROOM_STEPS` (globes stacked for the load test), `SLIP_MS` (frame time
that counts as a dropped frame), `WARPS` (weather time warp), `SUNS` (sun geometries),
`VIEWS`.

---

## 15. Recipes

**Physical, hazy Earth (like the naked eye from the ISS):**
`GlobeView(haze=1.0, atmo_scale=1.0, defines={'LIMB_GAIN': '1.0'})`.

**Punchier satellite image:** `haze=0.05`, `exposure=0.35`, `defines={'SURF_GAIN': '0.9'}`.

**Thicker, whiter clouds:** `clouds=0.7`, `views.CLOUD_TAU = 14`,
`defines={'CLOUD_GAIN': '2.4'}`.

**Fewer, wispier clouds:** `clouds=0.4`, `defines={'ERODE': '0.25', 'BASE_AMT': '0.8'}`.

**No cirrus:** `defines={'CI_OPAC': '0.0'}`.

**Calmer sea with a sharp sun spot:** `defines={'GLINT_ALPHA': '0.06'}`.

**Mountains everywhere:** `defines={'RELIEF_GAIN': '5.0'}` (and HIGH for shadows).

**Northern summer:** `python -m orbitgfx.tools.make_weather --decl 23`; in `sunset.py`
set `DECL = math.radians(23)` and re-pick site frames; for matching sea ice re-bake the
surface with summer `N_EDGE` latitudes and a July Blue Marble.

**Smoother MED on an older phone:** copy MED with `render_scale=0.4`, `cloud_steps=16`,
`light_steps=3`, `atmo_samples=8`.

**A beach sunset:** `EYE_KM = 0.002` in `sunset.py`.

---

## 16. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| A change has no effect | Pythonista cached the old module: force-quit it |
| The globe is a gold (or other solid) ellipse | the scene also draws in `draw()` and the EffectNode reused its state; `views.PRIMER` (on by default) fixes it. `examples/draw_mix_check.py` tests it |
| `TypeError` indexing with `self.speed` (or `alpha`, `rotation`, ...) | those are Scene/Node properties stored as floats; rename your attribute. `safety.assert_clean(type(self))` in `setup()` catches this and assignments to `self.t`, `self.dt`, `self.size`, ... |
| Shader fails to compile after an edit | GLSL ES 1.0: no `pow()` of a value that can be negative (write `x * x * x`), constant loop bounds, every float literal with a decimal point. Run `python -m orbitgfx.tests.test_raycore` on a desktop: it lints and compiles every variant |
| Clouds change shape when the camera moves | someone added screen-space jitter to the march; the on-disc march must sample fixed heights (`CLOUD_NS`) |
| City lights shine through clouds | the overlay's cloud test (`cloudViewT`) needs the same weather and noise textures as the radiance pass |
| Banding at the terminator | raise `atmo_samples` |
| Blocky cloud or cirrus edges from the ground | the weather film is 512 x 256 (~80 km pixels); raise `WARP_AMT`, or re-make the film at `--grid 512` |
| Desktop tests | `python -m orbitgfx.tests.test_raycore` and `test_mission` (sandbox with moderngl + skia), LEGACY must stay pixel-identical to `tests/baseline` |
