# orbitgfx: what is verified, what is approximate, what is art

orbitgfx mixes three kinds of model, and they need different claims:

- **governing equations** (gravity, the ODE solver, QG dynamics, Beer–Lambert
  extinction): implemented from physics and *verified* against their own
  equations and independent references;
- **reduced or empirical closures** (refraction formulas, moisture proxies,
  cloud categories, multiple-scattering ambient): physically informed, tuned,
  not validated against observations;
- **visual stylisation** (thicker atmosphere and clouds, haze removal, limb
  gain, a 2x sun disc, scripted hurricanes, a looping weather film).

Verification below means "the code does what its equations say". It is not
validation against nature: no number here is a statement of real-world
accuracy. A J2-only orbit that agrees with SciPy to centimetres is still a
J2-only orbit.

## Model inventory

| Subsystem | Model | Status | Describe it as |
| --- | --- | --- | --- |
| Orbit gravity | Two-body + J2 acceleration from the J2 potential | Verified: force = −∇potential (2e-10 relative), J2 nodal regression vs theory (`test_physics`) | idealised dynamics; no drag, higher harmonics, Sun/Moon, SRP |
| ODE solver (`ivp.py`) | Dormand–Prince 5(4), dense output | Verified vs SciPy RK45 and forward/backward runs (`test_rk45_against_reference`) | solid numerics |
| ECI ↔ ECEF, Sun | Mean GMST about a fixed z axis; Almanac low-precision Sun | Consistent; no UT1, precession, nutation | low-precision demonstration frames |
| Ground stations | WGS84 geodetic position and ellipsoid normal, elevation mask 5° | Verified fixtures: equator, poles, on-ellipsoid, normal = gradient, exact elevations (`test_geodetic_fixtures`, `test_elevation_horizon`) | geodetic; contact times still approximate (sampled, idealised orbit) |
| Ground track, footprints, map | Spherical lat/lon | Drawing approximation | spherical visualisation |
| Orbital eclipse | Cylinder for the timeline; shader solar disc, penumbra and limb lens | Disclosed approximation | idealised illumination |
| Atmosphere transmittance | Exponential Rayleigh / aerosol, tent ozone, Beer–Lambert | LUT vs its direct integral (`test_transmittance_lut_accuracy`); absolute radiance not benchmarked | physically informed reduced model |
| Scattering, refraction | Single scattering + tuned ambient, haze, limb gain; Saemundsson refraction above a point's horizon, exponential limb bend below it, one refracted sun direction per point (`sunAt`) | Heights in one unit (`test_shader_heights_use_one_unit`, `test_refraction_height_follows_atmo_scale`); GPU `refrDegAt` = CPU `astro.refraction_deg`, continuous and monotonic (`test_refraction_model_matches_cpu_twin`); `sunAt` lift, azimuth, off-switch (`test_sun_direction_refracted`); ground lit by the refracted sun (`test_refraction_lights_the_ground`). Formulas not checked against a reference ray tracer | physically informed with artistic controls |
| QG circulation (`weather.py` tier A, B) | Two-layer Phillips QG, Fourier/FD inversion, semi-Lagrangian advection, relaxation, friction, polar filter | PV ↔ streamfunction round trip 6e-14, constant tracer kept (`test_qg_inversion_round_trip`); short-run stability (`test_weather_model_short_run`) | idealised midlatitude dynamics |
| Moisture and clouds (tiers C, D) | Relative saturation proxy, normalised ascent, clipping, percentile coverage calibration | Visual only | dynamically driven procedural clouds |
| Tropical cyclones (tier E) | Seeded genesis, steering, spiral-band stamp into `cc` | Art: never feeds back into the dynamics | empirical storm artwork |
| Weather film (tier F) | 96 frames, 4-day seamless loop, one season (declination −9°) | Determinism and playback tests | a chosen season, not observations |
| Ocean, clouds rendering | GGX / Schlick / Smith; volumetric cloud approximations | No reference BRDF or radiance bench | plausible PBR with artistic controls |

## Units and conventions

| Quantity | Unit / convention |
| --- | --- |
| Positions (Python) | km, ECI unless named `ecef` |
| Positions (ray shader) | drawn Earth radii. With `SCATTER`, 1 unit = `u_atm.y` = 6378.137 / `atmo_scale` km; convert with `altitudeKm(r)`. Orbit, spacecraft and their eclipse are true scale (`RE_KM`) |
| Latitude, longitude | degrees; stations geodetic (WGS84), map and tracks spherical |
| Time | s in the mission app, days in `weather.py`, cloud hours in the shader |
| Sun irradiance | HDR units (`SUN_IRRADIANCE` = 16), shown through `SCATTER_EXPOSURE` |
| Extinction | per km, one coefficient per RGB channel (`ATMO` in `luts.py`, `BETA_*` in `shaders.py`) |
| Weather winds | radians per day on the unit sphere (1 rad/day = 73.7 m/s) |
| Cloud fields | 0–1 optical cover proxies, not cloud water or humidity |

## Parameter ledger (main knobs)

| Parameter | Default | Basis | Kind | Effect |
| --- | --- | --- | --- | --- |
| `MU`, `RE`, `J2` | 398600.4418 km³/s², 6378.137 km, 1.08262668e-3 | WGS84 / NASA Earth fact sheet | physical | orbit |
| WGS84 flattening | 1/298.257223563 | WGS84 | physical | station positions, elevation |
| `ELEV_MASK` | 5° | common ground-station mask | operational choice | contact windows |
| Rayleigh, aerosol scale heights | 8 km, 1.2 km | standard atmosphere | physical | extinction, colour |
| `ATMO_SCALE` | 1.5 | picked on device | artistic | atmosphere drawn 1.5x thicker; brighter, less red limb (grazing paths shorten). 1.0 is physical |
| `HAZE` | 0.10 | "Rayleigh-corrected" satellite imagery | artistic | removes air light looking down |
| `LIMB_GAIN` | 1.8 | look at satellite-image exposure | artistic | limb brightness |
| `MS_GAIN` | 0.10 | tuned | empirical closure | multiple-scattering ambient |
| `CLOUD_SCALE` | 4.0 | phone-scale readability | artistic | cloud shell drawn 4x thicker |
| Sun disc | 2x true radius | readability | artistic | sun size from the ground |
| Refraction | Saemundsson (true elevation in), 0.48° at 0°; limb bend 0.0203 rad · exp(−h/7.3 km) below a point's horizon | literature formulas | empirical closure | sunset timing, ground and cloud light at twilight, eclipse reddening |
| `CLEAR_PCT`, `OVERCAST_PCT` | 40, 88 | Earth ~67 % cloudy | calibration | global cloud cover |
| QG `ld_km`, `shear`, `t_rad`, `t_fric`, `t_drag1` | 1000 km, 1.6 rad/day, 20 d, 4 d, 5 d | tuned for a realistic-looking jet | closure | storm size, jet strength |

## Physical baseline vs. visual defaults

For checks against physics, build a globe with the stylisation off:

```python
GlobeView(host, tex, quality='HIGH', atmo_scale=1.0, haze=1.0, cloud_scale=1.0,
          defines={'LIMB_GAIN': '1.0'})
```

The shipped defaults (`atmo_scale=1.5`, `haze=0.1`, `cloud_scale=4`,
`LIMB_GAIN=1.8`, 2x sun disc) are the phone look and are not physical.

## Test tiers

| Tier | Needs | Where |
| --- | --- | --- |
| Numerics | NumPy (SciPy optional) | `tests/test_geometry.py`, `test_physics`, LUT, weather tests |
| Shader text | nothing | `test_variants_lint`, `test_shader_variants_lint`, `test_shader_heights_use_one_unit` |
| Desktop compile and render | moderngl + EGL, skia-python | everything marked `@gpu`; SKIPPED without a GL context, failed if `ORBITGFX_REQUIRE_GPU=1` |
| Pixel regression | the above | `tests/baseline/` (1.x frames; rendered with 1.x's spherical stations) |
| On device | Pythonista | the checklist below; the desktop sandbox is a harness, not the specification |

### On-device gate (Pythonista)

1. Compile every quality preset (`LEGACY`, `LOW`, `MED`, `HIGH`, `ULTRA`,
   including the reduced-resolution two-pass ones) with all features; record compiler warnings.
2. Record texture count, memory, frame rate and thermal behaviour over a few
   minutes per preset.
3. Screenshot fixed poses (camera, sun, weather time) and compare with the
   desktop renders; list the differences that are allowed.
4. Check `update()` runs once per frame (`examples/mission_control.py` no
   longer calls it from `draw()`).

## Data provenance

| File | Source (as recorded in this repo) | Notes |
| --- | --- | --- |
| `data/ne_110m_coastline.geojson` | Natural Earth 1:110m coastlines | public domain |
| `data/surface/color.jpg` | NASA Blue Marble Next Generation, March 2004 (`world.topo.200403`) | sea ice painted in by `tools/make_surface.py` |
| `data/surface/earth.png` (B) | NASA Black Marble 2016 | city lights |
| `data/surface/aux.png` | R: land height map passed to `make_surface.py`; G: sea-ice concentration drawn from tables | height-map source product not recorded: confirm before redistribution |
| `data/weather/` | generated by `tools/make_weather.py` | synthetic, not observations |

Confirm the exact NASA product, processing history and credit line for each
redistributed image before quoting source metadata.

## Known limits

- Contact predictions: AOS/LOS times depend on the trajectory sampling and on
  the idealised orbit; expect seconds to tens of seconds of difference from an
  operational tool.
- `atmo_scale ≠ 1` changes the limb's optical depth, not just its drawing.
- QG is weakest near the equator; tropical weather is mostly artwork.
- Desktop compile ≠ Pythonista compile.
