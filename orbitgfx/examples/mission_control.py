"""ORBIT / MISSION CONTROL 4.1 -- built on orbitgfx.

Needs only the orbitgfx package (in site-packages or beside this file);
the RK45 solver and Natural Earth coastlines are bundled with it.

TOUCH
  tap               cycle speed  HOLD / 60x / 240x / 960x / 2400x  (restart when complete)
  drag globe        orbit the camera (with inertia)
  pinch globe       zoom 0.7x .. 5x
  double-tap globe  TRACK camera (chases the spacecraft); again to release
  drag map          scrub mission time (works while paused)

Physics / ops: RK45 with J2, real epoch (GMST + solar ephemeris), cylindrical
eclipse model, AOS/LOS prediction, elevation, slant range, S-band Doppler,
osculating elements with measured nodal regression, mission timeline.
"""
import math
import os
import sys
import time

try:                                   # installed (e.g. Pythonista site-packages)
    import orbitgfx  # noqa: F401
except ImportError:                    # running from inside an unzipped orbitgfx folder
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import orbitgfx  # noqa: F401      (on a desktop this also installs the scene shim)

import numpy as np
import scene
from orbitgfx.ivp import solve_ivp
from orbitgfx.coastlines import coastlines_xyz

from orbitgfx import (Camera, Gestures, GlobeView, NightMapOverlay, cached_earth_image,
                      pil_texture, safety)
from orbitgfx.astro import (C_KM_S, J2, MU, OMEGA_E, RE, eci_to_ecef, ecef_to_eci, gmst,
                            hermite, horizon_angle, julian_date, latlon, llxyz,
                            shadow_margin, small_circle, sun_dir, windows)
from orbitgfx.draw2d import MapRect, glow_dot, line, mix, panel, rgba, seam_segments, sun_glyph

# ------------------------------------------------------------------ mission
F_DOWNLINK = 2.25e9         # Hz, S-band
ALTITUDE = 500.0            # km
INCLINATION = 51.6          # deg
RAAN0 = 0.0                 # deg
ORBITS = 5
POINTS = 2400
EPOCH_UNIX = None           # None = launch "now"; or a fixed unix time
ELEV_MASK = 5.0             # deg

STATIONS = (                # name, lat, lon   (max 6: the shader has 6 slots)
    ('KIRUNA',    67.86,   20.96),
    ('WALLOPS',   37.94,  -75.46),
    ('S.POINT',   19.01, -155.66),
    ('SANTIAGO', -33.15,  -70.67),
    ('HARTRAO',  -25.89,   27.69),
    ('CANBERRA', -35.40,  148.98),
)

SPEEDS = (0.0, 60.0, 240.0, 960.0, 2400.0)
TRAIL_SECONDS = 1400.0
TEX_W = 2048
CLOUDS = 0.55
QUALITY = 'LEGACY'          # orbitgfx 2.0: 'LEGACY', 'LOW', 'MED', 'HIGH' or 'ULTRA'
                            # (ray qualities: double tap the globe twice for the astronaut view)
PERSPECTIVE = False         # True: pinhole camera (ray-core qualities only)
GLOBE_SUN_GLYPH = False     # 1.x drew a sun glyph beside the globe (kept for the baseline test)
FILL_LAND = True

BG = (0.023, 0.039, 0.078)
PANEL = (0.033, 0.065, 0.105)
BORDER = (0.18, 0.30, 0.41)
GRID = (0.14, 0.25, 0.35)
TEXT = (0.66, 0.78, 0.87)
DIM = (0.37, 0.50, 0.60)
CYAN = (0.34, 0.96, 0.82)
GOLD = (1.0, 0.78, 0.36)
RED = (1.0, 0.48, 0.44)
LAND = (0.30, 0.65, 0.54)
ECL = (0.55, 0.50, 0.98)


def hms(s):
    s = int(max(0, s))
    return '{:02d}:{:02d}:{:02d}'.format(s // 3600, (s // 60) % 60, s % 60)


def mmss(s):
    s = int(max(0, round(s)))
    return '{:02d}:{:02d}'.format(s // 60, s % 60)


# ------------------------------------------------------------------ dynamics
def body(t, y):
    x, yy, z = y[0], y[1], y[2]
    r2 = x * x + yy * yy + z * z
    k = -MU / (r2 * math.sqrt(r2))
    q = 1.5 * J2 * RE * RE / r2
    zz = 5.0 * z * z / r2
    f = 1.0 + q * (1.0 - zz)
    return np.array((y[3], y[4], y[5], k * x * f, k * yy * f, k * z * (1.0 + q * (3.0 - zz))))


def energy(r, v):
    rn = np.linalg.norm(r, axis=-1)
    s = r[..., 2] / rn
    pot = -MU / rn * (1.0 - J2 * (RE / rn) ** 2 * (1.5 * s * s - 0.5))
    return 0.5 * np.sum(v * v, axis=-1) + pot


def elements(r, v):
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    rn = float(np.linalg.norm(r))
    v2 = float(v @ v)
    ev = ((v2 - MU / rn) * r - float(r @ v) * v) / MU
    a = 1.0 / (2.0 / rn - v2 / MU)
    inc = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / hn))))
    raan = math.degrees(math.atan2(h[0], -h[1])) % 360.0
    return a, float(np.linalg.norm(ev)), inc, raan


# ------------------------------------------------------------------ scene
class Mission(scene.Scene):

    def setup(self):
        safety.assert_clean(type(self))
        self.background_color = BG
        self.jd0 = julian_date(EPOCH_UNIX)
        self.theta0 = gmst(self.jd0)
        self._propagate()
        self._load_coasts()
        self._load_stations()
        rng = np.random.RandomState(17)
        self.stars = [(float(rng.rand()), float(rng.rand()), float(rng.rand()))
                      for _ in range(125)]

        self.wall0 = time.monotonic()
        self.met0 = self.met = 0.0
        self.speed_idx = 2
        self.complete = False
        self.last_update = time.monotonic()
        self.last_hud = -1.0
        self.link_k = -1

        r0 = self.rr[0]
        self.cam = Camera(az=math.atan2(r0[1], r0[0]) + math.radians(35), el=math.radians(28))

        self.gest = Gestures(lambda: (('globe', self.globe), ('map', self.map)),
                             double_tap_regions=('globe',), pinch_regions=('globe',))
        self.gest.on_press = self._on_press
        self.gest.on_drag = self._on_drag
        self.gest.on_release = self._on_release
        self.gest.on_tap = lambda region: self._single_tap(time.monotonic())
        self.gest.on_double_tap = self._toggle_track
        self.gest.on_pinch_begin = lambda region: self.cam.pinch_begin()
        self.gest.on_pinch = lambda region, ratio: self.cam.pinch(ratio)
        self.gest.on_pinch_end = lambda region: self.cam.pinch_end()

        tex = pil_texture(self.earth_img)
        self.globe_view = GlobeView(self, tex, clouds=CLOUDS, fill=0.44,
                                    ref_radius_km=RE + ALTITUDE,
                                    quality=QUALITY, perspective=PERSPECTIVE)
        self.night = NightMapOverlay(self, tex)
        self.mapr = MapRect()

        self._create_hud()
        self.layout()

    def _propagate(self):
        radius = RE + ALTITUDE
        vc = math.sqrt(MU / radius)
        inc, raan = math.radians(INCLINATION), math.radians(RAAN0)
        r0 = radius * np.array((math.cos(raan), math.sin(raan), 0.0))
        v0 = vc * np.array((-math.sin(raan) * math.cos(inc),
                            math.cos(raan) * math.cos(inc), math.sin(inc)))
        self.period = 2 * math.pi * math.sqrt(radius ** 3 / MU)
        ts = np.linspace(0.0, self.period * ORBITS, POINTS)
        sol = solve_ivp(body, (0.0, float(ts[-1])), np.concatenate((r0, v0)),
                        t_eval=ts, rtol=1e-9, atol=1e-12, method='RK45')
        if not sol.success:
            raise RuntimeError(sol.message)
        self.ts = ts
        self.end = float(ts[-1])
        self.rr = np.ascontiguousarray(sol.y[:3].T)
        self.vv = np.ascontiguousarray(sol.y[3:].T)
        self.t_step = float(ts[1] - ts[0])

        th = self.theta0 + OMEGA_E * ts
        self.ecef = eci_to_ecef(self.rr, th)
        self.lats, self.lons = latlon(self.ecef)
        suns = sun_dir(self.jd0 + ts / 86400.0)
        self.eclipses = windows(ts, -shadow_margin(self.rr, suns), 0.0)

        en = energy(self.rr, self.vv)
        hz = np.cross(self.rr, self.vv)[:, 2]
        self.edrift = float(np.max(np.abs((en - en[0]) / en[0])))
        self.hdrift = float(np.max(np.abs((hz - hz[0]) / hz[0])))
        e0 = elements(self.rr[0], self.vv[0])
        e1 = elements(self.rr[-1], self.vv[-1])
        draan = (e1[3] - e0[3] + 180.0) % 360.0 - 180.0
        self.raan_rate = draan / self.end * 86400.0

        self.track_seg = np.column_stack((self.lons[:-1], self.lats[:-1],
                                          self.lons[1:], self.lats[1:]))
        self.track_keep = np.abs(np.diff(self.lons)) < 180

    def _load_coasts(self):
        coasts = [c for c in (np.asarray(c, float) for c in coastlines_xyz(RE))
                  if c.ndim == 2 and c.shape[1] == 3 and len(c) >= 2]
        paths = []
        for c in coasts:
            la, lo = latlon(c)
            paths.append((lo, la))
        self.map_coast = seam_segments(paths)
        self.earth_img = cached_earth_image(coasts, width=TEX_W, fill_land=FILL_LAND)

    def _load_stations(self):
        mask_ang = horizon_angle(RE + ALTITUDE, ELEV_MASK)
        self.stations = []
        for name, lat, lon in STATIONS[:6]:
            s = np.array(llxyz(lon, lat))
            rho = self.ecef - s
            el = np.degrees(np.arcsin(rho @ (s / RE) / np.linalg.norm(rho, axis=1)))
            rl, ra = small_circle(lat, lon, mask_ang)
            self.stations.append({'name': name, 'lat': lat, 'lon': lon,
                                  'ecef': s, 'el': el, 'ring': (rl, ra)})
        self.passes = []
        for k, st in enumerate(self.stations):
            for a, b in windows(self.ts, st['el'], ELEV_MASK):
                sel = (self.ts >= a) & (self.ts <= b)
                mx = float(st['el'][sel].max()) if sel.any() else ELEV_MASK
                self.passes.append((a, b, mx, k))
        self.passes.sort()

    # ---------------------------------------------------------- HUD nodes
    def _label(self, txt, sz, color, bold=False, anchor=(0.5, 0.5)):
        node = scene.LabelNode(txt, font=('Menlo-Bold' if bold else 'Menlo-Regular', sz),
                               color=color)
        node.anchor_point = anchor
        self.add_child(node)
        self._fonts[node] = (sz, bold)
        return node

    def _create_hud(self):
        self._fonts = {}
        self.heading = self._label('ORBITAL / MISSION CONTROL', 17, TEXT, True)
        self.globe_cap = self._label('ECI  /  ORBITAL VIEW', 9, DIM, anchor=(0, 0.5))
        self.cam_cap = self._label('CAM FREE', 9, DIM, anchor=(1, 0.5))
        self.map_cap = self._label('ECEF  /  GROUND TRACE', 9, DIM, anchor=(0, 0.5))
        self.l1 = self._label('', 12, CYAN, True)
        self.l2 = self._label('', 10, TEXT)
        self.l3 = self._label('', 10, TEXT)
        self.l4 = self._label('', 10, GOLD, True)
        self.l5 = self._label('', 10, TEXT)
        self.footer = self._label('TAP SPEED · PINCH ZOOM · 2×TAP TRACK · DRAG MAP SCRUB', 9, DIM)
        self.state = self._label('', 12, GOLD, True)
        self.st_labels = [self._label(st['name'], 7, DIM) for st in self.stations]

    def layout(self):
        w, h = self.size
        m, head, foot = 10.0, 34.0, 22.0
        fs = min(1.1, max(0.7, min(w, 520.0) / 400.0))
        tele_h = 5 * 14 * fs + 32
        top = h - head
        if w > h * 1.15:                      # landscape: globe left, ops right
            gw = w * 0.52 - 1.5 * m
            self.globe = (m, foot, gw, top - foot)
            rx = 2 * m + gw
            rw = w - rx - m
            self.tele = (rx, top - tele_h, rw, tele_h)
            mh = max(60.0, min(rw / 2, top - tele_h - 8 - foot))
            mw = 2 * mh
            my = foot + (top - tele_h - 8 - foot - mh) / 2
            self.map = (rx + (rw - mw) / 2, my, mw, mh)
        else:                                 # portrait: globe / telemetry / map
            mw = w - 2 * m
            mh = mw / 2
            if mh > (top - foot) * 0.40:
                mh = (top - foot) * 0.40
                mw = 2 * mh
            self.map = ((w - mw) / 2, foot, mw, mh)
            ty = foot + mh + 6
            self.tele = (m, ty, w - 2 * m, tele_h)
            gy = ty + tele_h + 6
            self.globe = (m, gy, w - 2 * m, max(80.0, top - gy))
        self.mapr.rect = self.map

        for node in (self.heading, self.globe_cap, self.cam_cap, self.map_cap, self.l1,
                     self.l2, self.l3, self.l4, self.l5, self.footer, self.state):
            sz, bold = self._fonts[node]
            node.font = ('Menlo-Bold' if bold else 'Menlo-Regular', sz * fs)
        gx, gy, gw, gh = self.globe
        mx, my, mw, mh = self.map
        tx, ty, tw, th = self.tele
        self.heading.position = (w / 2, h - head / 2)
        self.globe_cap.position = (gx + 9, gy + gh - 11)
        self.cam_cap.position = (gx + gw - 9, gy + gh - 11)
        self.map_cap.position = (mx + 7, my + mh - 10)
        self.state.position = (gx + gw / 2, gy + 18)
        self.footer.position = (w / 2, 11)
        for i, node in enumerate((self.l1, self.l2, self.l3, self.l4, self.l5)):
            node.position = (tx + tw / 2, ty + th - 11 * fs - i * 14 * fs)
        for node, st in zip(self.st_labels, self.stations):
            node.font = ('Menlo-Regular', 7 * fs)
            x, y = self.mapr.xy(st['lon'], st['lat'])
            node.position = (float(x), float(y) + 7)

        self.globe_view.layout(self.globe)
        self.night.layout(self.map)

        rects = (self.globe, self.map, self.tele)
        self.star_px = [(a * w, b * h, c) for a, b, c in self.stars
                        if not any(rx < a * w < rx + rw and ry < b * h < ry + rh
                                   for rx, ry, rw, rh in rects)]

    def did_change_size(self):
        self.layout()

    # ---------------------------------------------------------- time & input
    def update(self):
        now = time.monotonic()
        step = min(0.1, max(0.0, now - self.last_update))
        self.last_update = now
        if not self.complete:
            self.met = min(self.end, self.met0 + (now - self.wall0) * SPEEDS[self.speed_idx])
            self.complete = self.met >= self.end
        self.gest.poll(now)
        self.cam.held = self.gest.active
        pv = self._sample(self.met) if self.cam.follow else (None, None)
        self.cam.step(step, pv[0], pv[1])
        if now - self.last_hud > 0.1:
            self.last_hud = now
            self._hud()

    def touch_began(self, touch):
        self.gest.began(touch)

    def touch_moved(self, touch):
        self.gest.moved(touch)

    def touch_ended(self, touch):
        self.gest.ended(touch)

    def _on_press(self, region, now):
        if region == 'globe':
            self.cam.grab(now)

    def _on_drag(self, region, dx, dy, now):
        if region == 'globe':
            self.cam.drag(dx, dy, now)
        elif region == 'map':
            self.met = max(0.0, min(self.end, self.met + dx / self.map[2] * self.period))
            self.met0, self.wall0 = self.met, now
            self.complete = self.met >= self.end
            self.last_hud = -1.0

    def _on_release(self, region, now, was_tap):
        self.cam.release(now, fling=not was_tap)

    def _toggle_track(self, region):
        """Double tap: FREE -> TRACK -> ASTRO (ray qualities: ride the spacecraft) -> FREE."""
        if not self.cam.follow:
            self.cam.follow, self.cam.astro = True, False
        elif not self.cam.astro and QUALITY != 'LEGACY':
            self.cam.astro = True
        else:
            self.cam.follow = self.cam.astro = False
        self._hud()

    def _single_tap(self, now):
        if self.complete:
            self.met = self.met0 = 0.0
            self.complete = False
        else:
            self.met0 = self.met
            self.speed_idx = (self.speed_idx + 1) % len(SPEEDS)
        self.wall0 = now
        self._hud()

    # ---------------------------------------------------------- state
    def _sample(self, t):
        return hermite(self.ts, self.rr, self.vv, t)

    def _link(self, p, v, theta):
        """Best-elevation station currently above the mask, or None."""
        best = None
        pe = eci_to_ecef(p, theta)
        for k, st in enumerate(self.stations):
            rho = pe - st['ecef']
            rn = float(np.linalg.norm(rho))
            el = math.degrees(math.asin(float(rho @ st['ecef']) / (RE * rn)))
            if el > ELEV_MASK and (best is None or el > best[1]):
                s = ecef_to_eci(st['ecef'], theta)
                sv = OMEGA_E * np.array((-s[1], s[0], 0.0))
                rdot = float((v - sv) @ (p - s)) / rn
                best = (k, el, rn, -F_DOWNLINK * rdot / C_KM_S / 1e3)
        return best

    def _hud(self):
        t = self.met
        p, v = self._sample(t)
        theta = self.theta0 + OMEGA_E * t
        lat, lon = latlon(eci_to_ecef(p, theta))
        sun = sun_dir(self.jd0 + t / 86400.0)
        eclipsed = float(shadow_margin(p, sun)) < 0
        a, e, inc, raan = elements(p, v)
        spd = 'HOLD' if SPEEDS[self.speed_idx] == 0 else '{}x'.format(int(SPEEDS[self.speed_idx]))
        rev = min(ORBITS, int(t / self.period) + 1)

        self.l1.text = 'T+{}    {}    REV {}/{}'.format(hms(t), spd, rev, ORBITS)
        self.l2.text = 'ALT {:5.1f}km  V {:.3f}km/s  LAT {:+05.1f}  LON {:+06.1f}'.format(
            np.linalg.norm(p) - RE, np.linalg.norm(v), float(lat), float(lon))
        self.l3.text = 'a {:.1f}  e {:.4f}  i {:.2f}°  Ω {:.2f}° ({:+.2f}°/d)'.format(
            a, e, inc, raan, self.raan_rate)

        link = self._link(p, v, theta)
        self.link_k = link[0] if link else -1
        dop = ''
        for k, node in enumerate(self.st_labels):
            node.color = GOLD if k == self.link_k else DIM
        if link:
            k, el, rng, dkhz = link
            los = next((b for a_, b, _, kk in self.passes if kk == k and a_ <= t + 1 <= b + 1), None)
            self.l4.text = 'AOS {}  EL {:.0f}°  RNG {:.0f}km{}'.format(
                self.stations[k]['name'], el, rng,
                '  LOS T-' + mmss(los - t) if los is not None else '')
            self.l4.color = GOLD
            dop = 'DOP {:+.1f}kHz   '.format(dkhz)
        else:
            nxt = next((pp for pp in self.passes if pp[0] > t), None)
            self.l4.text = ('NO LINK   NEXT {} T-{}  MAX {:.0f}°'.format(
                self.stations[nxt[3]]['name'], mmss(nxt[0] - t), nxt[2])
                if nxt else 'NO LINK   NO FURTHER PASSES')
            self.l4.color = DIM

        if eclipsed:
            ex = next((b for a_, b in self.eclipses if a_ <= t <= b), None)
            pw = 'ECLIPSE' + ('  EXIT T-' + mmss(ex - t) if ex is not None and ex < self.end else '')
        else:
            en = next((a_ for a_, b in self.eclipses if a_ > t), None)
            pw = 'SUNLIT' + ('  ECL T-' + mmss(en - t) if en is not None else '')
        self.l5.text = '{}{}   ΔE {:.0e} ΔHz {:.0e}'.format(dop, pw, self.edrift, self.hdrift)
        self.l5.color = ECL if eclipsed else TEXT

        z = self.cam.zoom
        mode = 'CAM ASTRO' if self.cam.follow and self.cam.astro else \
            'CAM TRACK' if self.cam.follow else 'CAM FREE'
        self.cam_cap.text = mode + \
            ('  {:.1f}×'.format(z) if abs(z - 1) > 0.05 else '')
        self.cam_cap.color = CYAN if self.cam.follow else DIM
        self.state.text = 'MISSION COMPLETE  /  TAP TO RESET' if self.complete else ''

    # ---------------------------------------------------------- map (classic layer)
    def _map_panel(self, t, p, sun, theta):
        M = self.mapr
        panel(self.map, PANEL, BORDER)
        M.graticule(GRID)
        if len(self.map_coast):
            M.segments(self.map_coast, rgba(LAND, .45), .9)

        sun_e = eci_to_ecef(sun, theta)
        self.night.update(sun_e)
        slat, slon = latlon(sun_e)
        px, py = M.xy(float(slon), float(slat))
        sun_glyph(float(px), float(py), GOLD, .95, core=3.0, ray0=4.5, ray1=7.0, weight=1.0)

        for k, st in enumerate(self.stations):
            rl, ra = st['ring']
            on = k == self.link_k
            M.polyline(rl, ra, rgba(GOLD if on else DIM, .9 if on else .4), 1.3 if on else .7)
            sx, sy = M.xy(st['lon'], st['lat'])
            scene.no_stroke(); scene.fill(*rgba(GOLD if on else TEXT, .95))
            scene.rect(float(sx) - 1.8, float(sy) - 1.8, 3.6, 3.6)

        # Past track solid, predicted track dashed.
        i_now = int(np.searchsorted(self.ts, t))
        ids = np.arange(0, len(self.track_seg), 4)
        ids = ids[self.track_keep[ids]]
        past, fut = ids[ids < i_now], ids[ids >= i_now]
        if len(past):
            M.segments(self.track_seg[past], rgba(mix(CYAN, BG, .50)), 1.0)
        if len(fut):
            fseg = self.track_seg[fut].copy()
            fseg[:, 2] = 0.5 * (fseg[:, 0] + fseg[:, 2])
            fseg[:, 3] = 0.5 * (fseg[:, 1] + fseg[:, 3])
            M.segments(fseg, rgba(mix(CYAN, BG, .66)), .8)

        # Bright trail, eclipse-coloured.
        tt = np.linspace(max(0.0, t - TRAIL_SECONDS), t, 110)
        pts = self._sample(tt)[0]
        la, lo = latlon(eci_to_ecef(pts, self.theta0 + OMEGA_E * tt))
        sh = shadow_margin(pts, sun) < 0
        X, Y = M.xy(lo, la)
        n = len(tt)
        for j in range(n - 1):
            if abs(lo[j + 1] - lo[j]) > 180:
                continue
            c = ECL if sh[j] else CYAN
            line(X[j], Y[j], X[j + 1], Y[j + 1], rgba(mix(c, BG, .6 * (1 - j / n))), 1.4 + 1.3 * j / n)

        # Live coverage footprint.
        lat, lon = latlon(eci_to_ecef(p, theta))
        rl, ra = small_circle(float(lat), float(lon), math.acos(RE / float(np.linalg.norm(p))), 64)
        M.polyline(rl, ra, rgba(GOLD, .6), 1.0)

        sx, sy = M.xy(float(lon), float(lat))
        glow_dot(float(sx), float(sy), RED, 2.5)
        for k in range(ORBITS + 1):
            tm = k * self.period
            if tm > t:
                break
            i = min(len(self.ts) - 1, int(round(tm / self.t_step)))
            mx, my = M.xy(self.lons[i], self.lats[i])
            scene.no_stroke(); scene.fill(*rgba(GOLD)); scene.ellipse(float(mx) - 2, float(my) - 2, 4, 4)

    def _tele_panel(self, t):
        panel(self.tele, PANEL, BORDER)
        tx, ty, tw, th = self.tele
        x0, x1 = tx + 12, tx + tw - 12
        by, bh = ty + 8, 5

        def X(tm):
            return x0 + (x1 - x0) * tm / self.end

        scene.no_stroke()
        scene.fill(*rgba(mix(GOLD, PANEL, .72))); scene.rect(x0, by, x1 - x0, bh)
        scene.fill(*rgba(mix(ECL, PANEL, .45)))
        for a, b in self.eclipses:
            scene.rect(X(a), by, max(1.0, X(b) - X(a)), bh)
        scene.fill(*rgba(CYAN, .85))
        for a, b, _, _ in self.passes:
            scene.rect(X(a), by + bh + 2, max(1.5, X(b) - X(a)), 3)
        for k in range(1, ORBITS):
            line(X(k * self.period), by - 2, X(k * self.period), by, rgba(DIM, .8), 1)
        cx = X(t)
        line(cx, by - 3, cx, by + bh + 6, (1, 1, 1, .95), 1.5)

    # ---------------------------------------------------------- frame
    def draw(self):
        self.update()
        w, h = self.size
        scene.no_stroke(); scene.fill(*rgba(BG)); scene.rect(0, 0, w, h)
        for x, y, c in self.star_px:
            scene.fill(1, 1, 1, .20 + .65 * c)
            s = 1.2 if c < .8 else 2.0
            scene.rect(x, y, s, s)
        t = self.met
        p, v = self._sample(t)
        sun = sun_dir(self.jd0 + t / 86400.0)
        theta = self.theta0 + OMEGA_E * t

        panel(self.globe, PANEL, BORDER)
        if GLOBE_SUN_GLYPH:
            mk = self.globe_view.sun_marker_position(sun, self.cam)
            if mk:
                sun_glyph(mk[0], mk[1], GOLD, mk[2])
        self.globe_view.update(
            self.cam, sun, theta, sat=p, vel=v,
            trail_rad=min(t, TRAIL_SECONDS) * 2 * math.pi / self.period,
            stations=[(st['ecef'], k == self.link_k) for k, st in enumerate(self.stations)],
            cloud_hours=t / 3600.0)
        self._tele_panel(t)
        self._map_panel(t, p, sun, theta)


if __name__ == '__main__':
    scene.run(Mission(), show_fps=False)
