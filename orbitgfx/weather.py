"""Offline weather model for orbitgfx clouds (2.0 phase 3b).

Run once on a desktop (or in Pythonista, slowly); the phone only plays back the
frames it writes. Not a forecast: an idealised but physically based general
circulation at 0.7-1.4 degree resolution.

Dynamics: the two-layer quasi-geostrophic (Phillips) model on the sphere.
    q1 = lap(psi1) + f - F(lat) (psi1 - psi2)        upper layer potential vorticity
    q2 = lap(psi2) + f + F(lat) (psi1 - psi2)        lower layer
    Dq_i/Dt = forcing_i
  - Radiative relaxation of the layer-thickness (temperature) field toward an
    equator-to-pole gradient (stronger in the winter hemisphere) feeds
    baroclinic instability: cyclones, anticyclones, fronts and the eddy-driven
    jet stream with its Rossby-wave meanders come out of the equations.
  - Surface friction pulls the lower layer toward the observed surface winds
    (trades, westerlies, polar easterlies).
  - F = f^2 / (g' H) with a 700 km deformation radius at 45 degrees.
  Numerics: Fourier in longitude, finite differences in latitude (per-
  wavenumber inverses precomputed), semi-Lagrangian advection along great
  circles, latitude-dependent spectral filter at the poles.

Moisture and cloud (column budgets, parameterised):
  - Sea surface temperature with warm pools and cold upwelling off the
    subtropical west coasts; land surface heated by the real sun position
    (diurnal cycle) and dried in the subtropics.
  - Vapour: evaporation toward saturation (Clausius-Clapeyron), advected by
    the lower layer; saturation follows the model's own temperature.
  - Vertical motion from the thermodynamic budget: the part of the layer
    temperature change that advection and radiation do not explain is
    adiabatic, i.e. rising (cooling) or sinking air.
  - Large-scale cloud where air rises or saturates (fronts, comma clouds,
    warm conveyor belts), convective cloud where warm moist air is unstable
    (ITCZ, tropical cyclones' bands, afternoon storms over land), decks of
    stratocumulus over cold water under sinking air, and decay in subsidence.

Units: Earth radius 1, time in days, winds in radians per day (1 rad/day =
73.7 m/s).

What this is (validation tiers, see VALIDATION.md):
  A  QG equations and discretisation (Model.pv / Model.invert, advection):
     verified numerically (inversion round trip, constant tracer).
  B  forcing and dissipation closures (relaxation, friction, drag, polar
     filter): prescribed, tuned for a realistic-looking jet.
  C  moisture proxies (relative saturation curve, normalised ascent, clipping):
     not quantitative humidity or cloud water.
  D  cloud categories and coverage, calibrated by percentile (CLEAR_PCT /
     OVERCAST_PCT) to a target clear/overcast share: visual.
  E  tropical cyclones (Climate._cyclones): seeded, scripted cloud artwork that
     does not touch the dynamics.
  F  the baked film: 4-day seamless loop of one season, cross-faded at the ends.
It is a QG midlatitude circulation plus empirical tropical convection and storm
artwork, not a primitive-equation atmosphere or a forecast.
"""
import math

import numpy as np

OMEGA = 2.0 * math.pi * 1.0027      # rad / day
DEG = math.pi / 180.0


class Grid(object):
    def __init__(self, nlon=512, nlat=256):
        self.nlon, self.nlat = nlon, nlat
        self.dlon = 2 * math.pi / nlon
        self.dlat = math.pi / nlat
        self.lon = -math.pi + (np.arange(nlon) + 0.5) * self.dlon        # cell centres
        self.lat = -math.pi / 2 + (np.arange(nlat) + 0.5) * self.dlat    # south -> north
        self.LAT, self.LON = np.meshgrid(self.lat, self.lon, indexing='ij')
        self.cos = np.cos(self.LAT)
        self.sin = np.sin(self.LAT)
        cl, sl = np.cos(self.LON), np.sin(self.LON)
        self.P = np.stack([self.cos * cl, self.cos * sl, self.sin], -1)       # unit vectors
        self.east = np.stack([-sl, cl, np.zeros_like(cl)], -1)
        self.north = np.stack([-self.sin * cl, -self.sin * sl, self.cos], -1)
        self.f = 2 * OMEGA * self.sin
        self.M = nlon // 3                                                    # spectral truncation
        self.m = np.arange(self.M + 1)
        # latitude-dependent filter (polar filter + 2/3 truncation)
        kmax = np.maximum(self.M * np.cos(self.lat)[:, None], 3.0)
        self.kmax = kmax
        self.filt = np.exp(-(self.m[None, :] / kmax) ** 8)

    # lat-lon field <-> Fourier coefficients
    def fwd(self, a):
        return np.fft.rfft(a, axis=1)[:, :self.M + 1]

    def inv(self, A):
        out = np.zeros((self.nlat, self.nlon // 2 + 1), complex)
        out[:, :self.M + 1] = A
        return np.fft.irfft(out, n=self.nlon, axis=1)

    def filter(self, a, dt=None, t_grid=0.25):
        """Polar filter + scale-selective (del^4-like) damping: the shortest resolved
        waves decay with e-folding t_grid days, synoptic scales are untouched."""
        F = self.filt
        if dt is not None:
            F = F * np.exp(-dt / t_grid * (self.m[None, :] / self.kmax) ** 4)
        out = self.shapiro(self.inv(self.fwd(a) * F))
        return out

    @staticmethod
    def shapiro(a):
        """4th-order Shapiro filter in latitude: a - (d2/4)^2 a. Removes the 2-grid
        wave completely, leaves smooth scales almost untouched."""
        def d2(x):
            p = np.vstack([x[:1], x, x[-1:]])
            return p[2:] - 2 * p[1:-1] + p[:-2]
        return a - d2(d2(a)) / 16.0

    def lat_operator(self, m, diag=None):
        """Matrix of the spherical Laplacian for zonal wavenumber m (+ diag)."""
        n, d = self.nlat, self.dlat
        cj = np.cos(self.lat)
        ch = np.cos(self.lat + d / 2)       # interface j+1/2
        cl = np.cos(self.lat - d / 2)       # interface j-1/2
        ch[-1] = 0.0
        cl[0] = 0.0
        A = np.zeros((n, n))
        idx = np.arange(n)
        A[idx, idx] = -(ch + cl) / (d * d * cj) - (m * m) / (cj * cj)
        A[idx[:-1], idx[:-1] + 1] = ch[:-1] / (d * d * cj[:-1])
        A[idx[1:], idx[1:] - 1] = cl[1:] / (d * d * cj[1:])
        if diag is not None:
            A[idx, idx] += diag
        return A

    def inverses(self, diag=None):
        mats = np.stack([self.lat_operator(m, diag) for m in self.m])
        if diag is None:
            inv = np.empty_like(mats)
            inv[1:] = np.linalg.inv(mats[1:])
            inv[0] = np.linalg.pinv(mats[0], rcond=1e-9)   # m = 0: drop the constant null space
            return inv
        return np.linalg.inv(mats)

    def solve(self, inv, rhs):
        R = self.fwd(rhs)                                     # (nlat, M+1)
        b = np.stack([R.real.T, R.imag.T], -1)                # (M+1, nlat, 2)
        x = np.matmul(inv, b)
        return self.inv((x[..., 0] + 1j * x[..., 1]).T)

    def d_lon(self, a):
        return self.inv(self.fwd(a) * (1j * self.m)[None, :])

    def d_lat(self, a):
        g = np.empty_like(a)
        g[1:-1] = (a[2:] - a[:-2]) / (2 * self.dlat)
        g[0] = (a[1] - a[0]) / self.dlat
        g[-1] = (a[-1] - a[-2]) / self.dlat
        return g

    def lap(self, a):
        """Spherical Laplacian (same discretisation as the inverses)."""
        A = self.fwd(a)
        d = self.dlat
        cj = np.cos(self.lat)[:, None]
        ch = np.cos(self.lat + d / 2)[:, None].copy()
        cl = np.cos(self.lat - d / 2)[:, None].copy()
        ch[-1] = 0.0
        cl[0] = 0.0
        up = np.vstack([A[1:], A[-1:]])
        dn = np.vstack([A[:1], A[:-1]])
        L = (ch * (up - A) - cl * (A - dn)) / (d * d * cj) - (self.m[None, :] ** 2) / (cj * cj) * A
        return self.inv(L)

    # semi-Lagrangian departure points and bilinear interpolation weights
    def departure(self, u, v, dt, u_old=None, v_old=None, iters=2):
        """Departure points of trajectories arriving at the grid in dt (2nd order):
        the midpoint wind is extrapolated to t + dt/2 (1.5 V(t) - 0.5 V(t - dt));
        without that, two-level semi-Lagrangian schemes slowly gain energy."""
        V = u[..., None] * self.east + v[..., None] * self.north
        if u_old is not None:
            Vo = u_old[..., None] * self.east + v_old[..., None] * self.north
            Vh = 1.5 * V - 0.5 * Vo
        else:
            Vh = V
        Vm = Vh
        for _ in range(iters):
            Dm = self.P - Vm * (dt / 2)
            Dm /= np.linalg.norm(Dm, axis=-1, keepdims=True)
            w = self.weights(Dm)
            Vm = np.stack([self.interp(Vh[..., k], w) for k in range(3)], -1)
        D = self.P - Vm * dt
        D /= np.linalg.norm(D, axis=-1, keepdims=True)
        return self.weights(D)

    def weights(self, D):
        lat = np.arcsin(np.clip(D[..., 2], -1, 1))
        lon = np.arctan2(D[..., 1], D[..., 0])
        x = (lon + math.pi) / self.dlon - 0.5
        y = (lat + math.pi / 2) / self.dlat - 0.5
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx, fy = x - x0, y - y0
        y0c = np.clip(y0, 0, self.nlat - 1)
        y1c = np.clip(y0 + 1, 0, self.nlat - 1)
        fyr = fy
        fy = np.where(y0 < 0, 1.0, np.where(y0 + 1 > self.nlat - 1, 0.0, fy))
        return (y0c, y1c, x0 % self.nlon, (x0 + 1) % self.nlon, fx, fy, fyr, y0)

    @staticmethod
    def interp(a, w):
        y0, y1, x0, x1, fx, fy = w[:6]
        return ((a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy)
                + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy)

    def interp3(self, a, w):
        """Bicubic (Catmull-Rom) interpolation: far less numerical diffusion than
        bilinear, which otherwise damps baroclinic waves as fast as they grow."""
        x0, fx, yr, yb = w[2], w[4], w[6], w[7]
        def cw(t):
            t2, t3 = t * t, t * t * t
            return (-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0,
                    -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2)
        wx, wy = cw(fx), cw(yr)
        ny, nx = self.nlat, self.nlon
        out = 0.0
        for j in range(4):
            yy = np.clip(yb + j - 1, 0, ny - 1)
            row = 0.0
            for i in range(4):
                row = row + a[yy, (x0 + i - 1) % nx] * wx[i]
            out = out + row * wy[j]
        return out


class Model(object):
    """Two-layer QG dynamics (see module docstring)."""

    def __init__(self, grid, ld_km=1000.0, shear=1.6, t_rad=20.0, t_fric=4.0, t_drag1=5.0,
                 season=0.0, seed=1):
        g = self.g = grid
        self.season = season                      # sun declination, radians
        f45 = 2 * OMEGA * math.sin(math.pi / 4)
        ld = ld_km / 6371.0
        self.F = (1.0 / ld ** 2) * (g.sin[:, 0] / math.sin(math.pi / 4)) ** 2       # per latitude
        self.F = np.maximum(self.F, 0.15 / ld ** 2)
        self.inv_b = g.inverses()
        self.inv_c = g.inverses(diag=-2 * self.F)
        self.t_rad, self.t_fric, self.t_drag1 = t_rad, t_fric, t_drag1
        self.shear = shear
        self.set_season(season)
        # surface wind target: trades, westerlies, polar easterlies (rad/day)
        la = g.lat
        u2 = (-0.10 * np.exp(-(la / (18 * DEG)) ** 2) * (1 - np.exp(-(la / (6 * DEG)) ** 2) * 0.6)
              + 0.12 * np.exp(-((np.abs(la) - 48 * DEG) / (10 * DEG)) ** 2)
              - 0.05 * np.exp(-((np.abs(la) - 75 * DEG) / (8 * DEG)) ** 2))
        self.u2_target = np.repeat(u2[:, None], g.nlon, 1)
        self.zeta2_target = -g.d_lat(self.u2_target * g.cos) / g.cos
        # start: equilibrium thickness, at rest, small random noise
        rng = np.random.RandomState(seed)
        self.psi1 = 0.5 * self.tau_eq + g.filter(rng.randn(g.nlat, g.nlon) * 1e-4)
        self.psi2 = -0.5 * self.tau_eq
        self.q1, self.q2 = self.pv(self.psi1, self.psi2)
        self.t = 0.0

    def set_season(self, decl):
        g = self.g
        la = g.lat
        winter = 1.0 - 0.35 * np.sign(la) * (decl / (23.44 * DEG))      # stronger in winter
        s = self.shear * np.exp(-((np.abs(la) - 45 * DEG) / (16 * DEG)) ** 2) * winter   # mid-latitude baroclinic zone
        # u1 - u2 = -d tau / d lat  (thermal wind in streamfunction space)
        tau = -np.concatenate([[0.0], np.cumsum((s[1:] + s[:-1]) / 2) * g.dlat])
        tau -= tau.mean()
        self.tau_eq = np.repeat(tau[:, None], g.nlon, 1)
        self.decl = decl

    def pv(self, p1, p2):
        g = self.g
        tau = p1 - p2
        F = self.F[:, None]
        return g.lap(p1) + g.f - F * tau, g.lap(p2) + g.f + F * tau

    def invert(self):
        g = self.g
        pb = g.solve(self.inv_b, 0.5 * (self.q1 + self.q2) - g.f)
        pc = g.solve(self.inv_c, 0.5 * (self.q1 - self.q2))
        self.psi1, self.psi2 = pb + pc, pb - pc

    def winds(self, psi):
        g = self.g
        u = -g.d_lat(psi)
        v = g.d_lon(psi) / np.maximum(g.cos, 0.02)
        return u, v

    def step(self, dt=1.0 / 48.0):
        g = self.g
        tau_old = self.psi1 - self.psi2
        u1, v1 = self.winds(self.psi1)
        u2, v2 = self.winds(self.psi2)
        o = getattr(self, '_old', None) or (u1, v1, u2, v2)
        self.u1, self.v1, self.u2, self.v2 = u1, v1, u2, v2
        w1 = g.departure(u1, v1, dt, o[0], o[1])
        w2 = g.departure(u2, v2, dt, o[2], o[3])
        wm = g.departure(0.5 * (u1 + u2), 0.5 * (v1 + v2), dt, 0.5 * (o[0] + o[2]), 0.5 * (o[1] + o[3]))
        self._old = (u1, v1, u2, v2)
        self.wm = wm
        F = self.F[:, None]
        heat = F * (tau_old - self.tau_eq) / self.t_rad
        z1 = g.lap(self.psi1)
        z2 = g.lap(self.psi2)
        q1 = g.interp3(self.q1, w1) + dt * (heat - z1 / self.t_drag1)
        q2 = g.interp3(self.q2, w2) + dt * (-heat - (z2 - self.zeta2_target) / self.t_fric)
        self.q1 = g.filter(q1 - g.f, dt) + g.f
        self.q2 = g.filter(q2 - g.f, dt) + g.f
        self.invert()
        tau = self.psi1 - self.psi2
        # adiabatic part of the thickness tendency -> vertical motion (>0 rising)
        dtau = (tau - g.interp(tau_old, wm)) / dt + (tau_old - self.tau_eq) / self.t_rad
        self.omega = -dtau
        self.t += dt
        return self

    def energy(self):
        u1, v1 = self.winds(self.psi1)
        return float(np.mean((u1 ** 2 + v1 ** 2) * self.g.cos) / np.mean(self.g.cos))


# ======================================================================= moisture
def qsat(T):
    """Saturation vapour (relative to 30 C), Clausius-Clapeyron ~7 % per K."""
    return np.exp(0.067 * (T - 30.0))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


KW = 30.0    # large-scale condensation per unit ascent (1/day)
KC = 30.0    # convective condensation (1/day)
LS_DECAY = 1.5   # frontal cloud decay (1/day)
CV_DECAY = 1.2   # convective cloud / anvil decay (1/day)
A_LS, A_CV = 2.0, 1.2   # cloud water -> optical cover
CLEAR_PCT, OVERCAST_PCT = 40.0, 88.0
CI_DECAY = 0.7   # cirrus lifetime ~1.4 days
HADLEY = 3.0     # extra cloud decay in the subtropical descent (1/day)


class Climate(object):
    """Dynamics + moisture + clouds on one grid. Time in days from 00:00 UTC."""

    def __init__(self, nlon=256, nlat=128, decl_deg=0.0, land=None, seed=1, model=None, **model_kw):
        self.decl = math.radians(decl_deg)
        if model is None:
            g = Grid(nlon, nlat)
            model = Model(g, season=self.decl, seed=seed, **model_kw)
            n = g.inv(g.fwd(np.random.RandomState(seed + 4).randn(nlat, nlon)) * np.exp(-(g.m / 8.0) ** 2)[None, :])
            model.psi1 += n * 2e-4
            model.q1, model.q2 = model.pv(model.psi1, model.psi2)
        self.model = model
        g = self.g = model.g
        nlon, nlat = g.nlon, g.nlat
        if land is None:
            from .luts import _land_mask
            land = np.flipud(_land_mask((nlon, nlat)))          # rows south -> north
        self.land = np.clip(land, 0.0, 1.0)
        la = g.LAT
        alat = np.abs(la)
        phs = 0.35 * self.decl                                    # thermal equator follows the sun
        # sea surface temperature (C): zonal profile, cold upwelling off subtropical west coasts,
        # warm western boundary currents and the west Pacific warm pool
        sst = 29.5 * np.cos(np.clip(la - phs, -1.5, 1.5)) ** 1.6 - 1.5
        east_land = np.zeros_like(self.land)
        west_land = np.zeros_like(self.land)
        step = max(1, nlon // 256)
        for k in range(step, 16 * step, step):
            east_land = np.maximum(east_land, np.roll(self.land, -k, axis=1) * (1 - k / (18.0 * step)))
            west_land = np.maximum(west_land, np.roll(self.land, k, axis=1) * (1 - k / (18.0 * step)))
        sub = np.exp(-((alat - 22 * DEG) / (11 * DEG)) ** 2)
        self.upwell = (1 - self.land) * (east_land > 0.4) * sub
        for _ in range(6):
            self.upwell = Grid.shapiro(g.inv(g.fwd(self.upwell) * np.exp(-(g.m / 24.0) ** 2)[None, :]))
        self.upwell = np.clip(self.upwell, 0.0, 1.0)
        wbc = (1 - self.land) * (west_land > 0.4) * np.exp(-((alat - 35 * DEG) / (9 * DEG)) ** 2)
        pool = np.exp(-((g.LON - 140 * DEG) / (25 * DEG)) ** 2 - (la / (14 * DEG)) ** 2)
        self.sst = np.maximum(sst - 6.0 * self.upwell + 3.0 * wbc + 1.2 * pool, -1.8)
        self.sst_zonal = self.sst.mean(1, keepdims=True)
        # land: dry subtropical deserts, wet tropics / mid-latitudes
        self.wet = np.clip(0.55 + 0.35 * np.exp(-(la / (12 * DEG)) ** 2)
                           - 0.50 * np.exp(-((alat - 24 * DEG) / (8 * DEG)) ** 2), 0.05, 1.0)
        self.T_clim = 29.0 * np.cos(np.clip(la - phs, -1.5, 1.5)) ** 1.5 - 4.0   # near-surface air
        T0 = self.T_clim
        self.qv = 0.75 * qsat(T0)
        self.cl = np.zeros_like(self.qv)          # large-scale (frontal) cloud water
        self.cc = np.zeros_like(self.qv)          # convective cloud water
        self.ci = np.zeros_like(self.qv)          # cirrus (ice, upper level)
        self.sc = np.zeros_like(self.qv)          # stratocumulus (diagnostic)
        self.w = np.zeros_like(self.qv)
        self.t = 0.0
        self.hsign = np.tanh(la / (8 * DEG))
        self._trig = None
        self._trig_t = -1.0
        self.tcs = []
        self._tc_rng = np.random.RandomState(seed + 11)
        self.tc_img = np.zeros_like(self.qv)

    # sun and surface heating ----------------------------------------------------
    def coszen(self, lag_hours=0.0):
        g = self.g
        sub_lon = math.pi - 2 * math.pi * (self.t % 1.0) + lag_hours / 24.0 * 2 * math.pi
        d = self.decl
        return g.sin * math.sin(d) + g.cos * math.cos(d) * np.cos(g.LON - sub_lon)

    def step(self, dt=1.0 / 48.0):
        g, m = self.g, self.model
        m.step(dt)
        # vertical motion: adiabatic thickness change, sign-corrected per hemisphere, smoothed
        w = self.hsign * m.omega
        w = Grid.shapiro(Grid.shapiro(w))
        # normalise: typical synoptic ascent ~ 1 (running mean of the spread)
        sd = float(np.sqrt(np.mean(w[np.abs(g.LAT) > 20 * DEG] ** 2))) + 1e-9
        self.wsd = sd if getattr(self, 'wsd', 0.0) == 0.0 else 0.98 * self.wsd + 0.02 * sd
        w = w / (2.0 * self.wsd)
        self.w = w
        tau = m.psi1 - m.psi2
        T_anom = 90.0 * self.hsign * (tau - tau.mean(1, keepdims=True))     # warm / cold sectors
        T_air = self.T_clim + np.clip(T_anom, -15, 15)
        heat_land = np.maximum(self.coszen(-3.0), 0.0)            # afternoon peak (3 h lag)
        T_surf = np.where(self.land > 0.5, T_air + 11.0 * heat_land - 3.0, self.sst)
        # advection: vapour with the lower layer, frontal cloud with the mean flow,
        # convective anvils with the upper flow
        u1, v1, u2, v2 = m.u1, m.v1, m.u2, m.v2
        dl = g.departure(u2, v2, dt)
        qv = g.interp(self.qv, dl)
        cl = g.interp(self.cl, m.wm)
        wu = g.departure(u1, v1, dt)
        cc = g.interp(self.cc, wu)
        ci = g.interp(self.ci, wu)                 # cirrus rides the upper-level (jet) wind
        qs_air = qsat(T_air)
        rh = qv / qs_air
        # evaporation toward 85 % of saturation at the surface temperature
        qs_s = qsat(np.where(self.land > 0.5, T_surf, self.sst))
        ocean = 1 - self.land
        E = np.maximum(0.85 * qs_s - qv, 0.0) / 1.5 * (ocean + self.land * self.wet)
        # condensation: rising air, supersaturation, and convection in warm moist columns
        wpos = np.maximum(w, 0.0)
        extra = smoothstep(8 * DEG, 22 * DEG, np.abs(g.LAT))        # QG ascent is meaningless near the equator
        C_ls = (KW * wpos * smoothstep(0.45, 0.85, rh) * extra + np.maximum(rh - 1.0, 0.0) * 12.0) * qv
        instab = np.where(self.land > 0.5, smoothstep(25.0, 34.0, T_surf), smoothstep(25.5, 29.5, self.sst))
        # organised convection: clusters from a slowly evolving 3D noise (stand-in for
        # easterly waves and mesoscale organisation)
        if self._trig is None or self._trig_t + 0.125 <= self.t:
            from .luts import fbm_noise
            P = g.P * np.array([6.0, 6.0, 9.0]) + np.array([0.0, 0.0, 0.0])
            ph = self.t * 0.9
            self._trig = np.clip(0.5 + 1.6 * fbm_noise(np.concatenate([P[..., :2] + ph * np.array([-1.0, 0.3]), P[..., 2:]], -1), 3, seed=77), 0.0, 1.0)
            self._trig_t = self.t
        C_cv = KC * instab * np.maximum(rh - 0.55, 0.0) * qv * (0.04 + 2.4 * self._trig ** 2)
        C = C_ls + C_cv
        qv = qv + dt * (E - C)
        # cloud amount follows condensation relative to saturation: cold air holds little
        # water but its clouds are just as opaque
        cl = cl + dt * C_ls / (0.25 + qs_air)
        cc = cc + dt * C_cv / (0.25 + qs_air)
        # cirrus: thunderstorm anvils blown downstream, warm-conveyor-belt tops, and jet-stream
        # cirrus where fast upper winds and rising air meet; long-lived ice cloud
        jet = smoothstep(0.35, 0.75, np.sqrt(u1 * u1 + v1 * v1))
        ci = ci + dt * (0.9 * C_cv + 0.35 * C_ls + 0.6 * jet * wpos * qv) / (0.25 + qs_air)
        ci = ci * np.exp(-dt * (CI_DECAY + 1.5 * np.maximum(-w, 0.0)))
        # cloud lifetime: shorter in sinking and dry air
        wneg = np.maximum(-w, 0.0)
        # Hadley-cell descent dries the subtropics; hot deserts evaporate cloud
        hadley = np.exp(-((np.abs(g.LAT) - (23 * DEG + np.sign(g.LAT) * 0.0)) / (8 * DEG)) ** 2)
        desert = self.land * (1 - self.wet)
        cl = cl * np.exp(-dt * (LS_DECAY + 3.0 * wneg + 4.0 * np.maximum(0.6 - rh, 0.0)
                                + HADLEY * hadley + 4.0 * desert))
        cc = cc * np.exp(-dt * (CV_DECAY + 0.5 * HADLEY * hadley + 4.0 * desert))
        self.qv = np.clip(g.filter(qv), 0.0, 3.0)
        self.cl = np.clip(g.filter(cl), 0.0, None)
        self.cc = np.clip(g.filter(cc), 0.0, None)
        self.ci = np.clip(g.filter(ci), 0.0, None)
        # marine stratocumulus: cold water (vs the zonal mean) under sinking air, moist boundary layer
        cold = smoothstep(0.5, 4.0, self.sst_zonal - self.sst) * ocean
        sub = smoothstep(0.0, 0.8, wneg)
        hi = ocean * smoothstep(48 * DEG, 62 * DEG, np.abs(g.LAT)) * smoothstep(0.5, 0.9, rh) * 0.5
        self.sc = np.clip(cold * (0.35 + 0.65 * sub) + hi, 0.0, 1.0)
        self._cyclones(dt, T_air)
        self.t += dt
        return self

    # tropical cyclones -----------------------------------------------------------
    def _cyclones(self, dt, T_air):
        """Genesis over warm water, steering + beta drift, intensity toward an SST
        potential intensity, decay over land / cold water; imprint spiral bands,
        central dense overcast and eye into the convective cloud field.

        Procedural artwork (tier E): only self.cc changes. The storms do not
        modify the QG streamfunctions, PV, winds, temperature or vapour."""
        g, m = self.g, self.model
        rng = self._tc_rng
        # genesis: attempts ~1.7 per day, only over water warmer than 27.8 C
        if rng.rand() < dt / 0.6:
            for _ in range(60):
                la = rng.uniform(9, 22) * (1 if rng.rand() < 0.62 else -1)
                lo = rng.uniform(-180, 180)
                j = int((la + 90) / 180.0 * g.nlat)
                i = int((lo + 180) / 360.0 * g.nlon) % g.nlon
                if self.land[j, i] < 0.2 and self.sst[j, i] > 27.8:
                    self.tcs.append(dict(lat=math.radians(la), lon=math.radians(lo), I=0.15,
                                         ph=rng.uniform(0, 6.28)))
                    break
        keep = []
        for tc in self.tcs:
            j = int(np.clip((tc['lat'] + math.pi / 2) / g.dlat, 0, g.nlat - 1))
            i = int(((tc['lon'] + math.pi) / g.dlon)) % g.nlon
            # steering: deep-layer mean wind, plus beta drift (poleward, westward ~ 2-3 m/s)
            k = float(smoothstep(10 * DEG, 25 * DEG, abs(tc['lat'])))      # QG winds only off the equator
            u = k * 0.5 * (m.u1[j, i] + m.u2[j, i]) + (1 - k) * (-0.075) - 0.02
            v = k * 0.5 * (m.v1[j, i] + m.v2[j, i]) + 0.045 * math.copysign(1.0, tc['lat'])
            tc['lat'] += v * dt
            tc['lon'] = (tc['lon'] + u * dt / max(math.cos(tc['lat']), 0.2) + math.pi) % (2 * math.pi) - math.pi
            sst = self.sst[j, i]
            over_land = self.land[j, i] > 0.5
            pi_ = np.clip((sst - 25.5) / 2.5, 0.0, 1.0)            # potential intensity from SST
            if over_land or abs(tc['lat']) > math.radians(38):
                tc['I'] -= dt * 0.8 * tc['I']
            else:
                tc['I'] += dt * (pi_ - tc['I']) / 1.5
            tc['ph'] += dt * 6.0
            if tc['I'] > 0.05 and abs(tc['lat']) < math.radians(55):
                keep.append(tc)
        self.tcs = keep
        img = np.zeros_like(self.cc)
        for tc in self.tcs:
            C = np.array([math.cos(tc['lat']) * math.cos(tc['lon']), math.cos(tc['lat']) * math.sin(tc['lon']),
                          math.sin(tc['lat'])])
            R = math.radians(4.5 + 4.0 * tc['I'])
            d = np.arccos(np.clip(g.P @ C, -1, 1))
            near = d < 2.5 * R
            if not near.any():
                continue
            e = np.array([-math.sin(tc['lon']), math.cos(tc['lon']), 0.0])
            n = np.cross(C, e)
            x, y = g.P @ e, g.P @ n
            th = np.arctan2(y, x)
            r = d / R
            hemi = math.copysign(1.0, tc['lat'])
            arms = 0.5 + 0.5 * np.cos(2 * (th * hemi + 2.2 * np.log(r + 0.05)) + tc['ph'])
            band = np.exp(-((r - 0.9) / 0.9) ** 2) * arms ** 2
            cdo = np.exp(-(r / 0.42) ** 2)
            eye = np.exp(-(r / max(0.06, 0.12 - 0.06 * tc['I'])) ** 2) * smoothstep(0.45, 0.8, tc['I'])
            img = np.maximum(img, tc['I'] * (1.6 * cdo + 1.1 * band) * (1 - eye) * near)
        self.cc = np.maximum(self.cc, img)
        self.tc_img = img

    def frame(self):
        """(coverage, type, cirrus) in [0, 1], rows south -> north."""
        a_ls, a_cv = A_LS, A_CV
        L, Cv, S = a_ls * self.cl, a_cv * self.cc, 0.9 * self.sc
        tot = L + Cv + S + 1e-6
        if getattr(self, 'cov_range', None) is None:
            # fixed once (from the first frame after spin-up): ~35 % of the globe clear,
            # ~30 % fully overcast, as in satellite composites
            self.cov_range = (float(np.percentile(tot, CLEAR_PCT)), float(np.percentile(tot, OVERCAST_PCT)))
        lo, hi = self.cov_range
        cov = np.clip((tot - lo) / (hi - lo), 0.0, 1.0) ** 0.8
        typ = (0.42 * L + 1.0 * Cv + 0.08 * S) / tot
        if getattr(self, 'ci_scale', None) is None:
            self.ci_scale = float(np.percentile(self.ci, 96)) + 1e-6
        cir = smoothstep(0.25, 1.0, self.ci / self.ci_scale)        # ~a quarter of the globe has cirrus
        return cov, np.clip(typ, 0.02, 1.0), cir


# =================================================================== recording
FILM_DIR = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),
                                      'data', 'weather')


def frame_image(cov, typ, thick, size=(512, 256)):
    """Model frame (rows south -> north) -> RGB PIL image like luts.bake_weather
    (row 0 = north), upsampled with bicubic filtering."""
    from PIL import Image
    chans = []
    for a in (cov, typ, thick):
        im = Image.fromarray(np.float32(np.flipud(a))).resize(size, Image.BICUBIC)
        chans.append(np.clip(np.asarray(im, float), 0.0, 1.0))
    return Image.fromarray(np.uint8(np.round(np.stack(chans, -1) * 255)), 'RGB')


def record(out_dir=FILM_DIR, spin_days=30.0, loop_days=4.0, blend_hours=12.0, frame_hours=1.0,
           nlon=256, decl_deg=-9.0, size=(512, 256), quality=90, seed=1, log=print):
    """Spin the climate up, then write a seamless loop of hourly weather frames.

    Frame k is model time k * frame_hours from 00:00 UTC (the diurnal cycle of
    convection over land is tied to UTC, so playback picks frames by the sun).
    The first blend_hours of the loop are cross-faded with the hours just after
    its end, so the last frame flows into the first.
    """
    import json
    import os
    import time
    os.makedirs(out_dir, exist_ok=True)
    c = Climate(nlon, nlon // 2, decl_deg=decl_deg, seed=seed)
    t0 = time.time()
    steps_per_frame = int(round(frame_hours / 24.0 * 48))
    dt = frame_hours / 24.0 / steps_per_frame
    n = int(round(spin_days * 48))
    for i in range(n):
        c.step(1.0 / 48.0)
        if log and i % (48 * 5) == 0:
            log('spin-up day %d  (%.0f s)' % (i // 48, time.time() - t0))
    c.t = 0.0                                   # loop starts at 00:00 UTC
    c.cov_range = None
    c.ci_scale = None
    nloop = int(round(loop_days * 24 / frame_hours))
    nblend = int(round(blend_hours / frame_hours))
    frames = []
    for k in range(nloop + nblend):
        frames.append(c.frame())
        for _ in range(steps_per_frame):
            c.step(dt)
        if log and k % 24 == 0:
            log('frame %d / %d  (%.0f s)' % (k, nloop + nblend, time.time() - t0))
    for name in os.listdir(out_dir):
        if name.startswith('w') and name.endswith('.jpg'):
            os.remove(os.path.join(out_dir, name))
    for k in range(nloop):
        f = frames[k]
        if k < nblend:                          # start blends from the continuation of the end
            a = k / float(nblend)
            a = a * a * (3 - 2 * a)
            f = tuple(frames[k + nloop][i] * (1 - a) + frames[k][i] * a for i in range(3))
        frame_image(*f, size=size).save(os.path.join(out_dir, 'w%03d.jpg' % k), quality=quality)
    meta = dict(frames=nloop, frame_hours=frame_hours, loop_days=loop_days, decl_deg=decl_deg,
                channels=['coverage', 'type', 'cirrus'],
                grid=[nlon, nlon // 2], size=list(size), spin_days=spin_days,
                model='two-layer QG + moisture (orbitgfx.weather)', version=2)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    if log:
        log('wrote %d frames to %s in %.0f s' % (nloop, out_dir, time.time() - t0))
    return meta
