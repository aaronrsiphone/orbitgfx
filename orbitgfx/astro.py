"""Frames, time and geometry helpers (NumPy, vectorised; scalars work too).

Conventions: ECI is mean-equator inertial, ECEF = Rz(-theta) ECI where theta
is the Earth rotation angle (GMST). Distances in km unless noted.

Fidelity: a low-precision demonstration model, for visualisation. GMST is the
mean sidereal angle (no UT1, precession or nutation), the Sun is the Almanac
low-precision series, the eclipse is a cylinder, and ground tracks, footprints
and the map use a sphere. Only ground stations use the WGS84 ellipsoid
(geodetic_to_ecef), because elevation masks are sensitive to it.
"""
import math
import time

import numpy as np

MU = 398600.4418            # km^3 / s^2
RE = 6378.137               # km
J2 = 1.08262668e-3
SIDEREAL_DAY = 86164.0905   # s
OMEGA_E = 2.0 * math.pi / SIDEREAL_DAY
C_KM_S = 299792.458
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def julian_date(unix=None):
    return (time.time() if unix is None else unix) / 86400.0 + 2440587.5


def gmst(jd):
    """Greenwich mean sidereal angle (rad)."""
    return math.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)


def earth_angle(theta0, t):
    """Earth rotation angle t seconds after an epoch whose GMST is theta0."""
    return theta0 + OMEGA_E * t


def sun_dir(jd):
    """Unit Sun vector in ECI (Astronomical Almanac low-precision, ~0.01 deg)."""
    n = np.asarray(jd, float) - 2451545.0
    L = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    return np.stack((np.cos(lam), np.cos(eps) * np.sin(lam),
                     np.sin(eps) * np.sin(lam)), axis=-1)


def eci_to_ecef(p, theta):
    c, s = np.cos(theta), np.sin(theta)
    x, y = p[..., 0], p[..., 1]
    return np.stack((c * x + s * y, -s * x + c * y, p[..., 2]), axis=-1)


def ecef_to_eci(p, theta):
    c, s = np.cos(theta), np.sin(theta)
    x, y = p[..., 0], p[..., 1]
    return np.stack((c * x - s * y, s * x + c * y, p[..., 2]), axis=-1)


def latlon(p):
    """(lat, lon) in degrees from Cartesian (...,3)."""
    lat = np.degrees(np.arctan2(p[..., 2], np.hypot(p[..., 0], p[..., 1])))
    lon = np.degrees(np.arctan2(p[..., 1], p[..., 0]))
    return lat, lon


def llxyz(lon, lat, r=RE):
    """Point on a sphere of radius r (spherical lat/lon; see geodetic_to_ecef)."""
    lo, la = math.radians(lon), math.radians(lat)
    return (r * math.cos(la) * math.cos(lo), r * math.cos(la) * math.sin(lo),
            r * math.sin(la))


def geodetic_to_ecef(lat, lon, alt_km=0.0):
    """WGS84 geodetic latitude/longitude (deg) and ellipsoidal height (km) to ECEF km."""
    la, lo = math.radians(lat), math.radians(lon)
    n = RE / math.sqrt(1.0 - WGS84_E2 * math.sin(la) ** 2)   # prime vertical radius
    return np.array(((n + alt_km) * math.cos(la) * math.cos(lo),
                     (n + alt_km) * math.cos(la) * math.sin(lo),
                     (n * (1.0 - WGS84_E2) + alt_km) * math.sin(la)))


def geodetic_up(lat, lon):
    """Ellipsoid normal (local vertical) at a geodetic latitude/longitude (deg), ECEF."""
    la, lo = math.radians(lat), math.radians(lon)
    return np.array((math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)))


def elevation(rho, up):
    """Elevation (deg) of line-of-sight vector(s) rho (...,3) above the plane normal to up."""
    rho = np.asarray(rho, float)
    return np.degrees(np.arcsin(np.clip(rho @ up / np.linalg.norm(rho, axis=-1), -1.0, 1.0)))


def shadow_margin(p, s):
    """Cylindrical Earth shadow: negative inside the umbra cylinder (km)."""
    d = np.asarray(np.sum(p * s, axis=-1))
    perp = np.linalg.norm(p - d[..., None] * s, axis=-1)
    return np.where(d < 0, perp - RE, RE)


def small_circle(lat, lon, ang, n=72):
    """(lon, lat) degrees of a small circle of angular radius ang (rad)."""
    la1, lo1 = math.radians(lat), math.radians(lon)
    b = np.linspace(0.0, 2 * math.pi, n + 1)
    la2 = np.arcsin(math.sin(la1) * math.cos(ang) + math.cos(la1) * math.sin(ang) * np.cos(b))
    lo2 = lo1 + np.arctan2(np.sin(b) * math.sin(ang) * math.cos(la1),
                           math.cos(ang) - math.sin(la1) * np.sin(la2))
    return (np.degrees(lo2) + 180.0) % 360.0 - 180.0, np.degrees(la2)


def horizon_angle(r_km, mask_deg=0.0):
    """Earth central angle (rad) visible from radius r above an elevation mask
    (spherical Earth: for drawing footprints, not for contact timing)."""
    em = math.radians(mask_deg)
    return math.acos(RE / r_km * math.cos(em)) - em


def windows(ts, f, level):
    """Intervals where f > level, edges refined by linear interpolation."""
    above = f > level
    out, start = [], (float(ts[0]) if above[0] else None)
    for i in np.nonzero(above[1:] != above[:-1])[0]:
        tc = float(ts[i] + (level - f[i]) / (f[i + 1] - f[i]) * (ts[i + 1] - ts[i]))
        if above[i + 1]:
            start = tc
        elif start is not None:
            out.append((start, tc))
            start = None
    if start is not None:
        out.append((start, float(ts[-1])))
    return out


def hermite(ts, rr, vv, t):
    """Cubic Hermite interpolation of a sampled trajectory; scalar or array t."""
    t = np.clip(np.asarray(t, dtype=float), ts[0], ts[-1])
    i = np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2)
    h = (ts[i + 1] - ts[i])[..., None]
    u = ((t - ts[i])[..., None]) / h
    a, b, va, vb = rr[i], rr[i + 1], vv[i], vv[i + 1]
    u2, u3 = u * u, u * u * u
    p = ((2 * u3 - 3 * u2 + 1) * a + (u3 - 2 * u2 + u) * h * va
         + (-2 * u3 + 3 * u2) * b + (u3 - u2) * h * vb)
    v = ((6 * u2 - 6 * u) * (a - b) / h + (3 * u2 - 4 * u + 1) * va
         + (3 * u2 - 2 * u) * vb)
    return p, v
