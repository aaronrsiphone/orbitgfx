"""Numerical verification that needs no GPU: camera basis, WGS84 stations, J2 force,
RK45, QG inversion (the independent V&V probes of the 2.0.0a16 review, as tests).

    python -m orbitgfx.tests.test_geometry
These verify the implementation against its own equations and independent
references; they are not a measure of real-world accuracy (see VALIDATION.md).
"""
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(PKG))
from orbitgfx.sandbox.runner import Runner  # noqa: E402,F401  (installs the shim)

from orbitgfx.astro import (MU, RE, WGS84_E2, elevation, geodetic_to_ecef,  # noqa: E402
                            geodetic_up)
from orbitgfx.tests._gpu import run_all  # noqa: E402
from orbitgfx.views import Camera, look_at  # noqa: E402


def _orthonormal(M):
    assert np.isfinite(M).all(), M
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12), M
    assert abs(np.linalg.det(M) - 1.0) < 1e-12, M         # right-handed


# ------------------------------------------------------------------- camera
def test_look_at_basis():
    """Finite, orthonormal, right-handed for nadir, zenith, near-radial and oblique views;
    row 2 always points back at the viewer."""
    cases = {
        'nadir': ((1.1, 0.0, 0.0), (0.0, 0.0, 0.0)),
        'zenith': ((1.1, 0.0, 0.0), (2.0, 0.0, 0.0)),
        'nadir_z': ((0.0, 0.0, 1.5), (0.0, 0.0, 0.0)),
        'nadir_diag': ((0.7, -0.7, 0.3), (0.0, 0.0, 0.0)),
        'near_radial': ((1.1, 0.0, 0.0), (0.0, 1e-9, 0.0)),
        'oblique': ((1.0, 0.2, 0.3), (0.2, 0.9, -0.1)),
        'eye_at_centre': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    }
    for name, (eye, look) in cases.items():
        M = look_at(eye, look)
        _orthonormal(M)
        f = np.subtract(eye, look)
        assert np.allclose(M[2], f / np.linalg.norm(f)), name
    # oblique views keep the horizon level: screen-right is horizontal at the eye
    e = np.array((1.0, 0.2, 0.3))
    assert abs(look_at(e, (0.2, 0.9, -0.1))[0] @ e) < 1e-12
    cam = Camera()
    cam.eye, cam.look = np.array((1.1, 0.0, 0.0)), np.zeros(3)
    with np.errstate(all='raise'):                          # no silent NaN on the way
        _orthonormal(cam.matrix())


def test_look_at_invalid():
    for eye, look in (((1.1, 0, 0), None), ((1.1, 0, 0), (1.1, 0, 0)),
                      ((float('nan'), 0, 0), (0, 0, 0))):
        try:
            look_at(eye, look)
        except ValueError:
            continue
        raise AssertionError('accepted eye=%r look=%r' % (eye, look))


def test_step_keeps_app_look_at_camera():
    """step() only undoes eye/look that astronaut mode placed, not the app's own."""
    cam = Camera()
    cam.eye, cam.look = np.array((1.2, 0.0, 0.0)), np.zeros(3)
    cam.step(1 / 60.0)
    assert cam.eye is not None and cam.look is not None
    sat = np.array((RE + 500.0, 0.0, 0.0))
    vel = np.array((0.0, 7.6, 0.0))
    cam.follow = cam.astro = True
    cam.step(1 / 60.0, sat, vel)
    assert cam.eye is not None
    cam.astro = False                                     # TRACK: astronaut eye released
    cam.step(1 / 60.0, sat, vel)
    assert cam.eye is None and cam.look is None and cam.fov_deg is None


# ------------------------------------------------------------------- WGS84
def test_geodetic_fixtures():
    a = RE
    b = a * math.sqrt(1.0 - WGS84_E2)
    assert np.allclose(geodetic_to_ecef(0.0, 0.0), (a, 0.0, 0.0), atol=1e-9)
    assert np.allclose(geodetic_to_ecef(0.0, 90.0, 1.0), (0.0, a + 1.0, 0.0), atol=1e-9)
    assert np.allclose(geodetic_to_ecef(90.0, 0.0), (0.0, 0.0, b), atol=1e-6)
    assert np.allclose(geodetic_to_ecef(-90.0, 0.0, 2.0), (0.0, 0.0, -b - 2.0), atol=1e-6)
    assert abs(b - 6356.752314245) < 1e-6                 # WGS84 semi-minor axis
    for lat, lon in ((45.0, 10.0), (67.86, 20.96), (-35.4, 148.98)):
        p = geodetic_to_ecef(lat, lon)
        x2 = p[0] ** 2 + p[1] ** 2
        assert abs(x2 / a ** 2 + p[2] ** 2 / b ** 2 - 1.0) < 1e-12   # on the ellipsoid
        up = geodetic_up(lat, lon)
        # the normal is the gradient of the ellipsoid, and height moves along it
        g = np.array((p[0] / a ** 2, p[1] / a ** 2, p[2] / b ** 2))
        assert np.allclose(up, g / np.linalg.norm(g), atol=1e-12)
        assert np.allclose(geodetic_to_ecef(lat, lon, 3.0) - p, 3.0 * up, atol=1e-9)
        # geodetic latitude exceeds geocentric away from the equator
        assert abs(math.degrees(math.atan2(p[2], math.sqrt(x2)))) < abs(lat)


def test_elevation_horizon():
    lat, lon = 37.94, -75.46
    s, up = geodetic_to_ecef(lat, lon), geodetic_up(lat, lon)
    east = np.cross((0.0, 0.0, 1.0), up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    for el in (0.0, 5.0, 45.0, 90.0, -3.0):
        e = math.radians(el)
        rho = 900.0 * (math.cos(e) * (0.6 * east + 0.8 * north) + math.sin(e) * up)
        assert abs(float(elevation(rho, up)) - el) < 1e-9, el
    assert elevation(np.stack([up, -up]), up).tolist() == [90.0, -90.0]


# ------------------------------------------------------------------- physics
def _mission():
    from orbitgfx.examples import mission_control as mc
    return mc


def test_j2_force_is_potential_gradient():
    mc = _mission()
    rng = np.random.default_rng(832)
    worst = 0.0
    for _ in range(20):
        x = rng.normal(size=3)
        x *= (RE + rng.uniform(300, 1400)) / np.linalg.norm(x)
        h = 0.1
        g = np.array([(mc.energy(x + h * np.eye(3)[i], np.zeros(3))
                       - mc.energy(x - h * np.eye(3)[i], np.zeros(3))) / (2 * h) for i in range(3)])
        acc = mc.body(0, np.r_[x, np.zeros(3)])[3:]
        worst = max(worst, np.linalg.norm(acc + g) / np.linalg.norm(g))
    assert worst < 2e-8, worst
    return '%.2g' % worst


def test_rk45_against_reference():
    """Same J2 model, two integrators: verifies numerics, not orbit accuracy."""
    from orbitgfx.ivp import solve_ivp
    mc = _mission()
    a, inc = RE + 500.0, math.radians(51.6)
    y0 = np.array([a, 0, 0, 0, math.sqrt(MU / a) * math.cos(inc), math.sqrt(MU / a) * math.sin(inc)])
    period = 2 * math.pi * math.sqrt(a ** 3 / MU)
    ts = np.linspace(0.0, 5 * period, 1001)
    out = solve_ivp(mc.body, (0.0, ts[-1]), y0, t_eval=ts, rtol=1e-10, atol=1e-12)
    assert out.success
    E = mc.energy(out.y[:3].T, out.y[3:].T)
    drift = float(np.max(np.abs((E - E[0]) / E[0])))
    assert drift < 1e-8, drift
    back = solve_ivp(mc.body, (ts[-1], 0.0), out.y[:, -1], rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(back.y[:3, -1] - y0[:3]) < 1e-3     # km, forward then back
    rep = {'energy_drift': '%.2g' % drift}
    try:
        from scipy.integrate import solve_ivp as reference
    except ImportError:
        return rep
    ref = reference(mc.body, (0.0, ts[-1]), y0, t_eval=ts, rtol=1e-11, atol=1e-13)
    err = float(np.max(np.linalg.norm(out.y[:3] - ref.y[:3], axis=0)))
    assert err < 5e-4, err                                      # km over 5 orbits
    rep['vs_scipy_km'] = '%.2g' % err
    return rep


def test_qg_inversion_round_trip():
    from orbitgfx.weather import Grid, Model
    grid = Grid(nlon=72, nlat=36)
    m = Model(grid, seed=13)
    q1, q2 = m.pv(m.psi1, m.psi2)
    m.q1, m.q2 = q1.copy(), q2.copy()
    m.invert()
    p1, p2 = m.pv(m.psi1, m.psi2)
    res = max(np.linalg.norm(p1 - q1) / np.linalg.norm(q1), np.linalg.norm(p2 - q2) / np.linalg.norm(q2))
    assert res < 1e-10, res
    ones = np.ones((grid.nlat, grid.nlon))
    w = grid.departure(ones * 0.07, np.zeros_like(ones), 0.1)
    assert np.allclose(grid.interp(ones, w), 1.0, atol=1e-13)   # constant tracer kept
    return '%.2g' % res


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    run_all(globals())
