"""Classic (immediate-mode) drawing helpers for Scene.draw().

These draw UNDER every node, including shader views.
"""
import math

import numpy as np
import scene


def rgba(rgb, a=1.0):
    return (rgb[0], rgb[1], rgb[2], a)


def mix(a, b, w):
    return tuple(a[i] * (1.0 - w) + b[i] * w for i in range(3))


def line(x1, y1, x2, y2, color, weight=1.0):
    scene.stroke(*color)
    scene.stroke_weight(weight)
    scene.line(float(x1), float(y1), float(x2), float(y2))


def panel(bounds, fill_rgb, border_rgb, border_alpha=0.9):
    x, y, w, h = bounds
    scene.fill(*rgba(fill_rgb)); scene.no_stroke(); scene.rect(x, y, w, h)
    scene.no_fill(); scene.stroke(*rgba(border_rgb, border_alpha)); scene.stroke_weight(1)
    scene.rect(x, y, w, h)


def glow_dot(x, y, rgb, rad=3.0):
    """Layered marker: soft halo, ring, core, white centre."""
    scene.no_stroke()
    scene.fill(*rgba(rgb, .12)); scene.ellipse(x - 4 * rad, y - 4 * rad, 8 * rad, 8 * rad)
    scene.fill(*rgba(rgb, .35)); scene.ellipse(x - 2 * rad, y - 2 * rad, 4 * rad, 4 * rad)
    scene.fill(*rgba(rgb, 1)); scene.ellipse(x - rad, y - rad, 2 * rad, 2 * rad)
    scene.fill(1, 1, 1, 1); scene.ellipse(x - 1.4, y - 1.4, 2.8, 2.8)


def sun_glyph(px, py, rgb, alpha=1.0, core=4.0, ray0=6.0, ray1=9.0, weight=1.2):
    for k in range(8):
        a = k * math.pi / 4
        line(px + ray0 * math.cos(a), py + ray0 * math.sin(a),
             px + ray1 * math.cos(a), py + ray1 * math.sin(a), rgba(rgb, alpha), weight)
    scene.no_stroke(); scene.fill(*rgba(rgb, alpha))
    scene.ellipse(px - core, py - core, 2 * core, 2 * core)


class MapRect(object):
    """Equirectangular projection into a screen rectangle (keep it 2:1)."""

    def __init__(self, rect=(0, 0, 360, 180)):
        self.rect = rect

    def xy(self, lon, lat):
        x, y, w, h = self.rect
        return (x + (np.asarray(lon) + 180.0) / 360.0 * w,
                y + (np.asarray(lat) + 90.0) / 180.0 * h)

    def segments(self, seg, color, weight):
        """seg: (N, 4) lon0, lat0, lon1, lat1 rows, already seam-filtered."""
        x0, y0 = self.xy(seg[:, 0], seg[:, 1])
        x1, y1 = self.xy(seg[:, 2], seg[:, 3])
        scene.stroke(*color); scene.stroke_weight(weight)
        for j in range(len(seg)):
            scene.line(float(x0[j]), float(y0[j]), float(x1[j]), float(y1[j]))

    def polyline(self, lon, lat, color, weight):
        """Draw a lon/lat polyline, breaking it at the date line."""
        X, Y = self.xy(lon, lat)
        scene.stroke(*color); scene.stroke_weight(weight)
        for j in range(len(lon) - 1):
            if abs(lon[j + 1] - lon[j]) < 180:
                scene.line(float(X[j]), float(Y[j]), float(X[j + 1]), float(Y[j + 1]))

    def graticule(self, rgb, step=30, alpha=0.6, equator_alpha=0.85, other_alpha=0.5):
        x, y, w, h = self.rect
        for lo in range(-180 + step, 180, step):
            px, _ = self.xy(lo, 0)
            line(px, y, px, y + h, rgba(rgb, alpha), .7)
        for la in range(-90 + step, 90, step):
            _, py = self.xy(0, la)
            line(x, py, x + w, py, rgba(rgb, equator_alpha if la == 0 else other_alpha), .8)


def seam_segments(lonlat_paths):
    """[(lon, lat) arrays] -> (N, 4) segment array without date-line jumps."""
    out = []
    for lon, lat in lonlat_paths:
        s = np.column_stack((lon[:-1], lat[:-1], lon[1:], lat[1:]))
        out.append(s[np.abs(np.diff(lon)) < 180])
    return np.vstack(out) if out else np.zeros((0, 4))
