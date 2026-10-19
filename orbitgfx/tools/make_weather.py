"""Precompute the weather film that GlobeView(weather='film') plays back.

    python -m orbitgfx.tools.make_weather [--days 4] [--spin 30] [--grid 256] [--decl -9]

Runs the two-layer QG + moisture model in orbitgfx/weather.py (about 5 minutes
on a laptop at the default 1.4 degree grid; much longer in Pythonista) and
writes orbitgfx/data/weather/w###.jpg (hourly, 512 x 256: coverage, type,
thickness) and meta.json. Bump earth.TEXTURE_VERSION is not needed: the frames
are read directly, not cached.
"""
import argparse
import os
import sys

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from orbitgfx import weather  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--days', type=float, default=4.0, help='loop length in days')
    ap.add_argument('--spin', type=float, default=30.0, help='spin-up days')
    ap.add_argument('--grid', type=int, default=256, help='longitude points (latitude = half)')
    ap.add_argument('--decl', type=float, default=-9.0, help='sun declination (season), degrees')
    ap.add_argument('--out', default=weather.FILM_DIR)
    a = ap.parse_args(argv)
    weather.record(a.out, spin_days=a.spin, loop_days=a.days, nlon=a.grid, decl_deg=a.decl)


if __name__ == '__main__':
    main()
