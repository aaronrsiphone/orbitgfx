"""Natural Earth 1:110m coastlines (bundled; offline from the first run).

Source: https://github.com/nvkelso/natural-earth-vector
Data license: public domain.

The GeoJSON ships in orbitgfx/data/. If it is ever missing, it is downloaded
once into that folder.
"""
import json
import os
from urllib.request import urlopen

URL = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
       'master/geojson/ne_110m_coastline.geojson')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                     'ne_110m_coastline.geojson')


def load_coastlines(path=CACHE):
    """Return a list of Nx2 NumPy arrays [longitude, latitude] (degrees)."""
    import numpy as np
    if not os.path.exists(path):
        with urlopen(URL, timeout=25) as response:
            content = response.read()
        data = json.loads(content.decode('utf-8'))  # validate before caching
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as target:
            target.write(content)
    else:
        with open(path, 'r', encoding='utf-8') as source:
            data = json.load(source)

    if data.get('type') != 'FeatureCollection':
        raise ValueError('Expected a GeoJSON FeatureCollection')
    lines = []
    for feature in data['features']:
        geom = feature.get('geometry') or {}
        kind, coords = geom.get('type'), geom.get('coordinates', [])
        if kind == 'LineString':
            paths = [coords]
        elif kind == 'MultiLineString':
            paths = coords
        else:
            continue
        for line in paths:
            arr = np.asarray(line, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                lines.append(arr[:, :2].copy())
    return lines


def coastlines_xyz(radius=6378.137):
    """Return 3D Earth-centered curves; rotate by sidereal angle when drawing."""
    import numpy as np
    curves = []
    for line in load_coastlines():
        lon, lat = np.radians(line[:, 0]), np.radians(line[:, 1])
        clat = np.cos(lat)
        curves.append(np.column_stack((radius * clat * np.cos(lon),
                                       radius * clat * np.sin(lon),
                                       radius * np.sin(lat))))
    return curves
