"""orbitgfx -- GPU Earth/orbit rendering for Pythonista `scene`.

    from orbitgfx import (GlobeView, NightMapOverlay, Camera, Gestures,
                          cached_earth_image, pil_texture, astro, draw2d, safety)

Package layout (everything lives inside this folder):

    orbitgfx/
      README.md, LICENSE
      astro, draw2d, earth, gestures, safety, shaders, views   <- the library
      ivp         NumPy-only RK45 solve_ivp (SciPy-compatible subset)
      coastlines  Natural Earth 1:110m coastlines, bundled in data/
      examples/   mission_control.py, globe_demo.py
      sandbox/    desktop emulation (Runner, render CLI, scene/ui shim)
      tests/      test_mission.py + baseline renders

On Pythonista the real `scene`/`ui` modules are used. On a desktop, where no
`scene` module exists, importing orbitgfx puts the sandbox shim on sys.path so
the same code imports and runs (shaders need moderngl, rendering needs skia).
"""
import os as _os
import sys as _sys

__version__ = '2.0.0a16'     # TUNING.md
PACKAGE_DIR = _os.path.dirname(_os.path.abspath(__file__))
SHIM_DIR = _os.path.join(PACKAGE_DIR, 'sandbox', 'shim')
EXAMPLES_DIR = _os.path.join(PACKAGE_DIR, 'examples')

try:                                    # Pythonista: the real thing
    import scene as _scene              # noqa: F401
    ON_DEVICE = 'shim' not in _os.path.dirname(getattr(_scene, '__file__', '') or '')
except ImportError:                     # desktop: fall back to the emulation
    _sys.path.insert(0, SHIM_DIR)
    ON_DEVICE = False

from . import astro, coastlines, draw2d, ivp, luts, safety, shaders  # noqa: E402
from .earth import cached_earth_image, earth_image, pil_texture  # noqa: E402
from .gestures import Gestures  # noqa: E402
from .shaders import (FEATURES, PALETTE, RAY_FEATURES, build_globe_shader,  # noqa: E402
                      build_ray_shader, lint_glsl)
from .views import QUALITY, Camera, GlobeView, NightMapOverlay  # noqa: E402

__all__ = ['astro', 'coastlines', 'ivp', 'draw2d', 'safety', 'shaders', 'earth_image', 'cached_earth_image',
           'pil_texture', 'Gestures', 'FEATURES', 'PALETTE', 'build_globe_shader',
           'lint_glsl', 'RAY_FEATURES', 'build_ray_shader', 'QUALITY', 'luts', 'Camera', 'GlobeView', 'NightMapOverlay', 'ON_DEVICE',
           'PACKAGE_DIR', 'EXAMPLES_DIR']
