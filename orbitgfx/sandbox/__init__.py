"""Desktop emulation of Pythonista's scene/ui for headless testing and screenshots.

    from orbitgfx.sandbox import Runner
    python -m orbitgfx.sandbox.render <app.py> <SceneClass> -o shot.png

Never imported by orbitgfx itself, so it costs nothing on the device.
"""
from .runner import Runner, VirtualClock, load_scene_class  # noqa: F401
