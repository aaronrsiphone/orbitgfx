"""orbitgfx device smoke run (Pythonista): run this before examples/device_check.py.

About a minute, hands off. Parses every orbitgfx file, imports every module meant
for the device, compiles every shader variant through Apple's compiler, and runs
every stage of the full check once with no timing (capture, GPU maths, screens,
bench, refraction, weather film, memory, heat, zip, share), so anything that would
crash the 8-minute run crashes here first. Tap DONE to share
orbitgfx_device_smoke.zip; if Pythonista quits, run it again and the report says
which step it skipped.
"""
import os
import sys

try:
    import orbitgfx  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import orbitgfx  # noqa: F401
import scene
from orbitgfx.examples import device_check

device_check.SMOKE = True
DeviceCheck = device_check.DeviceCheck

if __name__ == '__main__':
    scene.run(DeviceCheck(), orientation=getattr(scene, 'PORTRAIT', 0), show_fps=False)
