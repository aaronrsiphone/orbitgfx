"""Command line: render a Pythonista scene to PNG on the desktop.

    python -m orbitgfx.sandbox.render orbitgfx/examples/mission_control.py Mission -o frame.png \
        --size 390x844 --seconds 1.5 --unix 1792188000 \
        --exec "s.met0 = s.met = 4500; s.speed_idx = 0"

--exec runs Python with `s` = the scene and `r` = the Runner, after setup and
before advancing; repeat it to queue several statements.
"""
import argparse
import os
import sys

if __package__ in (None, ''):          # also runnable as a plain script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from orbitgfx.sandbox.runner import Runner  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('app')
    ap.add_argument('cls', nargs='?', default='Mission')
    ap.add_argument('-o', '--out', default='frame.png')
    ap.add_argument('--size', default='390x844')
    ap.add_argument('--scale', type=float, default=3.0)
    ap.add_argument('--seconds', type=float, default=0.5, help='frames to run before the shot')
    ap.add_argument('--unix', type=float, default=None, help='virtual wall-clock start (unix s)')
    ap.add_argument('--exec', dest='stmts', action='append', default=[])
    a = ap.parse_args(argv)
    w, h = (float(v) for v in a.size.lower().split('x'))
    with Runner(a.app, a.cls, size=(w, h), scale=a.scale, unix_time=a.unix) as r:
        for st in a.stmts:
            exec(st, {'s': r.scene, 'r': r})
        r.advance(a.seconds)
        r.render(a.out)
        print('wrote %s  (%d frames, %d classic draw calls in last frame)'
              % (a.out, r.frames, __import__('scene').stats['calls']))


if __name__ == '__main__':
    main()
