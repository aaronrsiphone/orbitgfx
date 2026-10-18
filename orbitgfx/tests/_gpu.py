"""Explicit GPU gate for the desktop tests.

The sandbox only lints a shader when there is no GL context (no moderngl, no
display / EGL): a test that compiles or renders must then be SKIPPED, never
reported as passed. Set ORBITGFX_REQUIRE_GPU=1 (release CI) to make those
skips failures instead.

Even a pass here is the desktop OpenGL 120 dialect on Mesa/llvmpipe, not
Pythonista's shader compiler: on-device compilation is a separate gate.
"""
import functools
import os
import unittest

import scene


def require(what='GPU'):
    ok, why = scene.gl_status()
    if ok and what == 'render' and getattr(scene, 'skia', None) is None:
        ok, why = False, 'skia-python not installed'
    if ok:
        return
    msg = '%s check needs the desktop GL sandbox: %s' % (what, why)
    if os.environ.get('ORBITGFX_REQUIRE_GPU'):
        raise AssertionError(msg + ' (ORBITGFX_REQUIRE_GPU is set)')
    raise unittest.SkipTest(msg)


def gpu(fn=None, what='GPU'):
    """Decorator: skip (or, with ORBITGFX_REQUIRE_GPU, fail) without a GL context."""
    if fn is None:
        return lambda f: gpu(f, what)

    @functools.wraps(fn)
    def run(*a, **kw):
        require(what)
        return fn(*a, **kw)
    return run


def run_all(namespace):
    """`python -m orbitgfx.tests.<module>`: PASS / SKIP per test."""
    for name, fn in sorted(namespace.items()):
        if name.startswith('test_'):
            try:
                res = fn()
            except unittest.SkipTest as e:
                print('SKIP', name, '--', e)
            else:
                print('PASS', name, res if res else '')
