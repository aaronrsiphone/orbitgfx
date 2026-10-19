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


def gpu_eval(lib, expr, x0, x1, n=64, uniforms=None, defines=()):
    """Evaluate a GLSL float expression of `x` (linear from x0 to x1 over n samples)
    on the sandbox GPU at 32-bit float precision. lib: GLSL functions and uniform
    declarations it needs. Returns (xs, values)."""
    import moderngl
    import numpy as np
    ctx = scene._gl()
    head = '#version 120\n' + ''.join('#define %s\n' % d for d in defines)
    fs = (head + lib + '\nuniform float u_x0;\nuniform float u_x1;\n'
          'void main() {\n'
          '    float x = mix(u_x0, u_x1, (gl_FragCoord.x - 0.5) / %.1f);\n'
          '    gl_FragColor = vec4(%s, 0.0, 0.0, 1.0);\n}\n' % (max(n - 1, 1), expr))
    vs = ('#version 120\nattribute vec2 in_pos;\n'
          'void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }')
    prog = ctx.program(vertex_shader=vs, fragment_shader=fs)
    for k, v in dict(uniforms or {}, u_x0=float(x0), u_x1=float(x1)).items():
        if k in prog:
            prog[k].value = v
    tex = ctx.texture((n, 1), 4, dtype='f4')
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')
    vao.render(moderngl.TRIANGLE_STRIP)
    out = np.frombuffer(fbo.read(components=4, dtype='f4'), np.float32).reshape(n, 4)[:, 0]
    for o in (vao, vbo, fbo, tex, prog):
        o.release()
    return np.linspace(x0, x1, n), out.astype(float)


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
