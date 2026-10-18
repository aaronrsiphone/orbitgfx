"""orbitgfx 2.0, phase 0: device spike.

Run this on the iPhone in Pythonista (any folder works). It takes ~6 minutes,
then copies a results block to the clipboard and saves spike_results.txt next
to this script. Paste that block back into the chat.

Before running: Low Power Mode OFF, phone charged or plugged in, other apps
closed, don't touch the screen.

If Pythonista crashes, open spike_log.txt (next to this script) and send its
last ~30 lines: every step is written to disk before it runs, and native
crashes (SIGSEGV/SIGABRT) dump a Python traceback into the same file.
Results gathered so far are saved to spike_results.txt after every stage.

What it measures (plan, "Work plan" phase 0):
  (a) frame time vs shader load: for each kernel, the largest per-pixel loop
      count that still holds 60 fps and 30 fps over the full globe panel
  (b) whether an EffectNode or render_to_texture gives a reduced-resolution pass
  (c) whether several set_uniform textures work (up to 8 samplers)
  (d) highp float, trig precision, loops with uniform break, derivatives,
      gl_FragCoord, 8-bit texture exactness and filtering precision
  plus the legacy globe's frame time and headroom, and NumPy bake speed.
"""
import io
import math
import os
import sys
import time
import faulthandler
import traceback

import numpy as np
from PIL import Image

# Pixel readback through ui.View.draw_snapshot() crashed Pythonista natively on an
# iPhone 16 Pro Max (v1 run, 2026-10-16), so it is off. Probe results come from the
# screenshot of that run instead.
READBACK = False
SCREENSHOT_PROBES = {k: 'GREEN' for k in (
    'node', 'shader', 'highp_eps', 'highp_range', 'sin_large', 'loop_break', 'early_return',
    'local_array', 'fragcoord', 'deriv_ext', 'u_scale', 'pow_neg', 'tex4', 'tex8',
    'exact_nearest', 'exact_linear', 'lerp', 'rgb_png', 'rgba_png', 'k_alu', 'k_alu_div',
    'k_fetch', 'k_cloud_empty', 'k_cloud_full')}
SCREENSHOT_PROBES['deriv'] = 'COMPILE_FAIL(shows texture)'
SPIKE_VERSION = 2
TOP_INSET = 64           # points kept clear for the Dynamic Island / status bar

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, 'spike_log.txt')
_LOG = None
_T0 = time.time()


def crumb(msg):
    """Append one line to spike_log.txt and force it to disk before the next step runs."""
    global _LOG
    try:
        if _LOG is None:
            _LOG = open(LOG_PATH, 'w', buffering=1)
            faulthandler.enable(file=_LOG, all_threads=True)
            _LOG.write('spike v%d log started %s\n' % (SPIKE_VERSION, time.ctime()))
        _LOG.write('%8.2f  %s\n' % (time.time() - _T0, msg))
        _LOG.flush()
        os.fsync(_LOG.fileno())
    except Exception:
        pass
MAXN = 2048              # constant loop bound in the load kernels
T60, T30 = 17.5, 34.5    # trimmed-mean frame time (ms) that counts as holding 60 / 30 fps

try:
    import orbitgfx  # noqa: F401
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
    try:
        import orbitgfx  # noqa: F401
    except ImportError:
        orbitgfx = None
import scene
import ui

ON_DEVICE = 'shim' not in os.path.dirname(getattr(scene, '__file__', '') or 'x/shim')


# ======================================================================= GLSL
HEAD = '''precision highp float;
varying vec2 v_tex_coord;
uniform sampler2D u_texture;
uniform float u_time;
uniform vec2 u_sprite_size;
uniform float u_scale;
uniform float u_n;
uniform float u_probe;
'''
GREEN = 'vec4(0.0, 1.0, 0.0, 1.0)'
RED = 'vec4(1.0, 0.0, 0.0, 1.0)'
YELLOW = 'vec4(1.0, 1.0, 0.0, 1.0)'

# one "atmosphere sample" worth of ALU (5 exp, length, no texture fetch)
ALU_STEP = '''
        float h_ = max(length(p_) - 0.5, 0.0);
        float rr_ = exp(-h_ * 8.0);
        float rm_ = exp(-h_ * 60.0);
        od_ += rr_ + rm_;
        vec3 T_ = exp(-od_ * vec3(0.0058, 0.0135, 0.0331));
        L_ += T_ * (rr_ * vec3(0.3, 0.5, 0.9) + rm_ * 0.2);
        p_ += vec3(0.011, 0.007, -0.013);
'''

K_ALU = HEAD + '''
void main() {
    vec3 p_ = vec3(v_tex_coord * 2.0 - 1.0, 0.3 + 0.1 * fract(u_time));
    float od_ = 0.0;
    vec3 L_ = vec3(0.0);
    for (int i = 0; i < %(MAXN)d; i++) {
        if (float(i) >= u_n) break;
%(STEP)s
    }
    vec4 c = vec4(L_ * 1e-4, 1.0);
%(EXTRA)s
    gl_FragColor = mix(c, %(GREEN)s, u_probe);
}
'''

# same step, but each pixel runs a random trip count in [0.5n, n] (mean 0.75n)
K_ALU_DIV = HEAD + '''
float hash2(vec2 q) { return fract(sin(dot(q, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
    vec3 p_ = vec3(v_tex_coord * 2.0 - 1.0, 0.3 + 0.1 * fract(u_time));
    float od_ = 0.0;
    vec3 L_ = vec3(0.0);
    vec2 px = floor(v_tex_coord * u_sprite_size * u_scale);
    float m = u_n * (0.5 + 0.5 * hash2(px));
    for (int i = 0; i < %(MAXN)d; i++) {
        if (float(i) >= m) break;
%(STEP)s
    }
    gl_FragColor = mix(vec4(L_ * 1e-4, 1.0), %(GREEN)s, u_probe);
}
'''

# dependent, spatially coherent texture fetches from a set_uniform texture
K_FETCH = HEAD + '''
uniform sampler2D u_tx;
void main() {
    vec2 uv = v_tex_coord * vec2(0.37, 0.5) + vec2(fract(u_time * 0.01), 0.0);
    vec3 a = vec3(0.0);
    for (int i = 0; i < %(MAXN)d; i++) {
        if (float(i) >= u_n) break;
        vec3 t = texture2D(u_tx, uv).rgb;
        a += t;
        uv = fract(uv + vec2(0.0137, 0.0071) + (t.xy - 0.5) * 0.002);
    }
    gl_FragColor = mix(vec4(a * 1e-4, 1.0), %(GREEN)s, u_probe);
}
'''

CLOUD_COMMON = HEAD + '''
uniform sampler2D u_wx;
uniform sampler2D u_noise;
vec2 sliceUV(vec2 xy, float s) {
    float col = mod(s, 8.0);
    float row = floor(s / 8.0);
    vec2 inner = (xy * 31.0 + 0.5) / 32.0;
    return vec2((col + inner.x) / 8.0, (row + inner.y) / 4.0);
}
float noise3(vec3 q) {
    q = fract(q);
    float z = q.z * 32.0;
    float z0 = floor(z);
    float a = texture2D(u_noise, sliceUV(q.xy, z0)).r;
    float b = texture2D(u_noise, sliceUV(q.xy, mod(z0 + 1.0, 32.0))).r;
    return mix(a, b, z - z0);
}
float hg(float c, float g) {
    float d = max(1.0 + g * g - 2.0 * g * c, 1e-4);
    return (1.0 - g * g) / (12.566371 * d * sqrt(d));
}
'''

# one cloud-march step: weather-map fetch (equirect, atan/asin) and nothing else
K_CLOUD_EMPTY = CLOUD_COMMON + '''
void main() {
    vec3 ro = vec3((v_tex_coord - 0.5) * 1.6, 1.2);
    vec3 rd = normalize(vec3(0.05, 0.02, -1.0));
    float cov = 0.0;
    float t = 0.0;
    for (int i = 0; i < %(MAXN)d; i++) {
        if (float(i) >= u_n) break;
        vec3 p = ro + rd * t;
        float rl = length(p);
        float lon = atan(p.y, p.x);
        float lat = asin(clamp(p.z / rl, -1.0, 1.0));
        vec3 w = texture2D(u_wx, vec2(lon / 6.2831853 + 0.5, lat / 3.14159265 + 0.5)).rgb;
        cov += w.r * clamp((rl - 1.0) * 100.0, 0.0, 1.0) + w.g * 1e-3;
        t += 0.0004;
    }
    gl_FragColor = mix(vec4(vec3(cov * 1e-4), 1.0), %(GREEN)s, u_probe);
}
'''

# a full MED cloud step: weather + base + detail noise (5 fetches) + 4 light
# samples (8 fetches) + Beer / powder / two-lobe HG
K_CLOUD_FULL = CLOUD_COMMON + '''
void main() {
    vec3 ro = vec3((v_tex_coord - 0.5) * 1.6, 1.2);
    vec3 rd = normalize(vec3(0.05, 0.02, -1.0));
    vec3 sun = normalize(vec3(0.6, 0.3, 0.74));
    float cth = dot(rd, sun);
    float ph = mix(hg(cth, 0.8), hg(cth, -0.3), 0.3);
    float Tv = 1.0;
    float L = 0.0;
    float t = 0.0;
    for (int i = 0; i < %(MAXN)d; i++) {
        if (float(i) >= u_n) break;
        vec3 p = ro + rd * t;
        float rl = length(p);
        float lon = atan(p.y, p.x);
        float lat = asin(clamp(p.z / rl, -1.0, 1.0));
        vec3 w = texture2D(u_wx, vec2(lon / 6.2831853 + 0.5, lat / 3.14159265 + 0.5)).rgb;
        float base = noise3(p * 4.0);
        float det = noise3(p * 17.0);
        float dens = clamp(base * w.r - det * 0.3 + 0.2, 0.0, 1.0);
        float ls = 0.0;
        for (int j = 0; j < 4; j++) {
            ls += noise3((p + sun * (float(j) + 0.5) * 0.004) * 4.0);
        }
        float Tl = exp(-ls * 1.5);
        float powder = 1.0 - exp(-2.0 * dens);
        L += Tv * dens * (Tl * powder * ph + 0.1);
        Tv *= exp(-dens * 0.02);
        t += 0.0004;
    }
    gl_FragColor = mix(vec4(vec3(L * 1e-4, Tv, 0.0), 1.0), %(GREEN)s, u_probe);
}
'''

# 1-px checker on gl_FragCoord in the top-left corner: blocky => rendered at reduced res
CHECKER = '''
    if (v_tex_coord.x < 0.2 && v_tex_coord.y > 0.8) {
        float ck = mod(floor(gl_FragCoord.x) + floor(gl_FragCoord.y), 2.0);
        c = vec4(vec3(ck), 1.0);
    }
'''

PASS_SHADER = '''precision highp float;
varying vec2 v_tex_coord;
uniform sampler2D u_texture;
void main() { gl_FragColor = texture2D(u_texture, v_tex_coord); }
'''


def k_src(tmpl, extra=''):
    return tmpl % {'MAXN': MAXN, 'STEP': ALU_STEP, 'GREEN': GREEN, 'EXTRA': extra}


def probe_src(body, decl='', pre=''):
    return pre + HEAD + decl + '\nvoid main() {\n' + body + '\n}\n'


MT_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
             (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def multitex_src(k):
    decl = ''.join('uniform sampler2D u_t%d;\n' % i for i in range(1, k))
    lines = ['    vec3 s0 = texture2D(u_texture, vec2(0.5)).rgb;']
    lines += ['    vec3 s%d = texture2D(u_t%d, vec2(0.5)).rgb;' % (i, i) for i in range(1, k)]
    lines.append('    float bad = 0.0;')
    for i in range(k):
        c = MT_COLORS[i]
        lines.append('    bad = max(bad, length(s%d - vec3(%r, %r, %r) / 255.0));' % ((i,) + c))
    lines.append('    if (bad < 3.0 / 255.0) { gl_FragColor = %s; return; }' % GREEN)
    lines.append('    float x = v_tex_coord.x * %d.0;' % k)
    lines.append('    vec3 c = s0;')
    for i in range(1, k):
        lines.append('    if (x >= %d.0) c = s%d;' % (i, i))
    lines.append('    gl_FragColor = vec4(c * 0.6 + 0.2 * vec3(0.0, 0.0, 1.0), 1.0);')
    return probe_src('\n'.join(lines), decl)


def build_probes():
    """[(key, label, src, uniforms, textures)]; green = pass, yellow = partial, red = fail."""
    P = []
    P.append(('node', 'ctrl node', None, {}, {}))
    P.append(('shader', 'ctrl shader', probe_src('    gl_FragColor = %s;' % GREEN), {}, {}))
    P.append(('highp_eps', 'highp eps', probe_src(
        '    float a = 1.0 + u_e;\n'
        '    gl_FragColor = (a - 1.0) > 5e-7 ? %s : %s;' % (GREEN, RED),
        'uniform float u_e;\n'), {'u_e': 1e-6}, {}))
    P.append(('highp_range', 'highp >65504', probe_src(
        '    float b = u_b * u_b;\n'
        '    gl_FragColor = (b > 89000.0 && b < 91000.0) ? %s : %s;' % (GREEN, RED),
        'uniform float u_b;\n'), {'u_b': 300.0}, {}))
    P.append(('sin_large', 'sin(1e4)', probe_src(
        '    float s = sin(u_k * 10000.0);\n'
        '    float e = abs(s + 0.30561439);\n'
        '    gl_FragColor = e < 2e-3 ? %s : (e < 0.05 ? %s : %s);' % (GREEN, YELLOW, RED),
        'uniform float u_k;\n'), {'u_k': 1.0}, {}))
    P.append(('loop_break', 'uniform break', probe_src(
        '    float s = 0.0;\n'
        '    for (int i = 0; i < 256; i++) { if (float(i) >= u_m) break; s += 1.0; }\n'
        '    gl_FragColor = abs(s - 37.0) < 0.5 ? %s : %s;' % (GREEN, RED),
        'uniform float u_m;\n'), {'u_m': 37.0}, {}))
    P.append(('early_return', 'early return', probe_src(
        '    if (v_tex_coord.x > -1.0) { gl_FragColor = %s; return; }\n'
        '    gl_FragColor = %s;' % (GREEN, RED)), {}, {}))
    P.append(('local_array', 'array[i]', probe_src(
        '    vec3 k[3];\n'
        '    k[0] = vec3(1.0, 0.0, 0.0); k[1] = vec3(0.0, 2.0, 0.0); k[2] = vec3(0.0, 0.0, 3.0);\n'
        '    vec3 s = vec3(0.0);\n'
        '    for (int i = 0; i < 3; i++) s += k[i];\n'
        '    gl_FragColor = length(s - vec3(1.0, 2.0, 3.0)) < 1e-3 ? %s : %s;' % (GREEN, RED)), {}, {}))
    P.append(('fragcoord', 'gl_FragCoord', probe_src(
        '    float f = fract(gl_FragCoord.x);\n'
        '    gl_FragColor = abs(f - 0.5) < 0.01 ? %s : %s;' % (GREEN, YELLOW)), {}, {}))
    P.append(('deriv_ext', 'fwidth +ext', probe_src(
        '    float w = fwidth(v_tex_coord.x * u_sprite_size.x);\n'
        '    gl_FragColor = (w > 0.05 && w < 2.0) ? %s : %s;' % (GREEN, YELLOW),
        pre='#extension GL_OES_standard_derivatives : enable\n'), {}, {}))
    P.append(('deriv', 'fwidth', probe_src(
        '    float w = fwidth(v_tex_coord.x * u_sprite_size.x);\n'
        '    gl_FragColor = (w > 0.05 && w < 2.0) ? %s : %s;' % (GREEN, YELLOW)), {}, {}))
    P.append(('u_scale', 'u_scale==3', probe_src(
        '    gl_FragColor = abs(u_scale - 3.0) < 0.01 ? %s : %s;' % (GREEN, YELLOW)), {}, {}))
    P.append(('pow_neg', 'pow(-2,2)', probe_src(
        '    float x = pow(u_neg, 2.0);\n'
        '    gl_FragColor = abs(x - 4.0) < 1e-3 ? %s : (x == x ? %s : %s);' % (GREEN, YELLOW, RED),
        'uniform float u_neg;\n'), {'u_neg': -2.0}, {}))
    P.append(('tex4', '4 textures', multitex_src(4), {}, {'mt': 4}))
    P.append(('tex8', '8 textures', multitex_src(8), {}, {'mt': 8}))
    P.append(('exact_nearest', '8bit exact N', probe_src(
        '    float bad = 0.0;\n'
        '    for (int i = 0; i < 256; i++) {\n'
        '        float fi = float(i);\n'
        '        vec3 t = texture2D(u_tx, vec2((fi + 0.5) / 256.0, 0.5)).rgb;\n'
        '        vec3 e = vec3(fi, 255.0 - fi, mod(fi * 7.0, 256.0)) / 255.0;\n'
        '        bad = max(bad, max(abs(t.r - e.r), max(abs(t.g - e.g), abs(t.b - e.b))));\n'
        '    }\n'
        '    gl_FragColor = bad < 0.5 / 255.0 ? %s : (bad < 1.5 / 255.0 ? %s : %s);' % (GREEN, YELLOW, RED),
        'uniform sampler2D u_tx;\n'), {}, {'ramp': 'nearest'}))
    P.append(('exact_linear', '8bit exact L', P[-1][2], {}, {'ramp': 'linear'}))
    P.append(('lerp', 'lerp weights', probe_src(
        '    float bad = 0.0;\n'
        '    for (int k = 0; k < 256; k++) {\n'
        '        float f = float(k) / 255.0;\n'
        '        float r = texture2D(u_tx, vec2(0.25 + 0.5 * f, 0.5)).r;\n'
        '        bad = max(bad, abs(r - f));\n'
        '    }\n'
        '    gl_FragColor = bad < 0.75 / 255.0 ? %s : (bad < 3.0 / 255.0 ? %s : %s);' % (GREEN, YELLOW, RED),
        'uniform sampler2D u_tx;\n'), {}, {'two': 1}))
    P.append(('rgb_png', 'RGB png', probe_src(
        '    vec3 t = texture2D(u_tx, vec2(0.5)).rgb;\n'
        '    gl_FragColor = length(t - vec3(200.0, 100.0, 50.0) / 255.0) < 3.0 / 255.0 ? %s : %s;' % (GREEN, RED),
        'uniform sampler2D u_tx;\n'), {}, {'solid': 'RGB'}))
    P.append(('rgba_png', 'RGBA a=.5', probe_src(
        '    vec3 t = texture2D(u_tx, vec2(0.5)).rgb;\n'
        '    bool straight = length(t - vec3(200.0, 100.0, 50.0) / 255.0) < 3.0 / 255.0;\n'
        '    bool premul = length(t - vec3(100.0, 50.0, 25.0) / 255.0) < 4.0 / 255.0;\n'
        '    gl_FragColor = straight ? %s : (premul ? %s : %s);' % (GREEN, YELLOW, RED),
        'uniform sampler2D u_tx;\n'), {}, {'solid': 'RGBA'}))
    for key, tmpl in (('k_alu', K_ALU), ('k_alu_div', K_ALU_DIV), ('k_fetch', K_FETCH),
                      ('k_cloud_empty', K_CLOUD_EMPTY), ('k_cloud_full', K_CLOUD_FULL)):
        P.append((key, key[2:], k_src(tmpl), {'u_n': 3.0}, {'kernel': key}))
    return P


# ======================================================================= helpers
def pil_to_texture(img, nearest=False):
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    tex = scene.Texture(ui.Image.from_data(buf.getvalue()))
    if nearest:
        tex.filtering_mode = getattr(scene, 'FILTERING_NEAREST', 0)
    return tex


def solid(rgb, mode='RGB', alpha=255, size=4):
    if mode == 'RGBA':
        return Image.new('RGBA', (size, size), tuple(rgb) + (alpha,))
    return Image.new('RGB', (size, size), tuple(rgb))


def smooth_noise(w, h, rng, octaves=5, base=8):
    acc = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cw, ch = base * 2 ** o, max(2, base * 2 ** o // 2)
        g = Image.fromarray(np.uint8(rng.rand(ch, cw) * 255))
        acc += amp * np.asarray(g.resize((w, h), Image.BICUBIC), float) / 255.0
        tot += amp
        amp *= 0.5
    return acc / tot


def worley3(n, cells, rng):
    """Tiling 3D Worley (F1) on an n^3 grid with cells^3 feature points; returns 1 - F1."""
    pts = (np.indices((cells,) * 3).reshape(3, -1).T + rng.rand(cells ** 3, 3)) / cells
    g = (np.indices((n,) * 3).reshape(3, -1).T + 0.5) / n
    cid = np.floor(g * cells).astype(int)
    best = np.full(len(g), 9.0)
    for off in np.indices((3, 3, 3)).reshape(3, -1).T - 1:
        c = (cid + off) % cells
        fp = pts[(c[:, 0] * cells + c[:, 1]) * cells + c[:, 2]]
        d = g - fp
        d -= np.round(d)                        # periodic
        best = np.minimum(best, (d * d).sum(1))
    return 1.0 - np.clip(np.sqrt(best) * cells, 0, 1).reshape((n,) * 3)


def atlas_from_volume(vol):
    """32^3 -> 256x128 atlas, 8 x 4 slices (slice s at col s % 8, row s // 8, v up)."""
    n = vol.shape[0]
    out = np.zeros((4 * n, 8 * n))
    for s in range(n):
        col, row = s % 8, s // 8
        out[(3 - row) * n:(4 - row) * n, col * n:(col + 1) * n] = vol[:, :, s].T[::-1]
    return out


def bench_numpy():
    rng = np.random.RandomState(1)
    out = []
    t = time.perf_counter()
    r = np.linspace(0, 1, 256)[:, None, None]
    mu = np.linspace(-1, 1, 64)[None, :, None]
    s = np.linspace(0, 1, 40)[None, None, :]
    h = r * 100.0 + s * (1.5 - mu) * 300.0
    beta = np.array([5.8e-3, 13.5e-3, 33.1e-3])
    dens = np.exp(-h / 8.0)[..., None] * beta + (np.exp(-h / 1.2) * 4.44e-3)[..., None]
    T = np.exp(-dens.sum(axis=2) * 7.5)
    Image.fromarray(np.uint8(np.sqrt(T) * 255))
    out.append(('transmittance 256x64x40', time.perf_counter() - t))
    t = time.perf_counter()
    cov = smooth_noise(1024, 512, rng, octaves=6, base=8)
    warp = smooth_noise(1024, 512, rng, octaves=4, base=4)
    yy, xx = np.mgrid[0:512, 0:1024]
    xi = ((xx + (warp - 0.5) * 120).astype(int)) % 1024
    yi = np.clip((yy + (warp - 0.5) * 60).astype(int), 0, 511)
    cov = cov[yi, xi]
    Image.fromarray(np.uint8(np.clip(cov, 0, 1) * 255))
    out.append(('weather fbm+warp 1024x512', time.perf_counter() - t))
    t = time.perf_counter()
    vol = 0.6 * worley3(32, 4, rng) + 0.4 * worley3(32, 8, rng)
    atlas_from_volume(vol)
    out.append(('worley 32^3 2 oct', time.perf_counter() - t))
    big = Image.fromarray(np.uint8(rng.rand(1024, 2048, 3) * 255))
    t = time.perf_counter()
    buf = io.BytesIO()
    big.save(buf, 'PNG')
    t1 = time.perf_counter()
    tex = scene.Texture(ui.Image.from_data(buf.getvalue()))
    t2 = time.perf_counter()
    out.append(('png encode 2048x1024 noise', t1 - t))
    out.append(('ui.Image+Texture 2048x1024', t2 - t1))
    del tex
    return out


def stats(samples):
    d = np.sort(np.asarray(samples, float) * 1000.0)
    if len(d) == 0:
        return {'mean': float('nan'), 'p50': float('nan'), 'p95': float('nan'), 'n': 0}
    keep = d[:max(1, int(math.ceil(len(d) * 0.95)))]          # drop top 5% (hitches)
    return {'mean': float(keep.mean()), 'p50': float(np.percentile(d, 50)),
            'p95': float(np.percentile(d, 95)), 'n': len(d)}


def device_info():
    info = {}
    try:
        info['machine'] = os.uname().machine
    except Exception:
        info['machine'] = '?'
    try:
        from objc_util import ObjCClass
        pi = ObjCClass('NSProcessInfo').processInfo()
        info['thermal'] = int(pi.thermalState())
        info['lowpower'] = bool(pi.isLowPowerModeEnabled())
        info['ios'] = str(ObjCClass('UIDevice').currentDevice().systemVersion())
        info['maxfps'] = int(ObjCClass('UIScreen').mainScreen().maximumFramesPerSecond())
        d = ObjCClass('NSBundle').mainBundle().infoDictionary()
        info['app'] = '%s (%s)' % (d['CFBundleShortVersionString'], d['CFBundleVersion'])
    except Exception as e:
        info['objc'] = repr(e)[:60]
    info['python'] = sys.version.split()[0]
    info['orbitgfx'] = getattr(orbitgfx, '__version__', 'not found')
    return info


def thermal():
    try:
        from objc_util import ObjCClass
        return int(ObjCClass('NSProcessInfo').processInfo().thermalState())
    except Exception:
        return -1


# ======================================================================= scene
class Spike(scene.Scene):

    def setup(self):
        crumb('setup: start')
        self.background_color = (0.10, 0.10, 0.12)
        w, h = self.size
        self.scr = (float(w), float(h))
        self.panel = self._mission_panel(w, h)
        self.k = scene.get_screen_scale()
        self.npx = self.panel[2] * self.panel[3] * self.k * self.k
        self.info = device_info()
        self.res = {'probes': {}, 'kernels': {}, 'rr': {}, 'log': []}
        self.nodes = []
        self.samples = None
        self.prev_ts = None
        self.resume_at = 0.0
        self.snap_state = None
        self.snap_img = None
        self.done = False
        self.hdr = scene.LabelNode('orbitgfx spike', ('Menlo-Bold', 13), parent=self,
                                     position=(w / 2, h - TOP_INSET - 12), color='white')
        self.hdr.z_position = 10
        self.sub = scene.LabelNode('', ('Menlo', 10), parent=self,
                                   position=(w / 2, h - TOP_INSET - 30), color=(0.7, 0.8, 0.9))
        self.sub.z_position = 10
        crumb('setup: device %r' % (self.info,))
        self.textures = self._make_textures()
        crumb('setup: textures made')
        self.job = self.run_all()
        try:
            import console
            console.set_idle_timer_disabled(True)
        except Exception:
            pass

    # ---------------------------------------------------------- layout
    @staticmethod
    def _mission_panel(w, h):
        """The portrait globe panel from examples/mission_control.py layout()."""
        m, head, foot = 10.0, 34.0, 22.0
        fs = min(1.1, max(0.7, min(w, 520.0) / 400.0))
        tele_h = 5 * 14 * fs + 32
        top = h - head
        mw = w - 2 * m
        mh = mw / 2
        if mh > (top - foot) * 0.40:
            mh = (top - foot) * 0.40
        ty = foot + mh + 6
        gy = ty + tele_h + 6
        return (m, gy, w - 2 * m, max(80.0, top - gy))

    def status(self, a, b=''):
        self.hdr.text = a
        self.sub.text = b

    def clear(self):
        for n in self.nodes:
            n.remove_from_parent()
        self.nodes = []

    def sprite(self, src, rect, uniforms=None, tex=None, parent=None, z=0):
        x, y, w, h = rect
        node = scene.SpriteNode(tex or self.textures['small'], parent=parent or self,
                                size=(w, h), position=(x + w / 2, y + h / 2))
        node.z_position = z
        if src is not None:
            sh = scene.Shader(src)
            for k, v in (uniforms or {}).items():
                sh.set_uniform(k, v)
            node.shader = sh
        if parent is None:
            self.nodes.append(node)
        return node

    def _make_textures(self):
        rng = np.random.RandomState(7)
        T = {}
        T['small'] = pil_to_texture(Image.fromarray(np.uint8(rng.rand(128, 256, 3) * 255)))
        T['big'] = pil_to_texture(Image.fromarray(np.uint8(rng.rand(1024, 2048, 3) * 255)))
        wx = smooth_noise(1024, 512, rng, octaves=4)
        T['wx'] = pil_to_texture(Image.fromarray(np.uint8(np.dstack([wx, wx[::-1], wx[:, ::-1]]) * 255)))
        vol = worley3(32, 4, rng)
        a = np.uint8(atlas_from_volume(vol) * 255)
        T['noise'] = pil_to_texture(Image.fromarray(np.dstack([a, a[::-1], a[:, ::-1]])))
        T['mt'] = [pil_to_texture(solid(c)) for c in MT_COLORS]
        i = np.arange(256)
        ramp = np.uint8(np.stack([i, 255 - i, (i * 7) % 256], -1))[None]
        T['ramp_nearest'] = pil_to_texture(Image.fromarray(ramp), nearest=True)
        T['ramp_linear'] = pil_to_texture(Image.fromarray(ramp))
        T['two'] = pil_to_texture(Image.fromarray(np.uint8([[[0, 255, 0], [255, 0, 0]]])))
        T['solid_RGB'] = pil_to_texture(solid((200, 100, 50)))
        T['solid_RGBA'] = pil_to_texture(solid((200, 100, 50), 'RGBA', 128))
        return T

    def kernel_setup(self, key):
        """(src template, uniforms with textures) for a load kernel."""
        T = self.textures
        return {
            'k_alu': (K_ALU, {}),
            'k_alu_div': (K_ALU_DIV, {}),
            'k_fetch_small': (K_FETCH, {'u_tx': T['small']}),
            'k_fetch_big': (K_FETCH, {'u_tx': T['big']}),
            'k_cloud_empty': (K_CLOUD_EMPTY, {'u_wx': T['wx'], 'u_noise': T['noise']}),
            'k_cloud_full': (K_CLOUD_FULL, {'u_wx': T['wx'], 'u_noise': T['noise']}),
        }[key]

    # ---------------------------------------------------------- frame loop
    def update(self):
        now = time.perf_counter()
        if self.prev_ts is not None and self.samples is not None:
            self.samples.append(now - self.prev_ts)
        self.prev_ts = now
        if self.done or now < self.resume_at:
            return
        try:
            req = next(self.job)
        except StopIteration:
            self.done = True
            return
        except Exception:
            self.done = True
            tb = traceback.format_exc()
            crumb('PYTHON EXCEPTION\n' + tb)
            print(tb)
            self.res['log'].append('CRASH ' + tb.strip().splitlines()[-1])
            self.finish(crashed=tb)
            return
        if req:
            self.resume_at = time.perf_counter() + float(req)

    def wait(self, sec):
        yield sec

    def measure(self, warm=0.35, dur=1.6, minf=40, cap=4.0):
        if not ON_DEVICE:
            yield 0
            return self.fake_stats()
        yield warm
        self.samples = []
        t0 = time.perf_counter()
        while True:
            yield 0
            el = time.perf_counter() - t0
            if (el >= dur and len(self.samples) >= minf) or el >= cap:
                break
        s, self.samples = stats(self.samples), None
        return s

    def fake_stats(self):
        """Desktop dry run: synthetic GPU cost so the search logic is exercised."""
        n = getattr(self, 'fake_n', 0.0) * getattr(self, 'fake_cost', 0.02)
        gpu = 3.0 + n
        ms = 16.667 * math.ceil(gpu / 16.667)
        return {'mean': ms, 'p50': ms, 'p95': ms, 'n': 60}

    # ---------------------------------------------------------- readback
    def snapshot(self):
        """Generator: returns a PIL RGB image of the screen (top-left origin) or None."""
        if not READBACK:
            return None
        if not ON_DEVICE:
            yield 0
            snap = scene.render(self, None)
            import skia
            arr = snap.toarray(colorType=skia.kRGBA_8888_ColorType)
            return Image.fromarray(arr[..., :3])
        self.snap_state, self.snap_img = 'pending', None

        def grab():
            try:
                w, h = self.scr
                with ui.ImageContext(w, h, self.k) as ctx:
                    self.view.draw_snapshot()
                    img = ctx.get_image()
                self.snap_img = Image.open(io.BytesIO(img.to_png())).convert('RGB')
                self.snap_state = 'ok'
            except Exception as e:
                self.snap_state = 'error ' + repr(e)[:80]
        crumb('snapshot: draw_snapshot scheduled')
        ui.delay(grab, 0.05)
        t0 = time.perf_counter()
        while self.snap_state == 'pending' and time.perf_counter() - t0 < 4.0:
            yield 0.1
        crumb('snapshot: %s' % self.snap_state)
        if self.snap_state != 'ok':
            self.res['log'].append('snapshot: %s' % self.snap_state)
            return None
        return self.snap_img

    def px(self, img, x, y):
        sx = img.size[0] / self.scr[0]
        return img.getpixel((int(x * sx), int((self.scr[1] - y) * sx)))

    # ---------------------------------------------------------- the run
    def run_all(self):
        self.res['thermal0'] = thermal()
        self.status('NumPy bake benchmark', 'about 10 s, screen may freeze')
        yield 0.3
        crumb('numpy bench')
        self.res['numpy'] = bench_numpy()
        self.save_partial('numpy')
        yield from self.run_probes()
        self.save_partial('probes')
        crumb('idle')
        self.status('Idle frame time', '')
        self.sprite(probe_src('    gl_FragColor = vec4(0.02, 0.02, 0.05, 1.0);'), self.panel)
        self.res['idle'] = yield from self.measure(dur=2.0)
        self.clear()
        self.save_partial('idle')
        crumb('legacy')
        yield from self.run_legacy()
        self.save_partial('legacy')
        order = ['k_alu', 'k_fetch_small', 'k_fetch_big', 'k_cloud_empty', 'k_cloud_full', 'k_alu_div']
        costs = {'k_alu': 0.05, 'k_fetch_small': 0.03, 'k_fetch_big': 0.06,
                 'k_cloud_empty': 0.08, 'k_cloud_full': 0.9, 'k_alu_div': 0.045}
        for i, key in enumerate(order):
            probe_key = key if key in self.res['probes'] else ('k_fetch' if 'fetch' in key else key)
            if self.res['probes'].get(probe_key, 'GREEN') not in ('GREEN', 'unread'):
                self.res['kernels'][key] = {'skip': 'probe ' + self.res['probes'][probe_key]}
                continue
            self.fake_cost = costs[key]
            crumb('kernel %s: build' % key)
            tmpl, uni = self.kernel_setup(key)
            node = self.sprite(k_src(tmpl), self.panel, dict(uni, u_n=0.0, u_probe=0.0))
            sh = node.shader

            def set_n(n, sh=sh):
                sh.set_uniform('u_n', float(n))
                self.fake_n = n
            r = {}
            tag = '%s (%d/%d)' % (key, i + 1, len(order))
            r['n60'], r['log60'] = yield from self.search(set_n, T60, 0, 16, tag + ' 60fps')
            lo = r['n60'] if isinstance(r['n60'], int) else MAXN
            r['n30'], r['log30'] = yield from self.search(set_n, T30, lo, max(lo * 2, 16), tag + ' 30fps')
            self.res['kernels'][key] = r
            self.save_partial(key)
            self.clear()
            self.fake_n = 0
        crumb('reduced-res')
        yield from self.run_reduced_res()
        self.res['thermal1'] = thermal()
        self.finish()

    def search(self, set_n, target, lo, start, label):
        """Largest n (int) with trimmed-mean frame time <= target ms; '>MAXN' if never fails."""
        log = []
        n, hi = max(start, 1), None
        while n <= MAXN:
            crumb('%s n=%d' % (label, n))
            set_n(n)
            self.status(label, 'n=%d  lo=%s hi=%s' % (n, lo, hi))
            st = yield from self.measure()
            ok = st['mean'] <= target
            log.append('%d%s%.1f' % (n, '+' if ok else '-', st['mean']))
            if ok:
                lo = n
                if n == MAXN:
                    break
                n = min(MAXN, n * 2)
            else:
                hi = n
                break
        if hi is None:
            return '>%d' % MAXN, ' '.join(log)
        while hi - lo > max(2, 0.06 * hi):
            mid = int(round((lo + hi) / 2.0)) if lo < 8 else int(round(math.sqrt(lo * hi)))
            crumb('%s n=%d' % (label, mid))
            set_n(mid)
            self.status(label, 'n=%d  lo=%s hi=%s' % (mid, lo, hi))
            st = yield from self.measure()
            ok = st['mean'] <= target
            log.append('%d%s%.1f' % (mid, '+' if ok else '-', st['mean']))
            if ok:
                lo = mid
            else:
                hi = mid
        return lo, ' '.join(log)

    # ---------------------------------------------------------- (c) (d) probes
    def run_probes(self):
        self.status('Probes', 'green = pass  yellow = partial  red/blank = fail')
        probes = build_probes()
        w, h = self.scr
        cols, tile, gap = 5, min(72.0, (w - 20) / 5 - 8), 8.0
        x0 = (w - cols * tile - (cols - 1) * gap) / 2
        top = h - TOP_INSET - 50
        cells = []
        for i, (key, label, src, uni, tex) in enumerate(probes):
            crumb('probe tile %s' % key)
            cx = x0 + (i % cols) * (tile + gap)
            cy = top - (i // cols + 1) * (tile + 16)
            rect = (cx, cy, tile, tile)
            t = None
            u = dict(uni)
            if 'mt' in tex:
                t = self.textures['mt'][0]
                for j in range(1, tex['mt']):
                    u['u_t%d' % j] = self.textures['mt'][j]
            if 'ramp' in tex:
                u['u_tx'] = self.textures['ramp_' + tex['ramp']]
            if 'two' in tex:
                u['u_tx'] = self.textures['two']
            if 'solid' in tex:
                u['u_tx'] = self.textures['solid_' + tex['solid']]
            if 'kernel' in tex:
                k = tex['kernel']
                k = 'k_fetch_big' if k == 'k_fetch' else k
                u.update(self.kernel_setup(k)[1])
                u['u_probe'] = 1.0
            if src is None:
                x, y, tw, th = rect
                node = scene.SpriteNode(None, parent=self, size=(tw, th),
                                        position=(x + tw / 2, y + th / 2), color='#00ff00')
                self.nodes.append(node)
            else:
                try:
                    self.sprite(src, rect, u, tex=t)
                except Exception as e:
                    self.res['probes'][key] = 'EXC ' + repr(e)[:40]
            lab = scene.LabelNode(label, ('Menlo', 8), parent=self,
                                  position=(cx + tile / 2, cy - 7), color='white')
            self.nodes.append(lab)
            cells.append((key, cx + tile / 2, cy + tile / 2, cx + tile * 0.2, cy + tile / 2))
        crumb('probes: all tiles built')
        yield 1.0
        crumb('probes: rendered 1 s')
        img = yield from self.snapshot()
        if img is None and not READBACK:
            self.res['probes'].update(SCREENSHOT_PROBES)
        for key, mx, my, sx, sy in cells:
            if key in self.res['probes']:
                continue
            if img is None:
                self.res['probes'][key] = 'unread'
                continue
            c = self.px(img, mx, my)
            self.res['probes'][key] = self.classify(c)
            if key.startswith('tex') and self.res['probes'][key] != 'GREEN':
                n = 4 if key == 'tex4' else 8
                x0c = mx - tile / 2
                cs = [self.px(img, x0c + (j + 0.5) * tile / n, my) for j in range(n)]
                self.res['probes'][key] += ' ' + '/'.join('%02x%02x%02x' % c3[:3] for c3 in cs)
        if img is None:
            self.status('Probes (graded from v1 screenshot)', 'continuing in 3 s')
            yield 3.0
        else:
            yield 1.0
        self.clear()

    @staticmethod
    def classify(c):
        r, g, b = [v / 255.0 for v in c[:3]]
        if g > 0.8 and r < 0.2 and b < 0.2:
            return 'GREEN'
        if r > 0.8 and g > 0.8 and b < 0.2:
            return 'YELLOW'
        if r > 0.8 and g < 0.2 and b < 0.2:
            return 'RED'
        return 'OTHER(%02x%02x%02x)' % tuple(int(v) for v in c[:3])

    # ---------------------------------------------------------- legacy globe
    def run_legacy(self):
        L = self.res['legacy'] = {}
        if orbitgfx is None:
            L['skip'] = 'orbitgfx not importable'
            return
        self.status('Legacy globe', 'baking/loading earth texture')
        yield 0.2
        from orbitgfx import GlobeView, Camera, cached_earth_image, pil_texture
        from orbitgfx.coastlines import coastlines_xyz
        from orbitgfx.shaders import build_globe_shader
        from orbitgfx.astro import RE
        t = time.perf_counter()
        tex = pil_texture(cached_earth_image(coastlines_xyz(RE)))
        L['tex_s'] = time.perf_counter() - t
        cam = Camera(az=0.8, el=0.45)
        gv = GlobeView(self, tex, fill=0.44, ref_radius_km=RE + 500.0)
        self.nodes.append(gv.node)
        a = RE + 500.0
        sat = (a * 0.6, a * 0.64, a * 0.48)
        u = np.array(sat) / a
        v = np.cross(u, (0.0, 0.0, 1.0))
        v = v / np.linalg.norm(v) * 7.6
        sun = (0.80, -0.55, 0.24)
        sun = tuple(np.array(sun) / np.linalg.norm(sun))
        args = dict(sat=sat, vel=tuple(v), trail_rad=0.6, stations=[], cloud_hours=1.0)
        gv.layout(self.panel)
        gv.update(cam, sun, 1.3, **args)
        self.status('Legacy globe (today\'s shader)', 'measuring')
        L['plain'] = yield from self.measure(dur=3.0)
        src = build_globe_shader()
        src = src.replace('uniform float u_time;', 'uniform float u_time;\nuniform float u_load;', 1)
        load = ('    vec3 p_ = vec3(v_tex_coord * 2.0 - 1.0, 0.3);\n    float od_ = 0.0;\n'
                '    vec3 L_ = vec3(0.0);\n    for (int li_ = 0; li_ < %d; li_++) {\n'
                '        if (float(li_) >= u_load) break;\n%s    }\n'
                '    acc.rgb += L_ * 1e-7;\n    gl_FragColor = acc;' % (MAXN, ALU_STEP))
        if src.count('gl_FragColor = acc;') == 1:
            src = src.replace('gl_FragColor = acc;', load)
            gv.shader = scene.Shader(src)
            gv.node.shader = gv.shader
            gv.shaders = [gv.shader]
            gv.shader.set_uniform('u_load', 0.0)
            gv.layout(self.panel)
            gv.update(cam, sun, 1.3, **args)
            self.fake_cost = 0.05

            def set_n(n):
                gv.shader.set_uniform('u_load', float(n))
                self.fake_n = n
            L['alu_n60'], L['log'] = yield from self.search(set_n, T60, 0, 16, 'Legacy + ALU headroom')
        self.clear()
        self.fake_n = 0

    # ---------------------------------------------------------- (b) reduced resolution
    def run_reduced_res(self):
        R = self.res['rr']
        n60 = self.res['kernels'].get('k_alu', {}).get('n60')
        if not isinstance(n60, int) or n60 < 4:
            R['skip'] = 'no k_alu n60'
            return
        n = min(MAXN, int(n60 * 2))
        R['n'] = n
        self.fake_cost = 0.05
        self.fake_n = n
        extra = CHECKER if self.res['probes'].get('fragcoord') in ('GREEN', 'YELLOW') else ''
        src = k_src(K_ALU, extra)
        x, y, w, h = self.panel
        cx, cy = x + w / 2, y + h / 2
        uni = {'u_n': float(n), 'u_probe': 0.0}
        has_eff = hasattr(scene, 'EffectNode')

        def eff_variant(with_shader, outer):
            holder = scene.Node(parent=self, position=(cx, cy)) if outer else None
            e = scene.EffectNode(parent=holder or self)
            if not outer:
                e.position = (cx, cy)
            (holder or e).x_scale = 2.0
            (holder or e).y_scale = 2.0
            if with_shader:
                e.shader = scene.Shader(PASS_SHADER)
            try:
                e.effects_enabled = True
            except Exception:
                pass
            self.sprite(src, (-w / 4, -h / 4, w / 2, h / 2), uni, parent=e)
            self.nodes.append(holder or e)

        def scaled_sprite():
            node = self.sprite(src, (cx - w / 4, cy - h / 4, w / 2, h / 2), uni)
            node.x_scale = node.y_scale = 2.0

        variants = [('full', lambda: self.sprite(src, self.panel, uni), 1.0),
                    ('sprite_x2', scaled_sprite, 1.0)]
        if has_eff:
            variants += [('eff_x2', lambda: eff_variant(False, False), 0.25),
                         ('eff_x2_shader', lambda: eff_variant(True, False), 0.25),
                         ('node_x2_eff', lambda: eff_variant(False, True), 0.25)]
        else:
            R['effectnode'] = 'missing'
        for name, build, frac in variants:
            self.status('Reduced resolution: ' + name, 'n=%d (2x the 60 fps load)' % n)
            crumb('reduced-res %s: build' % name)
            try:
                build()
            except Exception as e:
                R[name] = 'EXC ' + repr(e)[:60]
                self.clear()
                continue
            self.fake_n = n * frac
            crumb('reduced-res %s: measure' % name)
            st = yield from self.measure(dur=2.0)
            R[name] = {'mean': st['mean'], 'p95': st['p95']}
            if extra:
                img = yield from self.snapshot()
                if img is not None:
                    R[name]['checker'] = self.checker_metric(img, x + w * 0.1, y + h * 0.9)
            self.clear()
        yield from self.run_rtt(src, uni)

    def run_rtt(self, src, uni):
        """render_to_texture(): render a half-size sprite each frame, show it scaled x2."""
        R = self.res['rr']
        x, y, w, h = self.panel
        cands = [(n, getattr(scene.Scene, n)) for n in ('render_to_texture',)
                 if hasattr(scene.Scene, n)]
        cands += [('Node.' + n, None) for n in ('render_to_texture',) if hasattr(scene.Node, n)]
        R['rtt_api'] = ','.join(c[0] for c in cands) or 'none'
        if not cands:
            return
        off = self.sprite(src, (x + w / 4, y + h / 4, w / 2, h / 2), uni)
        show = scene.SpriteNode(self.textures['small'], parent=self, size=(w, h),
                                position=(x + w / 2, y + h / 2))
        show.z_position = 1
        self.nodes.append(show)

        def render_once():
            if hasattr(off, 'render_to_texture'):
                return off.render_to_texture()
            return self.render_to_texture(off)
        crumb('render_to_texture: first call (%s)' % R['rtt_api'])
        try:
            tex = render_once()
            R['rtt_size'] = '%sx%s' % tuple(tex.size)
        except Exception as e:
            R['rtt'] = 'EXC ' + repr(e)[:70]
            self.clear()
            return
        self.status('Reduced resolution: render_to_texture', 'every frame')
        yield 0.3
        self.samples = []
        t0 = time.perf_counter()
        errs = 0
        while time.perf_counter() - t0 < 2.0:
            try:
                show.texture = render_once()
            except Exception:
                errs += 1
            yield 0
        st, self.samples = stats(self.samples), None
        R['rtt'] = {'mean': st['mean'], 'p95': st['p95'], 'errs': errs}
        self.clear()

    def checker_metric(self, img, x, y):
        """Mean |difference| of horizontal neighbours (0..1) in a 24x24 px patch: 1-px checker ~1."""
        sx = img.size[0] / self.scr[0]
        px0, py0 = int(x * sx), int((self.scr[1] - y) * sx)
        a = np.asarray(img.crop((px0, py0, px0 + 24, py0 + 24)).convert('L'), float) / 255.0
        return float(np.abs(np.diff(a, axis=1)).mean())

    # ---------------------------------------------------------- report
    def report(self):
        I, R = self.info, self.res
        L = []
        L.append('ORBITGFX SPIKE v%d' % SPIKE_VERSION)
        L.append('device %s iOS %s app %s py %s orbitgfx %s' % (
            I.get('machine'), I.get('ios', '?'), I.get('app', '?'), I.get('python'), I.get('orbitgfx')))
        L.append('screen %.0fx%.0f pt @%g maxfps %s lowpower %s thermal %s->%s %s' % (
            self.scr[0], self.scr[1], self.k, I.get('maxfps', '?'), I.get('lowpower', '?'),
            R.get('thermal0'), R.get('thermal1', '?'), I.get('objc', '')))
        px = self.panel
        L.append('panel %.0fx%.0f pt = %.2f Mpx  T60 %.1f T30 %.1f ms' % (px[2], px[3], self.npx / 1e6, T60, T30))
        L.append('numpy ' + '  '.join('%s %.2fs' % kv for kv in R.get('numpy', [])))
        L.append('probes ' + ' '.join('%s=%s' % kv for kv in R['probes'].items()))
        i = R.get('idle')
        if i:
            L.append('idle mean %.2f p50 %.2f p95 %.2f ms (%d frames)' % (i['mean'], i['p50'], i['p95'], i['n']))
        lg = R.get('legacy', {})
        if 'plain' in lg:
            p = lg['plain']
            L.append('legacy mean %.2f p50 %.2f p95 %.2f ms  +alu n60 %s  tex %.2fs' % (
                p['mean'], p['p50'], p['p95'], lg.get('alu_n60', '?'), lg.get('tex_s', 0)))
            L.append('  log ' + lg.get('log', ''))
        elif lg:
            L.append('legacy ' + str(lg))
        for key, r in R['kernels'].items():
            if 'skip' in r:
                L.append('%s SKIP %s' % (key, r['skip']))
                continue
            line = '%s n60 %s n30 %s' % (key, r['n60'], r['n30'])
            if isinstance(r['n60'], int) and isinstance(r['n30'], int) and r['n30'] > r['n60']:
                line += '  %.3f ns/px/unit' % (16.667e6 / ((r['n30'] - r['n60']) * self.npx))
            L.append(line)
            L.append('  60: ' + r['log60'])
            L.append('  30: ' + r['log30'])
        rr = R.get('rr', {})
        if rr:
            parts = []
            for k, v in rr.items():
                if isinstance(v, dict):
                    parts.append('%s=%s' % (k, ','.join('%s:%.2f' % (a, b) if isinstance(b, float)
                                                       else '%s:%s' % (a, b) for a, b in v.items())))
                else:
                    parts.append('%s=%s' % (k, v))
            L.append('reduced-res ' + ' '.join(parts))
        if R['log']:
            L.append('notes ' + ' | '.join(R['log']))
        return '\n'.join(L)

    def save_partial(self, stage):
        crumb('stage done: %s' % stage)
        try:
            with open(os.path.join(HERE, 'spike_results.txt'), 'w') as f:
                f.write(self.report() + '\nPARTIAL (last stage: %s)\n' % stage)
        except Exception as e:
            crumb('save_partial failed %r' % e)

    def finish(self, crashed=None):
        crumb('finish')
        self.clear()
        text = self.report()
        if crashed:
            text += '\nCRASH\n' + crashed
        print(text)
        saved = ''
        try:
            path = os.path.join(HERE, 'spike_results.txt')
            with open(path, 'w') as f:
                f.write(text + '\n')
            saved = 'saved spike_results.txt'
        except Exception as e:
            saved = 'save failed %r' % e
        try:
            import clipboard
            clipboard.set(text)
            saved += ' + copied to clipboard'
        except Exception:
            pass
        self.res['saved'] = saved
        self.status('DONE: ' + saved, 'close (X) and paste the block into the chat')
        lines = text.splitlines()
        y = self.scr[1] - TOP_INSET - 50
        for ln in lines[:64]:
            lab = scene.LabelNode(ln[:78], ('Menlo', 7), parent=self, position=(6, y), color='white')
            lab.anchor_point = (0, 0.5)
            self.nodes.append(lab)
            y -= 11
        self.done = True
        crumb('done')
        try:
            import console
            console.set_idle_timer_disabled(False)
        except Exception:
            pass


if __name__ == '__main__':
    scene.run(Spike(), orientation=getattr(scene, 'PORTRAIT', 0), show_fps=True)
