"""Read an orbitgfx device report (examples/device_check.py) on the desktop.

    python -m orbitgfx.tools.device_report orbitgfx_device_report.zip [--render] [--out DIR]

Prints what the device found, flags what needs attention, and with --render re-renders
every screenshot and compile check in the sandbox at the same pose (needs moderngl +
skia) and diffs them: side-by-side PNGs go to --out.

Allowed differences device vs desktop: star twinkle, aurora, lightning and cloud
detail animate with the scene clock (u_time), fonts differ, Mesa and Apple GPUs round
differently. Flags: a globe missing (mostly background), NaN pixels, the raw data
texture showing (a shader that failed to compile can fall back to it), or a large
mean difference.
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from orbitgfx.sandbox.runner import Runner  # noqa: E402,F401  (installs the shim)
import scene  # noqa: E402

from orbitgfx.examples import device_check as DC  # noqa: E402

# 2.0.0a10 numbers on iPhone 16 Pro Max (TUNING.md): headroom = globes per 60 fps frame
REFERENCE_HEADROOM = {('LOW', 'overview'): 4, ('MED', 'overview'): 3, ('HIGH', 'overview'): 3,
                      ('ULTRA', 'overview'): 2, ('LOW', 'chase'): 2, ('MED', 'chase'): 2,
                      ('HIGH', 'chase'): 2, ('ULTRA', 'chase'): 1, ('LOW', 'chase/P'): 2,
                      ('MED', 'chase/P'): 1, ('HIGH', 'chase/P'): 1, ('LOW', 'astro'): 3,
                      ('MED', 'astro'): 3, ('HIGH', 'astro'): 2, ('ULTRA', 'astro'): 1}


class Source(object):
    """A report directory or zip."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path) if path.endswith('.zip') else None
        self.path = path
        names = self.zip.namelist() if self.zip else [
            os.path.relpath(os.path.join(r, f), path) for r, _, fs in os.walk(path) for f in fs]
        rep = [n for n in names if n.endswith('report.json')]
        if not rep:
            raise SystemExit('no report.json in %s' % path)
        self.root = os.path.dirname(rep[0])
        self.report = json.loads(self.read('report.json'))

    def read(self, rel):
        p = os.path.join(self.root, rel) if self.root else rel
        if self.zip:
            return self.zip.read(p)
        with open(os.path.join(self.path, p), 'rb') as fh:
            return fh.read()

    def image(self, rel):
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(self.read(rel))).convert('RGB'))


# ------------------------------------------------------------------ findings
def findings(R):
    st = R['stages']
    out = []
    pc = st.get('pacing', {})
    if pc and pc.get('updates') != pc.get('draws'):
        out.append('LIFECYCLE: %s updates vs %s draws while pacing (expected one each per frame)'
                   % (pc.get('updates'), pc.get('draws')))
    lc = st.get('lifecycle', {})
    if lc and abs(lc.get('updates', 0) - lc.get('draws', 0)) > 2:
        out.append('LIFECYCLE: %s updates vs %s draws over the run' % (lc['updates'], lc['draws']))
    cap = st.get('capture', {})
    if cap and not cap.get('method'):
        out.append('CAPTURE: no working screen read-back: %s' % json.dumps(cap.get('tried'))[:400])
    elif cap and (cap.get('ramp_r_err', 0) > 2 or not cap.get('ramp_monotonic', True)):
        out.append('CAPTURE: colour response not identity (r err %s, monotonic %s): GPU maths '
                   'decoded through the measured ramp' % (cap.get('ramp_r_err'), cap.get('ramp_monotonic')))
    for c in st.get('compile', {}).get('checks', []):
        px = c.get('pixels', {})
        if 'error' in c:
            out.append('COMPILE %s: Python error\n%s' % (c['name'], c['error']))
        elif px.get('nan', 0) > 0:
            out.append('COMPILE %s: %.3f%% NaN pixels' % (c['name'], 100 * px['nan']))
        elif px.get('bg', 0) > 0.97:
            out.append('COMPILE %s: panel is background only (shader failed or drew nothing)' % c['name'])
        elif px.get('raw_red', 0) > 0.05:
            out.append('COMPILE %s: raw data texture visible (%.1f%% pure red): compile fallback?'
                       % (c['name'], 100 * px['raw_red']))
        fs = c.get('first_frames', {})
        if fs.get('settle_max_ms') and fs['settle_max_ms'] > 500:
            out.append('COMPILE %s: first frame %.0f ms (compile hitch)' % (c['name'], fs['settle_max_ms']))
    for c in st.get('nan_scan', {}).get('checks', []):
        if c['nan'] > 0:
            out.append('NAN %s %s %s: %.3f%% of pixels' % (c['quality'], c['view'], c['sun'], 100 * c['nan']))
    for p in st.get('gpu_math', {}).get('probes', []):
        if 'error' in p:
            out.append('GPU MATHS %s: error\n%s' % (p['name'], p['error']))
        elif p.get('ok') is False:
            out.append('GPU MATHS %s: max error %.2g at x=%.3f (tolerance %.2g)'
                       % (p['name'], p['max_err'], p['at_x'], p['tol']))
        elif p['name'] in ('pow(x, 5)', 'pow(x, 2)') and p.get('values'):
            xs, vs = np.array(p['xs']), np.array(p['values'])
            neg = xs < 0
            k = 5 if '5' in p['name'] else 2
            if np.abs(vs[neg] - xs[neg] ** k).max() > 0.01:
                out.append('INFO GPU MATHS %s for x < 0 gives %s, not x^%d: undefined here as the '
                           'spec says (README rule 1)' % (p['name'], vs[neg].round(3).tolist(), k))
            else:
                out.append('INFO GPU MATHS %s for x < 0 equals x^%d here (rewritten or defined by this '
                           'driver; still undefined by the spec, keep the rule)' % (p['name'], k))
        elif p['name'].startswith('1 - ') and p.get('values'):
            lo = min(p['values'])
            if lo < -1e-9:
                out.append('INFO GPU MATHS %s rounds to %.2g (< 0) here: pow(1.0 - z, y) would be NaN; '
                           'the max(.., 0.0) clamps are needed' % (p['name'], lo))
    vis = st.get('gpu_math', {}).get('nan_visible', {})
    if vis and not any(vis.values()):
        out.append('NAN DETECTION: shaders on this device cannot see NaNs (both tests folded '
                   'away): a clean nan_scan proves nothing; compare compile/screens colours instead')
    elif vis and not vis.get('sees NaN of sqrt(x)', True):
        out.append('INFO NAN DETECTION: x == x is folded away here, the abs() test works; '
                   'NANCHECK uses !(abs(x) < limit) since 2.0.0a17, so nan_scan is valid for '
                   'builds from 2.0.0a17 on (not for older ones)')
    slip = st.get('bench', {}).get('slip_ms')
    for r in st.get('bench', {}).get('runs', []):
        s = r['stats']
        ref = REFERENCE_HEADROOM.get((r['quality'], r['view']))
        if slip and s.get('mean95_ms', 0) > slip and r['quality'] != 'LEGACY':
            out.append('PERF %s %s: %.1f ms per frame, below the display rate (p95 %.1f)'
                       % (r['quality'], r['view'], s['mean95_ms'], s['p95_ms']))
        elif ref and r.get('headroom') and r['headroom'] < ref:
            out.append('PERF %s %s: headroom x%d, the 2.0.0a10 reference device had x%d'
                       % (r['quality'], r['view'], r['headroom'], ref))
        if s.get('settle_max_ms') and s['settle_max_ms'] > 250:
            out.append('HITCH %s %s: %.0f ms frame right after building' % (r['quality'], r['view'],
                                                                            s['settle_max_ms']))
    for p in st.get('refraction', {}).get('pairs', []):
        on = [x['mean95_ms'] for x in p.get('on', []) if 'mean95_ms' in x]
        off = [x['mean95_ms'] for x in p.get('off', []) if 'mean95_ms' in x]
        if on and off:
            d = np.mean(on) / np.mean(off) - 1.0
            drift = abs(on[0] - on[-1]) / max(on[0], 1e-6)
            out.append('REFRACTION COST %s: %+.1f%% frame time (on %s, off %s, on-drift %.0f%%)%s'
                       % (p['case'], 100 * d, on, off, 100 * drift,
                          '  <- worth optimising' if d > 0.05 and drift < 0.03 else ''))
    w = st.get('weather', {}).get('runs', [])
    if len(w) > 1:
        base = w[0]['stats'].get('p99_ms', 0)
        for r in w[1:]:
            if r['stats'].get('max_ms', 0) > max(2.5 * base, 34.0):
                out.append('WEATHER x%d: max frame %.0f ms (static p99 %.0f): film frame loads hitch'
                           % (r['warp'], r['stats']['max_ms'], base))
    v = R.get('vitals', [])
    th = [x.get('thermal') for x in v if x.get('thermal') is not None]
    if th and max(th) >= 2:
        out.append('THERMAL: reached state %d (serious/critical): later timings are throttled' % max(th))
    info = st.get('info', {})
    if info.get('low_power'):
        out.append('LOW POWER MODE was on: timings are throttled')
    for k, e in R.get('errors', {}).items():
        out.append('STAGE ERROR %s:\n%s' % (k, e))
    for k in R.get('skipped_after_crash', []):
        out.append('CRASHED/HUNG on the previous run in %s (skipped this time)' % k)
    return out


# ------------------------------------------------------------------ desktop renders
class Host(scene.Scene):
    pass


def render_like(spec, pose, size, rect, u_time, tex):
    """Render one spec + pose in the sandbox; returns the rect crop (RGB) and its scale."""
    import skia
    host = Host()
    host._fw_set(size=scene.Size(*size), bounds=scene.Rect(0, 0, *size))
    host.background_color = DC.BG
    g = DC.build_globe(host, tex, spec, rect)
    DC.apply_pose(g, pose)
    snap = scene.render(host, None, u_time=u_time)
    a = snap.toarray(colorType=skia.kRGBA_8888_ColorType)
    k = a.shape[1] / float(size[0])
    x, y, w, h = rect
    r0, r1 = int((size[1] - y - h) * k), int((size[1] - y) * k)
    return a[r0:r1, int(x * k):int((x + w) * k), :3], k


def resize(a, shape):
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((shape[1], shape[0]), Image.LANCZOS))


def compare(src, out):
    from PIL import Image
    from orbitgfx import cached_earth_image, pil_texture
    from orbitgfx.coastlines import coastlines_xyz
    from orbitgfx.astro import RE
    R = src.report
    size = R['scene_size']
    tex = pil_texture(cached_earth_image(coastlines_xyz(RE)))
    os.makedirs(out, exist_ok=True)
    rows = []
    for s in R['stages'].get('screens', {}).get('shots', []):
        dev = src.image(s['file'])
        desk, _ = render_like(s['spec'], s['pose'], size, s['rect'], s['scene_t'], tex)
        desk = resize(desk, dev.shape)
        d = np.abs(dev.astype(int) - desk.astype(int)).max(-1)
        rows.append((os.path.basename(s['file']), float(d.mean()), float((d > 24).mean())))
        Image.fromarray(np.concatenate([dev, desk, np.repeat(np.minimum(d * 4, 255)[..., None], 3, -1)
                                        .astype(np.uint8)], 1)).save(os.path.join(out, 'cmp_' + os.path.basename(s['file'])))
    print('\nscreens: device | desktop | 4x difference  ->  %s' % out)
    for name, mean, frac in rows:
        print('  %-48s mean |d| %5.1f   pixels off by >24: %5.1f%%%s' % (
            name, mean, 100 * frac, '   <- look' if frac > 0.10 or mean > 12 else ''))
    print('\ncompile checks: device vs desktop pixel stats')
    for c in R['stages'].get('compile', {}).get('checks', []):
        if 'pixels' not in c:
            continue
        rect = R['panel'] if c['spec']['rect'] == 'panel' else [0, 0] + list(size)
        desk, _ = render_like(c['spec'], c['pose'], size, rect, 1.0, tex)
        ds = DC.pixel_stats(desk)
        dv = c['pixels']
        flag = ''
        if dv['bg'] > ds['bg'] + 0.1:
            flag = '  <- device drew less'
        elif abs(np.subtract(dv['mean'], ds['mean'])).max() > 25:
            flag = '  <- colour differs'
        print('  %-26s bg %.3f/%.3f  mean %s / %s%s' % (c['name'], dv['bg'], ds['bg'], dv['mean'],
                                                         ds['mean'], flag))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('report', help='orbitgfx_device_report.zip or its folder')
    ap.add_argument('--render', action='store_true', help='re-render on the desktop and diff')
    ap.add_argument('--out', default='device_compare')
    ap.add_argument('--json', action='store_true', help='also dump the full report')
    a = ap.parse_args(argv)
    src = Source(a.report)
    R = src.report
    print(DC.summarize(R))
    print('\nFINDINGS')
    for f in findings(R) or ['none']:
        print(' - ' + f)
    gm = R['stages'].get('gpu_math', {}).get('probes', [])
    if gm:
        print('\nGPU maths (device vs CPU)')
        for p in gm:
            print('  %-22s %s' % (p['name'], 'max err %.2g (tol %.2g)' % (p['max_err'], p['tol'])
                                  if 'max_err' in p else p.get('values', p.get('error', ''))))
    if a.json:
        print(json.dumps(R, indent=1))
    if a.render:
        compare(src, a.out)


if __name__ == '__main__':
    main()
