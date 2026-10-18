"""examples/device_check.py (the on-device check) runs in the sandbox, and
tools/device_report.py reads what it writes.

    python -m orbitgfx.tests.test_device_check
Only the quick stages run here (capture, GPU maths); the full check is for the device.
"""
import os
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(PKG))
from orbitgfx.sandbox.runner import Runner  # noqa: E402
from orbitgfx.tests._gpu import gpu, run_all  # noqa: E402


@gpu(what='render')
def test_device_check_quick_stages():
    import orbitgfx.examples.device_check as DC
    from orbitgfx.tools import device_report
    base = tempfile.mkdtemp()

    def quick(mod):                       # the Runner loads the script as its own module
        mod.ONLY, mod.OUT_BASE = ('gpu_math',), base

    with Runner(DC.__file__, 'DeviceCheck', size=(390, 844), before_setup=quick) as r:
        s = r.scene
        for _ in range(60 * 120):
            if s.done:
                break
            r.frame()
        assert s.done
        R = s.report
    st = R['stages']
    assert not R['errors'], R['errors']
    assert st['capture']['method'] == 'sandbox' and st['capture']['ramp_r_err'] == 0
    assert st['pacing']['updates'] == st['pacing']['draws'] > 0
    gm = st['gpu_math']
    assert gm['all_ok'], [p for p in gm['probes'] if not p.get('ok', True)]
    assert all(gm['nan_visible'].values()), gm['nan_visible']        # Mesa keeps NaN tests
    assert 'compile' not in st and 'bench' not in st                   # ONLY honoured
    assert os.path.exists(s.zip_path)
    found = device_report.findings(R)
    assert not [f for f in found if f.startswith(('LIFECYCLE', 'CAPTURE', 'GPU MATHS', 'STAGE'))], found
    src = device_report.Source(s.zip_path)
    assert src.report['format'] == DC.VERSION
    return {p['name']: p.get('max_err') for p in gm['probes']}


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    run_all(globals())
