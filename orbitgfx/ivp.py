"""Small NumPy-only replacement for scipy.integrate.solve_ivp (RK45 only).

Supports adaptive Dormand-Prince 5(4), optional t_eval, forward/backward
integration, dense output at requested t_eval, args, max_step, first_step.
Not a full SciPy solve_ivp implementation.
"""
from types import SimpleNamespace
import numpy as np

_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)
_A = (
    (),
    (1/5,),
    (3/40, 9/40),
    (44/45, -56/15, 32/9),
    (19372/6561, -25360/2187, 64448/6561, -212/729),
    (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
)
_B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
_E = _B - np.array([5179/57600, 0, 7571/16695, 393/640,
                   -92097/339200, 187/2100, 1/40])
# Quartic dense-output polynomial coefficients, as used in SciPy RK45.
_P = np.array([
    [1, -8048581381/2820520608, 8663915743/2820520608, -12715105075/11282082432],
    [0, 0, 0, 0],
    [0, 131558114200/32700410799, -68118460800/10900136933, 87487479700/32700410799],
    [0, -1754552775/470086768, 14199869525/1410260304, -10690763975/1880347072],
    [0, 127303824393/49829197408, -318862633887/49829197408, 701980252875/199316789632],
    [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
    [0, 40617522/29380423, -110615467/29380423, 69997945/29380423],
], dtype=float)


def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None,
              rtol=1e-3, atol=1e-6, args=None, max_step=np.inf,
              first_step=None, **kwargs):
    """Integrate y'=fun(t,y), returning SciPy-style t, y and status fields.

    Only method='RK45' and the named options are supported.
    """
    if method != 'RK45':
        raise ValueError("Only method='RK45' is supported")
    if kwargs:
        raise TypeError('Unsupported solve_ivp options: ' + ', '.join(kwargs))
    t0, tf = map(float, t_span)
    y = np.asarray(y0, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError('y0 must be a finite 1-D array')
    if not (np.isfinite(t0) and np.isfinite(tf)):
        raise ValueError('t_span must contain finite times')
    if not (np.isfinite(max_step) or np.isinf(max_step)) or max_step <= 0:
        raise ValueError('max_step must be positive')
    if first_step is not None and (not np.isfinite(first_step) or first_step <= 0):
        raise ValueError('first_step must be finite and positive')
    rtol = np.asarray(rtol, dtype=float)
    atol = np.asarray(atol, dtype=float)
    if np.any(rtol <= 0) or np.any(atol < 0):
        raise ValueError('rtol must be positive and atol nonnegative')
    if args is None:
        args = ()
    else:
        args = tuple(args)

    direction = 1 if tf >= t0 else -1
    if t_eval is not None:
        desired = np.asarray(t_eval, dtype=float)
        if desired.ndim != 1 or not np.all(np.isfinite(desired)):
            raise ValueError('t_eval must be a finite 1-D array')
        if np.any(direction * np.diff(desired) <= 0):
            raise ValueError('t_eval must be strictly ordered in integration direction')
        if np.any(direction * (desired - t0) < 0) or np.any(direction * (tf - desired) < 0):
            raise ValueError('t_eval must lie within t_span')
    else:
        desired = None

    nfev = 0
    def f(t, v):
        nonlocal nfev
        out = np.asarray(fun(t, v, *args), dtype=float)
        nfev += 1
        if out.shape != y.shape or not np.all(np.isfinite(out)):
            raise ValueError('fun must return a finite vector matching y0')
        return out

    if desired is None:
        ts, ys = [t0], [y.copy()]
    else:
        ts, ys = [], []
    idx = 0
    if desired is not None and len(desired) and desired[0] == t0:
        ts.append(t0)
        ys.append(y.copy())
        idx = 1
    if t0 == tf:
        return SimpleNamespace(t=np.asarray(ts), y=np.stack(ys, axis=1) if ys else np.empty((len(y), 0)),
                               success=True, status=0, message='The solver successfully reached the end of the integration interval.',
                               nfev=0, njev=0, nlu=0)

    t = t0
    k0 = f(t, y)
    if first_step is None:
        scale = atol + np.abs(y) * rtol
        d0 = np.linalg.norm(y / scale) / np.sqrt(len(y))
        d1 = np.linalg.norm(k0 / scale) / np.sqrt(len(y))
        h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
        h0 = min(h0, abs(tf - t0), max_step)
        y1 = y + direction * h0 * k0
        d2 = np.linalg.norm((f(t + direction*h0, y1) - k0) / scale) / np.sqrt(len(y)) / h0
        h = min(100*h0, (0.01 / max(d1, d2))**0.2 if max(d1, d2) > 1e-15 else 1e-6)
    else:
        h = first_step
    h = min(h, max_step, abs(tf - t))
    status, message = 0, 'The solver successfully reached the end of the integration interval.'
    while direction * (tf-t) > 0:
        h = min(h, max_step, abs(tf-t))
        if h < 10 * np.finfo(float).eps * max(abs(t), 1.0):
            status, message = -1, 'Required step size became too small.'
            break
        dt = direction*h
        ks = [k0]
        for i in range(1, 6):
            v = y + dt * sum(_A[i][j] * ks[j] for j in range(i))
            ks.append(f(t + _C[i]*dt, v))
        y_new = y + dt * sum(_B[j]*ks[j] for j in range(6))
        ks.append(f(t+dt, y_new))
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = dt * sum(_E[j]*ks[j] for j in range(7))
        err_norm = np.linalg.norm(err / scale) / np.sqrt(len(y))
        if err_norm <= 1:
            old_t, old_y = t, y
            t, y, k0 = t+dt, y_new, ks[-1]
            if desired is None:
                ts.append(t)
                ys.append(y.copy())
            else:
                q = None
                while idx < len(desired) and direction*(desired[idx]-t) <= 0:
                    if q is None:
                        q = np.stack(ks, axis=0).T @ _P
                    x = (desired[idx] - old_t) / dt
                    value = old_y + dt * (q @ np.array([x, x*x, x*x*x, x*x*x*x]))
                    ts.append(desired[idx])
                    ys.append(value)
                    idx += 1
            factor = 10 if err_norm == 0 else min(10, max(0.2, 0.9*err_norm**(-0.2)))
            h *= factor
        else:
            h *= max(0.2, 0.9*err_norm**(-0.2))
    return SimpleNamespace(
        t=np.asarray(ts, dtype=float),
        y=np.stack(ys, axis=1) if ys else np.empty((len(y), 0)),
        success=(status == 0), status=status, message=message,
        nfev=nfev, njev=0, nlu=0,
    )
