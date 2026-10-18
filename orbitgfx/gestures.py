"""Touch routing for Pythonista scenes: regions, taps, double taps, drags, pinch.

Forward your Scene's touch_began/moved/ended to began/moved/ended and call
poll(now) from update(). Never store touches in `self.touches` -- Scene owns
that dict (see safety.py).

Callbacks (assign any you need; all optional):
    on_tap(region)                 single tap (deferred if region allows double tap)
    on_double_tap(region)
    on_press(region, now)          finger down (single-finger)
    on_drag(region, dx, dy, now)   single-finger move
    on_release(region, now, was_tap)
    on_pinch_begin(region)
    on_pinch(region, ratio)        current distance / starting distance
    on_pinch_end(region)
"""
import math
import time


class Gestures(object):

    def __init__(self, regions, double_tap_regions=(), pinch_regions=(),
                 double_tap=0.30, tap_slop=10.0, tap_time=0.45):
        """regions: callable() -> [(name, (x, y, w, h)), ...] checked in order."""
        self.regions = regions
        self.double_tap_regions = set(double_tap_regions)
        self.pinch_regions = set(pinch_regions)
        self.double_tap = double_tap
        self.tap_slop = tap_slop
        self.tap_time = tap_time
        self.fingers = {}
        self.region = None
        self._t0 = 0.0
        self._travel = 0.0
        self._pinch = None
        self._pending = None          # (time, region)
        for name in ('on_tap', 'on_double_tap', 'on_press', 'on_drag', 'on_release',
                     'on_pinch_begin', 'on_pinch', 'on_pinch_end'):
            setattr(self, name, None)

    def region_at(self, loc):
        for name, (x, y, w, h) in self.regions():
            if x <= loc[0] <= x + w and y <= loc[1] <= y + h:
                return name
        return None

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _call(self, name, *args):
        fn = getattr(self, name)
        if fn is not None:
            return fn(*args)

    @property
    def active(self):
        return bool(self.fingers)

    def began(self, touch, now=None):
        now = time.monotonic() if now is None else now
        self.fingers[touch.touch_id] = tuple(touch.location)
        if len(self.fingers) == 2:
            a, b = list(self.fingers.values())
            ra, rb = self.region_at(a), self.region_at(b)
            if ra == rb and ra in self.pinch_regions:
                self._pinch = (max(10.0, self._dist(a, b)), ra)
                self._call('on_pinch_begin', ra)
            self._travel = 1e9             # a two-finger gesture is never a tap
            return
        if len(self.fingers) > 2:
            return
        self.region = self.region_at(touch.location)
        self._t0 = now
        self._travel = 0.0
        self._call('on_press', self.region, now)

    def moved(self, touch, now=None):
        now = time.monotonic() if now is None else now
        self.fingers[touch.touch_id] = tuple(touch.location)
        if self._pinch is not None and len(self.fingers) == 2:
            a, b = list(self.fingers.values())
            self._call('on_pinch', self._pinch[1], self._dist(a, b) / self._pinch[0])
            return
        if len(self.fingers) > 1:
            return
        dx = touch.location[0] - touch.prev_location[0]
        dy = touch.location[1] - touch.prev_location[1]
        self._travel += abs(dx) + abs(dy)
        self._call('on_drag', self.region, dx, dy, now)

    def ended(self, touch, now=None):
        now = time.monotonic() if now is None else now
        self.fingers.pop(touch.touch_id, None)
        if len(self.fingers) < 2 and self._pinch is not None:
            self._call('on_pinch_end', self._pinch[1])
            self._pinch = None
        if self.fingers:
            return
        tap = self._travel < self.tap_slop and now - self._t0 <= self.tap_time
        self._call('on_release', self.region, now, tap)
        if not tap:
            return
        region = self.region
        if self._pending is not None:
            pt, pr = self._pending
            self._pending = None
            if pr == region and region in self.double_tap_regions:
                self._call('on_double_tap', region)
                return
            self._call('on_tap', pr)
        if region in self.double_tap_regions:
            self._pending = (now, region)
        else:
            self._call('on_tap', region)

    def poll(self, now=None):
        """Call every frame: fires a deferred single tap once the window passes."""
        now = time.monotonic() if now is None else now
        if self._pending is not None and now - self._pending[0] > self.double_tap:
            _, region = self._pending
            self._pending = None
            self._call('on_tap', region)
