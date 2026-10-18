"""Catch the Pythonista bug class that bit mission control twice.

scene.Scene keeps its own state in ordinary attributes (`dt`, `t`, `touches`,
`size`, ...). Assigning your own value to one of them silently breaks the
framework -- e.g. `self.dt = sample_spacing` is overwritten with the frame
time (0 on the first frame), and `self.touches = {}` crashes touch dispatch.

    from orbitgfx import safety
    safety.assert_clean(MyScene)      # raises with line numbers if you collide

Works on device (reads your source with inspect) and in the sandbox, which
additionally blocks such assignments at runtime.
"""
import inspect
import re

# Framework-owned state. Assigning these from your code is almost always a bug.
OWNED = frozenset((
    't', 'dt', 'touches', 'size', 'bounds', 'view', 'root_layer',
    'presented_scene', 'presenting_scene', 'children', 'parent', 'scene',
    'frame', 'bbox',
))

# Node / Scene properties you are SUPPOSED to set.
INTENDED = frozenset((
    'background_color', 'position', 'z_position', 'alpha', 'rotation', 'speed',
    'x_scale', 'y_scale', 'paused', 'effects_enabled', 'shader', 'blend_mode',
    'anchor_point', 'crop_rect', 'fixed_time_step',
))

# Callbacks and methods: defining them is fine, assigning attributes is not.
METHODS = frozenset((
    'setup', 'update', 'draw', 'touch_began', 'touch_moved', 'touch_ended',
    'did_change_size', 'did_evaluate_actions', 'pause', 'resume', 'stop',
    'present_modal_scene', 'dismiss_modal_scene', 'add_child', 'remove_from_parent',
    'run_action', 'remove_action', 'remove_all_actions', 'point_to_scene',
    'point_from_scene', 'render_to_texture',
))

_ASSIGN = re.compile(r'\bself\.([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)?(?:[-+*/%]?=)(?!=)')


# Node properties stored as floats (CGFloat): `TABLE[self.speed]` breaks on the device
_FLOAT_INDEX = re.compile(r'\[\s*self\.(speed|alpha|rotation|x_scale|y_scale|z_position)\s*\]')


def collisions(cls_or_obj):
    """[(lineno, attr, kind, source_line)] for assignments that hit Scene state."""
    cls = cls_or_obj if inspect.isclass(cls_or_obj) else type(cls_or_obj)
    out = []
    for klass in cls.__mro__:
        if klass.__module__ in ('scene', 'pythonista.scene', '_scene2', 'builtins'):
            continue
        try:
            lines, start = inspect.getsourcelines(klass)
        except (OSError, TypeError):
            continue
        for i, src in enumerate(lines):
            code = src.split('#', 1)[0]
            for m in _ASSIGN.finditer(code):
                name = m.group(1)
                if name in OWNED:
                    out.append((start + i, name, 'framework state', src.strip()))
                elif name in METHODS:
                    out.append((start + i, name, 'shadows a Scene method', src.strip()))
            for m in _FLOAT_INDEX.finditer(code):
                out.append((start + i, m.group(1), 'Node property, a float on the device: '
                            'not usable as an index', src.strip()))
    return out


def assert_clean(cls_or_obj):
    bad = collisions(cls_or_obj)
    if bad:
        msg = '\n'.join('  line %d: self.%s (%s): %s' % b for b in bad)
        raise RuntimeError('Scene attribute collisions:\n' + msg +
                           '\nRename these attributes (e.g. self.dt -> self.step_s).')
