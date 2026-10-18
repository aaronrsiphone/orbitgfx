"""Desktop stand-in for Pythonista's `scene` module.

* Classic drawing (fill/stroke/rect/ellipse/line) -> Skia, y-up, at SCALE.
* Shader SpriteNodes -> real GLSL on an OpenGL context (moderngl; EGL headless).
* Node tree is composited AFTER classic draw, by z_position, like Pythonista.
* Framework bookkeeping is emulated: t, dt, touches, size, bounds, and
  Pythonista's touch dispatch (which reads self.touches[touch_id].layer).
* STRICT: assigning a framework-owned Scene attribute from your code raises
  SandboxError (catches `self.dt = ...` / `self.touches = {}` bugs).

* EffectNode: shader-sprite children render at their own size x SCALE (as on
  the device) and are drawn scaled by the effect node's x_scale / y_scale with
  linear filtering, so a half-size child gives a half-resolution pass.

Not emulated: Actions, ShapeNode, sounds, real fonts (DejaVu Sans Mono stands
in for Menlo), EffectNode shaders and crop_rect.
"""
import math
import warnings

import numpy as np

try:
    import skia
except ImportError:            # still usable for logic tests without rendering
    skia = None

SCALE = 3.0
STRICT = True
FILTERING_NEAREST = 0          # SpriteKit's SKTextureFilteringMode values
FILTERING_LINEAR = 1
OWNED = ('t', 'dt', 'touches', 'size', 'bounds', 'view')
# Node / Scene properties you may set, but that Pythonista stores as floats
FLOAT_PROPS = ('speed', 'alpha', 'rotation', 'x_scale', 'y_scale', 'z_position')


class SandboxError(RuntimeError):
    pass


# ------------------------------------------------------------------ geometry
class Point(tuple):
    def __new__(cls, x=0.0, y=0.0):
        return tuple.__new__(cls, (float(x), float(y)))

    x = property(lambda s: s[0])
    y = property(lambda s: s[1])

    def __add__(self, o):
        return Point(self[0] + o[0], self[1] + o[1])

    def __sub__(self, o):
        return Point(self[0] - o[0], self[1] - o[1])


Vector2 = Point


class Size(tuple):
    def __new__(cls, w=0.0, h=0.0):
        return tuple.__new__(cls, (float(w), float(h)))

    w = property(lambda s: s[0])
    h = property(lambda s: s[1])
    width = w
    height = h


class Rect(tuple):
    def __new__(cls, x=0.0, y=0.0, w=0.0, h=0.0):
        return tuple.__new__(cls, (float(x), float(y), float(w), float(h)))

    x = property(lambda s: s[0]); y = property(lambda s: s[1])
    w = property(lambda s: s[2]); h = property(lambda s: s[3])
    width = w; height = h

    def contains_point(self, p):
        return self[0] <= p[0] <= self[0] + self[2] and self[1] <= p[1] <= self[1] + self[3]


class Touch(object):
    def __init__(self, x, y, prev_x, prev_y, touch_id):
        self.location = Point(x, y)
        self.prev_location = Point(prev_x, prev_y)
        self.touch_id = touch_id
        self.timestamp = 0.0
        self.layer = None


# ------------------------------------------------------------------ classic drawing state
_canvas = None
_H = 0.0
_fill = (1, 1, 1, 1)
_stroke = None
_sw = 1.0
stats = {'calls': 0}


def get_screen_scale():
    return SCALE


def get_screen_size():
    return Size(390, 844)


def _paint(c, style, w=1.0):
    p = skia.Paint(AntiAlias=True, Color=skia.Color4f(*c).toColor(), Style=style)
    if style == skia.Paint.kStroke_Style:
        p.setStrokeWidth(w)
    return p


def _c4(args):
    if len(args) == 1:
        args = args[0]
    args = tuple(float(a) for a in args)
    return args + (1.0,) * (4 - len(args)) if len(args) < 4 else args[:4]


def fill(*c):
    global _fill
    _fill = _c4(c)


def no_fill():
    global _fill
    _fill = None


def stroke(*c):
    global _stroke
    _stroke = _c4(c)


def no_stroke():
    global _stroke
    _stroke = None


def stroke_weight(w):
    global _sw
    _sw = float(w)


def background(*c):
    rect_c = _c4(c)
    if _canvas is not None:
        _canvas.clear(skia.Color4f(*rect_c).toColor())


def _y(y):
    return _H - y


def rect(x, y, w, h):
    stats['calls'] += 1
    if _canvas is None:
        return
    r = skia.Rect.MakeXYWH(x, _y(y + h), w, h)
    if _fill:
        _canvas.drawRect(r, _paint(_fill, skia.Paint.kFill_Style))
    if _stroke:
        _canvas.drawRect(r, _paint(_stroke, skia.Paint.kStroke_Style, _sw))


def ellipse(x, y, w, h):
    stats['calls'] += 1
    if _canvas is None:
        return
    r = skia.Rect.MakeXYWH(x, _y(y + h), w, h)
    if _fill:
        _canvas.drawOval(r, _paint(_fill, skia.Paint.kFill_Style))
    if _stroke:
        _canvas.drawOval(r, _paint(_stroke, skia.Paint.kStroke_Style, _sw))


def line(x1, y1, x2, y2):
    stats['calls'] += 1
    if _canvas is None or not _stroke:
        return
    p = _paint(_stroke, skia.Paint.kStroke_Style, _sw)
    p.setStrokeCap(skia.Paint.kRound_Cap)
    _canvas.drawLine(x1, _y(y1), x2, _y(y2), p)


# ------------------------------------------------------------------ GPU
_ctx = None


def _gl():
    global _ctx
    if _ctx is None:
        import moderngl
        try:
            _ctx = moderngl.create_standalone_context(backend='egl')
        except Exception:
            _ctx = moderngl.create_standalone_context()
    return _ctx


_VS = ('#version 120\nattribute vec2 in_pos; varying vec2 v_tex_coord;\n'
       'void main(){ v_tex_coord = in_pos*0.5+0.5; gl_Position = vec4(in_pos,0.0,1.0);}')


class Texture(object):
    """Accepts a ui.Image (sandbox), a PIL image, or an image path."""

    def __init__(self, img):
        from PIL import Image
        if hasattr(img, 'pil'):
            img = img.pil
        elif isinstance(img, str):
            img = Image.open(img)
        self.pil = img.convert('RGBA')
        self.size = Size(*self.pil.size)
        self.filtering_mode = 1
        self._gl = None

    def _bind(self, unit=0):
        import moderngl
        from PIL import Image
        if self._gl is None:
            # SpriteKit convention: v = 0 is the BOTTOM row of the image.
            im = self.pil.transpose(Image.FLIP_TOP_BOTTOM)
            self._gl = _gl().texture(im.size, 4, im.tobytes())
        f = moderngl.NEAREST if self.filtering_mode == FILTERING_NEAREST else moderngl.LINEAR
        self._gl.filter = (f, f)
        self._gl.use(unit)


_PROGRAMS = {}


class Shader(object):
    """Compiles at creation (like the device) so GLSL errors surface early."""

    def __init__(self, src):
        self.src = src
        self.uniforms = {}
        self._prog = None
        try:
            from orbitgfx.shaders import lint_glsl
            for w in lint_glsl(src):
                warnings.warn('GLSL lint: ' + w)
        except ImportError:
            pass
        try:
            # one program per distinct source: llvmpipe programs with long loops are
            # large, and tests build hundreds of identical shaders
            self._prog = _PROGRAMS.get(src)
            if self._prog is None:
                fs = '#version 120\n' + src.replace('precision highp float;', '')
                self._prog = _PROGRAMS[src] = _gl().program(vertex_shader=_VS, fragment_shader=fs)
        except ImportError:
            warnings.warn('moderngl not installed: shaders will not render')
        except Exception as e:
            raise SandboxError('GLSL compile failed:\n%s' % e)

    def set_uniform(self, name, value):
        if isinstance(value, Texture):
            self.uniforms[name] = value
            return
        if isinstance(value, (tuple, list)):
            if not 2 <= len(value) <= 4:
                raise SandboxError('set_uniform(%r): tuples must have 2-4 items' % name)
            value = tuple(float(v) for v in value)
        else:
            value = float(value)
        if any(math.isnan(v) or math.isinf(v) for v in np.atleast_1d(value)):
            raise SandboxError('set_uniform(%r) got NaN/inf: %r' % (name, value))
        self.uniforms[name] = value

    def get_uniform(self, name):
        return self.uniforms.get(name)


# ------------------------------------------------------------------ nodes
class Node(object):
    def __init__(self, position=(0, 0), z_position=0.0, scale=1.0, x_scale=1.0,
                 y_scale=1.0, alpha=1.0, speed=1.0, parent=None):
        self.children = []
        self.parent = None
        self.position = Point(*position)
        self.z_position = z_position
        self.alpha = alpha
        self.x_scale, self.y_scale = x_scale, y_scale
        self.rotation = 0.0
        self.paused = False
        if parent is not None:
            parent.add_child(self)

    def add_child(self, node):
        if node.parent is not None:
            node.remove_from_parent()
        node.parent = self
        self.children.append(node)

    def remove_from_parent(self):
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent = None

    def run_action(self, *a, **k):
        pass

    def remove_all_actions(self):
        pass


class SpriteNode(Node):
    def __init__(self, texture=None, color='white', blend_mode=0, position=(0, 0),
                 z_position=0.0, alpha=1.0, size=None, parent=None, **kw):
        super(SpriteNode, self).__init__(position=position, z_position=z_position,
                                         alpha=alpha, parent=parent)
        if isinstance(texture, str):
            texture = Texture(texture)
        self.texture = texture
        self.size = Size(*(size or (texture.size if texture else (0, 0))))
        self.anchor_point = Point(0.5, 0.5)
        self.shader = None
        self.color = color
        self.blend_mode = blend_mode


class EffectNode(Node):
    def __init__(self, position=(0, 0), z_position=0.0, parent=None, **kw):
        super(EffectNode, self).__init__(position=position, z_position=z_position, parent=parent)
        self.shader = None
        self.effects_enabled = True
        self.crop_rect = None


class LabelNode(SpriteNode):
    def __init__(self, text='', font=('Helvetica', 20), color='white', position=(0, 0),
                 z_position=0.0, parent=None, **kw):
        super(LabelNode, self).__init__(None, position=position, z_position=z_position,
                                        parent=parent)
        self.text = text
        self.font = font
        self.color = color


# ------------------------------------------------------------------ Scene
class Scene(Node):
    """Framework-owned attributes raise SandboxError if your code assigns them."""

    def __init__(self, *a, **k):
        object.__setattr__(self, '_fw', True)
        super(Scene, self).__init__()
        self.size = Size(390, 844)
        self.bounds = Rect(0, 0, 390, 844)
        self.t = 0.0
        self.dt = 0.0
        self.touches = {}
        self.view = None
        self.background_color = (0, 0, 0)
        object.__setattr__(self, '_fw', False)

    def __setattr__(self, name, value):
        if STRICT and name in OWNED and not self.__dict__.get('_fw', True):
            raise SandboxError(
                'your code assigned self.%s, but Pythonista owns Scene.%s and will '
                'overwrite or depend on it. Rename your attribute.' % (name, name))
        if name in FLOAT_PROPS and isinstance(value, (int, float)):
            value = float(value)          # Node properties are CGFloat on the device
        object.__setattr__(self, name, value)

    def _fw_set(self, **kw):
        object.__setattr__(self, '_fw', True)
        try:
            for k, v in kw.items():
                setattr(self, k, v)
        finally:
            object.__setattr__(self, '_fw', False)

    # user callbacks
    def setup(self): pass
    def update(self): pass
    def draw(self): pass
    def touch_began(self, touch): pass
    def touch_moved(self, touch): pass
    def touch_ended(self, touch): pass
    def did_change_size(self): pass
    def stop(self): pass
    def pause(self): pass
    def resume(self): pass

    # framework internals (mirror pythonista/scene.py)
    def _touch_began(self, touch):
        touch.layer = self
        self.touches[touch.touch_id] = touch
        self.touch_began(touch)

    def _touch_moved(self, touch):
        old_touch = self.touches[touch.touch_id]
        touch.layer = old_touch.layer
        self.touches[touch.touch_id] = touch
        self.touch_moved(touch)

    def _touch_ended(self, touch):
        old_touch = self.touches[touch.touch_id]
        touch.layer = old_touch.layer
        del self.touches[touch.touch_id]
        self.touch_ended(touch)


def run(scene_obj, *a, **kw):
    """On device this presents the scene. In the sandbox use sandbox.Runner."""
    pass


# ------------------------------------------------------------------ compositor
def _walk(node):
    for c in node.children:
        yield c
        if isinstance(c, EffectNode):          # composited as one layer
            continue
        for g in _walk(c):
            yield g


def _render_sprite(n, u_time):
    import moderngl
    W, H = max(1, int(round(n.size[0] * SCALE))), max(1, int(round(n.size[1] * SCALE)))
    sh = n.shader
    prog = sh._prog
    if prog is None:
        return None
    ctx = _gl()
    if n.texture is not None:
        n.texture._bind(0)
    builtins = {'u_time': u_time, 'u_texture': 0, 'u_scale': SCALE,
                'u_sprite_size': (float(n.size[0]), float(n.size[1])), 'u_offset': (0.0, 0.0)}
    unit = 1                                   # each set_uniform texture gets its own unit
    for k, v in list(builtins.items()) + list(sh.uniforms.items()):
        if k in prog:
            if isinstance(v, Texture):
                v._bind(unit)
                v = unit
                unit += 1
            prog[k].value = v
    fbo = ctx.simple_framebuffer((W, H), components=4)
    fbo.use()
    fbo.clear(0, 0, 0, 0)
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], 'f4').tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')
    vao.render(moderngl.TRIANGLE_STRIP)
    arr = np.frombuffer(fbo.read(components=4, dtype='f1'), np.uint8).reshape(H, W, 4)[::-1]
    vao.release(); vbo.release(); fbo.release()
    return np.ascontiguousarray(arr)          # premultiplied RGBA


def _color(c):
    if isinstance(c, str):
        return {'white': (1, 1, 1, 1), 'black': (0, 0, 0, 1)}.get(c, (1, 1, 1, 1))
    c = tuple(c)
    return c + (1.0,) * (4 - len(c))


def render(scene_obj, path=None, u_time=None, font_family='DejaVu Sans Mono'):
    """Run one draw() into a PNG: classic layer, then nodes by z_position."""
    global _canvas, _H
    if skia is None:
        raise SandboxError('pip install skia-python to render')
    w, h = scene_obj.size
    _H = h
    surf = skia.Surface(int(w * SCALE), int(h * SCALE))
    _canvas = surf.getCanvas()
    _canvas.clear(skia.Color4f(*_color(scene_obj.background_color)).toColor())
    _canvas.scale(SCALE, SCALE)
    stats['calls'] = 0
    try:
        scene_obj.draw()
        u_time = scene_obj.t if u_time is None else u_time
        nodes = sorted(_walk(scene_obj), key=lambda n: n.z_position)  # stable: tree order
        for n in nodes:
            if isinstance(n, LabelNode):
                if not n.text:
                    continue
                sz = n.font[1]
                bold = 'Bold' in n.font[0]
                tf = skia.Typeface(font_family, skia.FontStyle.Bold() if bold else skia.FontStyle.Normal())
                f = skia.Font(tf, sz)
                tw = f.measureText(n.text)
                x = n.position[0] - n.anchor_point[0] * tw
                y = _y(n.position[1]) + sz * 0.35
                c = _color(n.color)
                _canvas.drawString(n.text, x, y, f,
                                   skia.Paint(AntiAlias=True, Color=skia.Color4f(*c).toColor()))
            elif isinstance(n, EffectNode):
                lin = skia.SamplingOptions(skia.FilterMode.kLinear)
                for c in sorted(n.children, key=lambda k: k.z_position):
                    if not (isinstance(c, SpriteNode) and c.shader is not None):
                        continue
                    arr = _render_sprite(c, u_time)
                    if arr is None:
                        continue
                    img = skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType,
                                               alphaType=skia.kPremul_AlphaType)
                    sx, sy = n.x_scale, n.y_scale
                    cw, ch = c.size[0] * sx, c.size[1] * sy
                    ccx = n.position[0] + c.position[0] * sx
                    ccy = n.position[1] + c.position[1] * sy
                    _canvas.drawImageRect(img, skia.Rect.MakeXYWH(ccx - cw / 2, _y(ccy + ch / 2), cw, ch), lin)
            elif isinstance(n, SpriteNode) and n.shader is not None:
                arr = _render_sprite(n, u_time)
                if arr is None:
                    continue
                img = skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType,
                                           alphaType=skia.kPremul_AlphaType)
                x = n.position[0] - n.anchor_point[0] * n.size[0]
                ytop = n.position[1] + (1 - n.anchor_point[1]) * n.size[1]
                _canvas.drawImageRect(img, skia.Rect.MakeXYWH(x, _y(ytop), n.size[0], n.size[1]))
            elif isinstance(n, SpriteNode) and n.texture is not None:
                arr = np.asarray(n.texture.pil)
                img = skia.Image.fromarray(np.ascontiguousarray(arr), colorType=skia.kRGBA_8888_ColorType)
                x = n.position[0] - n.anchor_point[0] * n.size[0]
                ytop = n.position[1] + (1 - n.anchor_point[1]) * n.size[1]
                _canvas.drawImageRect(img, skia.Rect.MakeXYWH(x, _y(ytop), n.size[0], n.size[1]))
        snap = surf.makeImageSnapshot()
        if path:
            snap.save(path, skia.kPNG)
        return snap
    finally:
        _canvas = None
