"""GLSL ES 1.0 fragment shaders for Pythonista SpriteNodes.

`build_globe_shader(features, palette)` assembles the globe shader from a
template: every visual layer is behind a `#define`, and every colour is a
palette entry, so you can reuse the renderer for a different look or a
different mission without touching GLSL.

Pythonista supplies: v_tex_coord, u_texture (the sprite's texture), u_time,
u_sprite_size, u_scale, u_offset. Everything else is set by GlobeView.

Rules that keep shaders working on iOS (see lint_glsl):
  * `precision highp float;` first line.
  * never pow(x, y) with x < 0 (undefined; NaN on Apple GPUs) -- square with x*x.
  * no dynamic indexing of uniform arrays; use one uniform per slot.
  * output premultiplied alpha in gl_FragColor.
"""
import re

STATION_SLOTS = 6

FEATURES = ('STARS', 'ATMOSPHERE', 'ORBIT', 'CLOUDS', 'LIGHTS', 'AURORA',
            'GRATICULE', 'FOOTPRINT', 'STATIONS', 'SATELLITE', 'BIOMES', 'GLINT')

PALETTE = {
    'BGC':   (0.023, 0.039, 0.078),   # panel background (trail fades into it)
    'CYAN':  (0.34, 0.96, 0.82),      # orbit / trail
    'GOLD':  (1.0, 0.78, 0.36),       # spacecraft, footprint, RF link
    'ECLC':  (0.55, 0.50, 0.98),      # eclipse colour (orbit in shadow)
    'TEXTC': (0.66, 0.78, 0.87),      # graticule, idle stations
    'NIGHT': (0.006, 0.014, 0.032),   # unlit surface
    'DEEP':  (0.010, 0.045, 0.105),   # deep ocean
    'OCEAN': (0.050, 0.230, 0.400),   # sunlit ocean
    'COAST': (0.34, 0.90, 0.76),      # coastline glow
    'LIGHT': (1.0, 0.76, 0.42),       # city lights
}

GLOBE_TEMPLATE = r'''
precision highp float;
varying vec2 v_tex_coord;
uniform sampler2D u_texture;
uniform float u_time;
uniform vec2 u_size;
uniform float u_er;
uniform float u_pxs;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_fwd;
uniform vec3 u_sun;
uniform vec2 u_rot;
uniform vec3 u_sat;
uniform vec3 u_e2;
uniform float u_trail;
uniform float u_cloud;
uniform float u_ctime;
uniform vec2 u_cam;
uniform vec4 u_st0;
uniform vec4 u_st1;
uniform vec4 u_st2;
uniform vec4 u_st3;
uniform vec4 u_st4;
uniform vec4 u_st5;

const float PI = 3.14159265;
{PALETTE}

vec4 acc;
void over(vec3 c, float a) {
    a = clamp(a, 0.0, 1.0);
    acc = vec4(c * a + acc.rgb * (1.0 - a), a + acc.a * (1.0 - a));
}
float lineA(float d, float w) { return clamp((0.5 * w - d) / u_pxs + 0.5, 0.0, 1.0); }
float segd(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
    return length(pa - ba * h);
}
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1.0, 0.0, 0.0)), f.x),
                   mix(hash(i + vec3(0.0, 1.0, 0.0)), hash(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
               mix(mix(hash(i + vec3(0.0, 0.0, 1.0)), hash(i + vec3(1.0, 0.0, 1.0)), f.x),
                   mix(hash(i + vec3(0.0, 1.0, 1.0)), hash(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * vnoise(p);
        p = p * 2.03 + vec3(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s;
}
bool inShadow(vec3 c) {
    float d = dot(c, u_sun);
    return d < 0.0 && length(c - d * u_sun) < 1.0;
}
vec2 proj(vec3 w) { return vec2(dot(w, u_right), dot(w, u_up)); }

vec3 station(vec3 c, vec3 nw, vec4 st) {
    if (dot(st.xyz, st.xyz) < 0.5) return c;
    float d = acos(clamp(dot(nw, st.xyz), -1.0, 1.0)) * u_er;
    float on = st.w;
    float ring = lineA(abs(d - 4.5), 1.0) * on * (0.6 + 0.4 * sin(u_time * 5.0));
    c = mix(c, GOLD, ring);
    return mix(c, mix(TEXTC, GOLD, on), lineA(d, 4.0) * (0.65 + 0.35 * on));
}
void link(vec2 P, vec4 st, vec2 ps, bool satVis) {
    if (st.w < 0.5 || !satVis) return;
    if (dot(st.xyz, u_fwd) <= 0.0) return;
    vec2 a = proj(st.xyz);
    float d = segd(P, a, ps) * u_er;
    float t = clamp(dot(P - a, ps - a) / max(dot(ps - a, ps - a), 1e-9), 0.0, 1.0);
    float pulse = 0.55 + 0.45 * sin(t * 40.0 - u_time * 9.0);
    over(GOLD, lineA(d, 1.3) * 0.85 * pulse);
}

void main() {
    acc = vec4(0.0);
    vec2 pt = (v_tex_coord - 0.5) * u_size;
    vec2 P = pt / u_er;
    float r = length(P);
    vec3 sunc = vec3(dot(u_sun, u_right), dot(u_sun, u_up), dot(u_sun, u_fwd));

#ifdef STARS
    vec2 sp = pt + u_cam * vec2(-160.0, 160.0);
    vec2 cell = floor(sp / 7.0);
    float h = hash(vec3(cell, 3.7));
    if (h > 0.972) {
        vec2 o = (vec2(hash(vec3(cell, 1.3)), hash(vec3(cell, 8.1))) * 0.7 + 0.15) * 7.0;
        float d = length(sp - cell * 7.0 - o);
        float tw = 0.65 + 0.35 * sin(u_time * (1.0 + 3.0 * h) + h * 50.0);
        over(vec3(0.85, 0.92, 1.0), lineA(d, 0.9 + 1.4 * fract(h * 37.0)) * tw * 0.8);
    }
#endif

#ifdef ATMOSPHERE
    vec2 dir = r > 1e-5 ? P / r : vec2(0.0);
    float nd = dot(dir, sunc.xy);
    float lit = smoothstep(-0.30, 0.45, nd + 0.25 * sunc.z);
    float twx = nd / 0.25;
    float tw = exp(-twx * twx);
    float behind = pow(max(-sunc.z, 0.0), 2.0);
    float hgt = max(r - 1.0, 0.0);
    float shell = exp(-hgt / 0.028);
    vec3 atmc = mix(vec3(0.28, 0.68, 1.0), vec3(1.0, 0.52, 0.26), clamp(0.6 * tw + 0.6 * behind, 0.0, 0.85));
    float aatm = shell * (0.08 + 0.70 * lit + 0.30 * tw + 0.85 * behind * (0.35 + 0.65 * smoothstep(-0.5, 0.9, nd)));
    over(atmc, min(aatm, 0.95));
#endif

    float R = max(length(u_sat), 1.0001);
    vec3 e1 = length(u_sat) > 0.5 ? u_sat / length(u_sat) : vec3(1.0, 0.0, 0.0);
#ifdef ORBIT
    vec2 a2 = proj(e1) * R;
    vec2 b2 = proj(u_e2) * R;
    float det = a2.x * b2.y - a2.y * b2.x;
    float phi;
    if (abs(det) > 1e-4) {
        vec2 q = vec2(b2.y * P.x - b2.x * P.y, -a2.y * P.x + a2.x * P.y) / det;
        phi = atan(q.y, q.x);
    } else {
        phi = atan(dot(P, b2), dot(P, a2));
    }
    for (int i = 0; i < 4; i++) {
        vec2 c = a2 * cos(phi) + b2 * sin(phi);
        vec2 d1 = -a2 * sin(phi) + b2 * cos(phi);
        phi -= dot(c - P, d1) / max(dot(d1, d1), 1e-6);
    }
    vec2 cp = a2 * cos(phi) + b2 * sin(phi);
    float od = length(cp - P) * u_er;
    vec3 C = (e1 * cos(phi) + u_e2 * sin(phi)) * R;
    float cdepth = dot(C, u_fwd);
    phi = mod(phi + PI, 2.0 * PI) - PI;
    float frac = u_trail > 1e-4 ? 1.0 + phi / u_trail : -1.0;
    bool inTrail = phi <= 0.0 && frac >= 0.0;
    vec3 oc = inShadow(C) ? ECLC : CYAN;
    float ow = inTrail ? 1.1 + 1.8 * frac : 1.15;
    vec3 ocol = inTrail ? mix(oc, BGC, 0.70 * (1.0 - frac)) : mix(oc, BGC, cdepth < 0.0 ? 0.62 : 0.36);
    float oa = lineA(od, ow) * (inTrail ? 1.0 : 0.9);
    if (cdepth < 0.0) over(ocol, oa);
#endif

    if (r < 1.0 + 2.0 * u_pxs / u_er) {
        float rr = min(r, 1.0);
        float z = sqrt(max(1.0 - rr * rr, 0.0));
        vec3 nc = vec3(P / max(r, 1e-6) * rr, z);
        vec3 nw = u_right * nc.x + u_up * nc.y + u_fwd * nc.z;
        vec3 ne = vec3(u_rot.x * nw.x + u_rot.y * nw.y, -u_rot.y * nw.x + u_rot.x * nw.y, nw.z);
        float lon = atan(ne.y, ne.x);
        float lat = asin(clamp(ne.z, -1.0, 1.0));
        vec3 tx = texture2D(u_texture, vec2(lon / (2.0 * PI) + 0.5, lat / PI + 0.5)).rgb;
        float land = tx.r;
        float coast = tx.g;
        float lights = tx.b;

        float mu = dot(nw, u_sun);
        float day = smoothstep(-0.07, 0.11, mu);
        float lam = max(mu, 0.0);
        float dark = 1.0 - smoothstep(-0.16, 0.03, mu);
        float alat = abs(lat) * 57.2958;

#ifdef BIOMES
        float vary = vnoise(ne * 9.0);
        float desert = smoothstep(12.0, 20.0, alat) * (1.0 - smoothstep(33.0, 42.0, alat));
        vec3 landc = mix(vec3(0.13, 0.30, 0.19), vec3(0.50, 0.43, 0.29), clamp(desert * (0.35 + 0.95 * vary), 0.0, 1.0));
        landc = mix(landc, vec3(0.20, 0.33, 0.24), smoothstep(48.0, 60.0, alat));
        landc = mix(landc, vec3(0.86, 0.91, 0.96), smoothstep(63.0, 72.0, alat));
        vec3 sea = mix(DEEP, OCEAN, 0.15 + 0.85 * lam);
        sea = mix(sea, vec3(0.80, 0.86, 0.92) * (0.2 + 0.8 * lam), smoothstep(77.0, 84.0, alat) * 0.85);
#else
        vec3 landc = vec3(0.13, 0.30, 0.19);
        vec3 sea = mix(DEEP, OCEAN, 0.15 + 0.85 * lam);
#endif
        vec3 surf = mix(sea, landc * (0.30 + 1.05 * lam), land);
        vec3 col = mix(NIGHT, surf, day);

        float cl = 0.0;
#ifdef CLOUDS
        if (u_cloud > 0.0) {
            vec3 q = ne * 2.6 + vec3(u_ctime * 0.035, 0.0, u_ctime * 0.01);
            float w = fbm(q * 1.7 + vec3(4.0));
            float n = fbm(q + 0.85 * vec3(w, w * 0.7, -w));
            float band = 0.72 + 0.28 * cos(lat * 6.0);
            cl = smoothstep(0.50, 0.80, n * band + 0.05) * u_cloud;
            vec3 cc = vec3(0.93, 0.96, 1.0) * (0.10 + 0.95 * smoothstep(-0.05, 0.35, mu));
            col = mix(col, mix(vec3(0.02, 0.03, 0.05), cc, day), cl * 0.9);
        }
#endif

        float mux = mu / 0.05;
        col += vec3(0.65, 0.32, 0.10) * 0.11 * exp(-mux * mux) * (1.0 - cl);
#ifdef GLINT
        vec3 hv = normalize(sunc + vec3(0.0, 0.0, 1.0));
        float nh = max(dot(nc, hv), 0.0);
        float wet = day * (1.0 - land) * (1.0 - cl);
        col += vec3(0.80, 0.90, 1.0) * (0.95 * pow(nh, 220.0) + 0.10 * pow(nh, 16.0)) * wet;
#endif
        col = mix(col, COAST * mix(0.45, 1.0, day), coast * mix(0.30, 0.55, day) * (1.0 - 0.6 * cl));
#ifdef LIGHTS
        col += LIGHT * lights * dark * 1.35 * (1.0 - 0.75 * cl);
#endif
        col += vec3(0.20, 0.55, 1.0) * 0.38 * pow(1.0 - z, 2.5) * (0.20 + 0.80 * day);

#ifdef AURORA
        vec3 mp = vec3(0.0478, -0.1540, 0.9869);
        float ga = acos(clamp(abs(dot(ne, mp)), 0.0, 1.0));
        float gx = (ga - 0.36 - 0.025 * sin(lon * 3.0 + u_time * 0.3)) / 0.028;
        float oval = exp(-gx * gx);
        if (oval > 0.01) {
            float ray = vnoise(vec3(lon * 26.0, ga * 6.0, u_time * 0.7));
            float sheet = vnoise(ne * 6.0 + vec3(u_time * 0.12));
            float curtain = ray * ray * (0.4 + 1.2 * sheet);
            vec3 ac = mix(vec3(0.20, 1.0, 0.50), vec3(0.70, 0.35, 1.0), smoothstep(0.2, 1.4, -gx));
            col += ac * oval * (0.10 + 1.5 * curtain) * dark * 0.5;
        }
#endif

#ifdef GRATICULE
        float latd = lat * 57.2958;
        float dl = abs(latd - 30.0 * floor(latd / 30.0 + 0.5)) / 57.2958 * u_er;
        float lond = lon * 57.2958;
        float dn = abs(lond - 30.0 * floor(lond / 30.0 + 0.5)) / 57.2958 * cos(lat) * u_er;
        float g = max(alat < 75.0 ? lineA(dl, 0.7) : 0.0, lineA(dn, 0.7));
        col = mix(col, TEXTC, g * 0.12);
#endif

#ifdef FOOTPRINT
        float lamfp = acos(1.0 / R);
        float dfp = abs(acos(clamp(dot(nw, e1), -1.0, 1.0)) - lamfp) * u_er;
        col = mix(col, GOLD, lineA(dfp, 1.1) * 0.55);
        float inside = 1.0 - smoothstep(lamfp - 0.002, lamfp + 0.002, acos(clamp(dot(nw, e1), -1.0, 1.0)));
        col = mix(col, GOLD, inside * 0.03);
#endif

#ifdef STATIONS
        col = station(col, nw, u_st0);
        col = station(col, nw, u_st1);
        col = station(col, nw, u_st2);
        col = station(col, nw, u_st3);
        col = station(col, nw, u_st4);
        col = station(col, nw, u_st5);
#endif

        float cover = clamp((1.0 - r) * u_er / u_pxs + 0.5, 0.0, 1.0);
        over(col, cover);
    }

#ifdef ORBIT
    if (cdepth >= 0.0) over(ocol, oa);
#endif

#ifdef SATELLITE
    vec2 ps = proj(u_sat);
    bool satVis = dot(u_sat, u_fwd) >= 0.0 || dot(ps, ps) >= 1.0;
#ifdef STATIONS
    link(P, u_st0, ps, satVis);
    link(P, u_st1, ps, satVis);
    link(P, u_st2, ps, satVis);
    link(P, u_st3, ps, satVis);
    link(P, u_st4, ps, satVis);
    link(P, u_st5, ps, satVis);
#endif
    if (satVis) {
        vec3 sc = inShadow(u_sat) ? ECLC : GOLD;
        vec2 tip = proj(u_sat + u_e2 * 0.149);
        over(GOLD, lineA(segd(P, ps, tip) * u_er, 1.5) * 0.8);
        float d = length(P - ps) * u_er;
        over(sc, exp(-d * d / 60.0) * 0.45);
        over(sc, lineA(d, 6.0));
        over(vec3(1.0), lineA(d, 2.8));
    }
#endif
    gl_FragColor = acc;
}
'''

MAP_NIGHT_SHADER = r'''
precision highp float;
varying vec2 v_tex_coord;
uniform sampler2D u_texture;
uniform vec3 u_sun;
const float PI = 3.14159265;
void main() {
    float lon = (v_tex_coord.x - 0.5) * 2.0 * PI;
    float lat = (v_tex_coord.y - 0.5) * PI;
    vec3 n = vec3(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat));
    float se = dot(n, u_sun);
    float dark = 0.46 * smoothstep(0.015, -0.31, se);
    float ex = se / 0.012;
    float edge = exp(-ex * ex) * 0.10;
    float lights = texture2D(u_texture, v_tex_coord).b * smoothstep(-0.02, -0.12, se) * 0.9;
    vec3 c = vec3(0.0, 0.01, 0.03) * dark + vec3(1.0, 0.76, 0.42) * lights + vec3(1.0, 0.65, 0.35) * edge;
    gl_FragColor = vec4(c, clamp(dark + lights + edge, 0.0, 1.0));
}
'''


def build_globe_shader(features=None, palette=None):
    """Return GLSL for the globe.

    features: iterable of names from FEATURES (default: all).
              e.g. features=FEATURES minus {'ORBIT', 'SATELLITE'} for a bare globe.
    palette:  dict overriding PALETTE entries with (r, g, b) tuples.
    """
    feats = FEATURES if features is None else tuple(features)
    unknown = set(feats) - set(FEATURES)
    if unknown:
        raise ValueError('unknown features: %s' % ', '.join(sorted(unknown)))
    pal = dict(PALETTE)
    for k, v in (palette or {}).items():
        if k not in PALETTE:
            raise ValueError('unknown palette key: %s' % k)
        pal[k] = tuple(float(c) for c in v)
    consts = '\n'.join('const vec3 %-5s = vec3(%r, %r, %r);' % ((k,) + tuple(v))
                       for k, v in pal.items())
    src = GLOBE_TEMPLATE.replace('{PALETTE}', consts)
    head, rest = src.split('\n', 2)[1], src.split('\n', 2)[2]
    defines = ''.join('#define %s\n' % f for f in feats)
    return head + '\n' + defines + rest


def lint_glsl(src):
    """Cheap static checks for GLSL ES 1.0 hazards on iOS. Returns warnings."""
    warn = []
    lines = src.splitlines()
    first = next((l.strip() for l in lines if l.strip()), '')
    if not first.startswith('precision'):
        warn.append('first statement should be `precision highp float;`')
    safe = re.compile(r'^\s*(max\(|abs\(|clamp\(|exp\(|1\.0\s*-\s*z\b|nh\b|length\()')
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r'\bpow\(([^,]+),', line):
            if not safe.match(m.group(1)):
                warn.append('line %d: pow(%s, ...) -- base must be >= 0 on iOS' % (i, m.group(1).strip()))
        if re.search(r'\b(texture2DLod|textureLod|texture\()', line):
            warn.append('line %d: not available in GLSL ES 1.0 fragment shaders' % i)
        if re.search(r'#version', line):
            warn.append('line %d: remove #version; Pythonista supplies it' % i)
    if 'gl_FragColor' not in src:
        warn.append('shader never writes gl_FragColor')
    return warn


# =========================================================================
# orbitgfx 2.0 ray core (quality LOW / MED / HIGH).  GLOBE_TEMPLATE above is
# the LEGACY preset and is never edited, so LEGACY stays pixel-identical.
#
# One primary ray per pixel (orthographic or pinhole), analytic sphere and
# orbit-circle tests, ray-depth ordering. The shader can be built as one pass
# or split in two:
#   radiance pass  (DO_RAD):  atmosphere + Earth surface; may run at reduced
#                             resolution inside an EffectNode
#   overlay pass   (DO_OVL):  stars, graticule, footprint, stations, orbit
#                             ring, RF link, spacecraft; always full resolution
# =========================================================================

RAY_FEATURES = FEATURES + ('TONEMAP', 'SCATTER', 'WEATHER', 'VCLOUDS', 'PENUMBRA', 'GGX', 'CRAFTLIGHT', 'SKYREFLECT',
                               'SURFACE', 'RELIEF', 'RELIEF_SHADOW', 'REFRACTION')
PASSES = ('single', 'radiance', 'overlay')

RAY_UNIFORMS = r'''
varying vec2 v_tex_coord;
uniform sampler2D u_texture;
uniform float u_time;
uniform vec2 u_size;
uniform float u_er;
uniform float u_pxs;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_fwd;
uniform vec3 u_sun;
uniform vec2 u_rot;
uniform vec3 u_sat;
uniform vec3 u_e2;
uniform float u_trail;
uniform float u_cloud;
uniform float u_ctime;
uniform vec2 u_cam;
uniform vec4 u_st0;
uniform vec4 u_st1;
uniform vec4 u_st2;
uniform vec4 u_st3;
uniform vec4 u_st4;
uniform vec4 u_st5;
uniform vec3 u_campos;
uniform float u_persp;
uniform float u_focal;
uniform float u_exposure;
#if defined(SURFACE) && defined(DO_RAD)
uniform sampler2D u_surf;                 // radiance pass only: the overlay keeps the
uniform sampler2D u_aux;                  // 2.0.0a9 sampler set (texture, LUTs, weather)
#endif
#ifdef SCATTER
uniform sampler2D u_trans;
uniform vec4 u_atm;
uniform float u_haze;
#endif
#ifdef WEATHER
uniform sampler2D u_weather;
uniform vec2 u_wrot;
uniform vec4 u_cld;
#ifdef WEATHER2
uniform sampler2D u_weather2;
uniform float u_wmix;
#endif
#endif
#ifdef VCLOUDS
uniform sampler2D u_noise;
#endif

const float PI = 3.14159265;
{PALETTE}
'''

# helpers shared by both passes ------------------------------------------------
RAY_COMMON_GLSL = r'''
vec4 acc;
#ifdef NANCHECK
bool g_bad = false;
#endif
void over(vec3 c, float a) {
#ifdef NANCHECK
    if (!(a == a) || !(dot(c, c) == dot(c, c)) || abs(a) > 1e6 || dot(c, c) > 1e12) g_bad = true;
#endif
    a = clamp(a, 0.0, 1.0);
    acc = vec4(c * a + acc.rgb * (1.0 - a), a + acc.a * (1.0 - a));
}
float lineA(float d, float w) { return clamp((0.5 * w - d) / u_pxs + 0.5, 0.0, 1.0); }
float segd(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
    return length(pa - ba * h);
}
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1.0, 0.0, 0.0)), f.x),
                   mix(hash(i + vec3(0.0, 1.0, 0.0)), hash(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
               mix(mix(hash(i + vec3(0.0, 0.0, 1.0)), hash(i + vec3(1.0, 0.0, 1.0)), f.x),
                   mix(hash(i + vec3(0.0, 1.0, 1.0)), hash(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float s = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        s += a * vnoise(p);
        p = p * 2.03 + vec3(1.7, 9.2, 3.1);
        a *= 0.5;
    }
    return s;
}
bool inShadow(vec3 c) {
    float d = dot(c, u_sun);
    return d < 0.0 && length(c - d * u_sun) < 1.0;
}
#ifdef REFRACTION
// Astronomical refraction (Saemundsson 1986, standard air): how much the air lifts an
// object at true elevation e (degrees), in degrees; 0.57 deg at the horizon, 0.1 at 9 deg.
// Scaled by the air density at the observer's height h (km).
float refrDeg(float e, float h) {
    float a = radians(max(e, -2.0) + 10.3 / (max(e, -2.0) + 5.11));
    return 1.02 / 60.0 / tan(a) * exp(-max(h, 0.0) / 8.0) * smoothstep(-2.0, -1.0, e);
}
// sine of the elevation the sun appears at, seen from radius r with the true value s
float refrSin(float r, float s) {
    float h = (r - 1.0) * 6371.0;
    if (h > 40.0) return s;
    float e = degrees(asin(clamp(s, -1.0, 1.0)));
    return sin(radians(e + refrDeg(e, h)));
}
// total bending of a ray that grazes the Earth at tangent height h km and leaves again
// (twice the horizon refraction at that height), radians
float limbBend(float h) { return 0.0203 * exp(-h / 7.3); }
#endif
#ifdef PENUMBRA
// Sunlight reaching a point outside the Earth (Earth radii): the visible part of the
// solar disc past the limb (0.267 deg radius, exact for a straight limb edge) times the
// transmission of the air the grazing ray crosses (Rayleigh + aerosol, real 6378 km
// Earth, Chapman grazing path), so eclipse entry fades over seconds and goes red.
vec3 sunVis(vec3 p) {
    float r = length(p);
    float s = dot(p, u_sun);
    if (s >= 0.0) return vec3(1.0);
    float b = sqrt(max(r * r - s * s, 0.0));               // closest approach of the sun ray
    float rho = asin(min(1.0 / r, 1.0));                    // Earth's angular radius from p
    float th = acos(clamp(-s / r, -1.0, 1.0));              // angle between Earth's centre and the Sun
#ifdef REFRACTION
    // the air is a lens: a ray grazing at height h is bent by limbBend(h), so sunlight
    // still arrives after the Sun has set geometrically, from a lower, redder, thinner
    // slice of air, and spread out (the vertical squeeze dims it)
    float L = sqrt(max(r * r - 1.0, 0.0)) * 6378.137;      // km from p to the limb
    float d = rho - th;                                     // Sun's depth below the limb (rad)
    float h = max((b - 1.0) * 6378.137, 0.0);
    if (h < 120.0) {
        // the ray that actually arrives: tangent height h where its bend makes up the
        // Sun's depth, bend(h) - h / L = d (continuous with the straight ray higher up)
        float lo = 0.0, hi = 120.0;
        for (int i = 0; i < 12; i++) {
            float m = 0.5 * (lo + hi);
            if (limbBend(m) - m / L > d) lo = m; else hi = m;
        }
        h = 0.5 * (lo + hi);
    }
    float x = clamp((limbBend(h) - d) / 0.004654, -1.0, 1.0);
    float geo = (0.5 + (x * sqrt(1.0 - x * x) + asin(x)) / PI) / (1.0 + L * limbBend(h) / 7.3);
#else
    float x = clamp((th - rho) / 0.004654, -1.0, 1.0);      // Sun's centre past the limb, in solar radii
    float geo = 0.5 + (x * sqrt(1.0 - x * x) + asin(x)) / PI;
    float h = max((b - 1.0) * 6378.137, 0.0);               // tangent height of the grazing ray, km
#endif
    vec3 tau = vec3(3.29, 7.68, 18.7) * exp(-h / 8.0) + 0.97 * exp(-h / 1.2);
    return geo * exp(-tau);
}
// ring / marker colour: eclipse colour in shadow, the normal colour in sunlight, and the
// red of the sunset the spacecraft sees in between
vec3 eclipseTint(vec3 lit, vec3 sv) {
    float v = dot(sv, vec3(0.3, 0.5, 0.2));
    return mix(mix(ECLC, lit, v), vec3(1.0, 0.42, 0.12), clamp((sv.r - sv.b) * 1.6, 0.0, 1.0) * 0.85);
}
#endif

// linear HDR radiance -> display (exposure, ACES fit, gamma)
vec3 tonemapL(vec3 l) {
    l = max(l, vec3(0.0)) * u_exposure;
    l = (l * (2.51 * l + 0.03)) / (l * (2.43 * l + 0.59) + 0.14);
    return pow(clamp(l, 0.0, 1.0), vec3(1.0 / 2.2));
}
vec3 toLin(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
// composite a premultiplied colour
void overP(vec3 cp, float a) {
    a = clamp(a, 0.0, 1.0);
    acc = vec4(cp + acc.rgb * (1.0 - a), a + acc.a * (1.0 - a));
}
#ifdef CLOUDS
// legacy 2D cloud cover (0..1) at an ECEF surface point
float clouds2D(vec3 ne, float lat) {
    if (u_cloud <= 0.0) return 0.0;
    vec3 q = ne * 2.6 + vec3(u_ctime * 0.035, 0.0, u_ctime * 0.01);
    float w = fbm(q * 1.7 + vec3(4.0));
    float n = fbm(q + 0.85 * vec3(w, w * 0.7, -w));
    float band = 0.72 + 0.28 * cos(lat * 6.0);
    return smoothstep(0.50, 0.80, n * band + 0.05) * u_cloud;
}
#endif

// Display colour -> tone-mapped display colour (identity unless TONEMAP).
// Radiance layers go through this; line work (ring, stations, ...) does not.
vec3 tonemap(vec3 c) {
#ifdef TONEMAP
    vec3 l = pow(max(c, vec3(0.0)), vec3(2.2)) * u_exposure;
    l = (l * (2.51 * l + 0.03)) / (l * (2.43 * l + 0.59) + 0.14);
    return pow(clamp(l, 0.0, 1.0), vec3(1.0 / 2.2));
#else
    return c;
#endif
}
'''

# ray generation and primitives ---------------------------------------------------
RAY_GLSL = r'''
// Earth radii covered by one pixel at ray depth t, and points per Earth radius.
float pixw(float t) { return u_persp > 0.5 ? u_pxs * t / u_focal : u_pxs / u_er; }
float ppr(float t) { return u_persp > 0.5 ? u_focal / max(t, 1e-4) : u_er; }

// Ray / sphere (centre 0, radius r), rd unit length: (t_near, t_far), or
// (1e9, -1e9) on a miss. Near root from c / q to stay accurate at the limb.
vec2 sphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float h = b * b - c;
    if (h < 0.0) return vec2(1e9, -1e9);
    h = sqrt(h);
    float t1 = -b + h;
    float t0 = b < 0.0 ? c / t1 : -b - h;
    return vec2(t0, t1);
}

// ECI (Earth radii) -> panel points, for both projections.
vec2 proj(vec3 w) {
    if (u_persp > 0.5) {
        vec3 v = w - u_campos;
        float z = max(dot(v, -u_fwd), 1e-4);
        return vec2(dot(v, u_right), dot(v, u_up)) * (u_focal / z);
    }
    return vec2(dot(w, u_right), dot(w, u_up)) * u_er;
}
// unit vector from a point toward the camera
vec3 toCam(vec3 w) { return u_persp > 0.5 ? normalize(u_campos - w) : u_fwd; }
// true if the Earth blocks the line of sight from point w (outside the Earth)
bool hidden(vec3 w) {
    vec3 d = toCam(w);
    float L = u_persp > 0.5 ? length(u_campos - w) : 1e9;
    vec2 h = sphere(w, d, 1.0);
    return h.x <= h.y && h.y > 0.0 && h.x < L;
}
'''


# physically based atmosphere (feature SCATTER) ------------------------------------
# Units: the Earth is radius 1; u_atm.y km per unit (a real 100 km atmosphere on a
# planet atmo_scale times smaller, so the shell is visible at phone scale).
#   u_atm = (top radius in units, km per unit, sun irradiance, multiple-scattering gain)
ATMOS_GLSL = r'''
#ifdef SCATTER
#ifndef ATMO_N
#define ATMO_N 10
#endif
#ifndef LAND_GAIN
#define LAND_GAIN 1.0
#endif
#ifndef SURF_GAIN
#define SURF_GAIN 0.8
#endif
#ifndef RELIEF_GAIN
#define RELIEF_GAIN 3.0
#endif
#ifndef RS_N
#define RS_N 8
#endif
#ifndef GLINT_ALPHA
#define GLINT_ALPHA 0.15
#endif
#ifndef REFL_N
#define REFL_N 4
#endif
#ifndef LIMB_GAIN
#define LIMB_GAIN 1.0
#endif
const vec3 BETA_R = vec3(5.802e-3, 13.558e-3, 33.1e-3);
const float BETA_MS = 3.996e-3;
const float BETA_ME = 4.44e-3;
const vec3 BETA_O = vec3(0.650e-3, 1.881e-3, 0.085e-3);

// transmittance to the top of the atmosphere from radius r along cos-zenith mu
// (Bruneton's LUT parameterisation; the LUT stores sqrt(T))
vec3 transmittance(float r, float mu) {
    float rt = u_atm.x;
    float H = sqrt(rt * rt - 1.0);
    float rho = sqrt(max(r * r - 1.0, 0.0));
    float d = max(-r * mu + sqrt(max(r * r * (mu * mu - 1.0) + rt * rt, 0.0)), 0.0);
    float dmin = rt - r;
    float xm = clamp((d - dmin) / max(rho + H - dmin, 1e-6), 0.0, 1.0);
    float xr = clamp(rho / H, 0.0, 1.0);
    vec3 c = texture2D(u_trans, vec2(0.5 / 256.0 + xm * (255.0 / 256.0),
                                     0.5 / 64.0 + xr * (63.0 / 64.0))).rgb;
    return c * c;
}
// sunlight reaching radius r with the sun at cos-zenith mus (0 below the horizon)
vec3 sunT(float r, float mus) {
#ifdef REFRACTION
    mus = refrSin(r, mus);                    // the air lifts the low sun: light lasts longer
#endif
    float muh = -sqrt(max(1.0 - 1.0 / (r * r), 0.0));
    return transmittance(r, mus) * smoothstep(muh - 0.006, muh + 0.006, mus);
}
vec3 density(float h) {                       // Rayleigh, Mie, ozone at altitude h km
    return vec3(exp(-h / 8.0), exp(-h / 1.2), max(0.0, 1.0 - abs(h - 25.0) / 15.0));
}
float phaseR(float c) { return 0.0596831 * (1.0 + c * c); }
float phaseM(float c) {                       // Cornette-Shanks, g = 0.76
    float d = max(1.5776 - 1.52 * c, 1e-4);
    return 0.0195620 * (1.0 + c * c) / (d * sqrt(d));
}

// Single scattering along ro + t rd for t in [ta, tb], samples bunched toward tm
// (the ground hit or the closest approach). L: radiance per unit sun irradiance.
void scatter(vec3 ro, vec3 rd, float ta, float tb, float tm, out vec3 L, out vec3 Tv) {
    float km = u_atm.y;
    float cth = dot(rd, u_sun);
    float pR = phaseR(cth);
    float pM = phaseM(cth);
    vec3 od = vec3(0.0);
    L = vec3(0.0);
    float fl = clamp((tm - ta) / max(tb - ta, 1e-6), 0.0, 1.0);
    for (int i = 0; i < ATMO_N; i++) {
        float u = (float(i) + 0.5) / float(ATMO_N);
        float t;
        float dt;
        if (u < fl) {
            float v = (fl - u) / fl;
            t = tm - (tm - ta) * v * v;
            dt = 2.0 * v * (tm - ta) / fl;
        } else {
            float v = (u - fl) / max(1.0 - fl, 1e-6);
            t = tm + (tb - tm) * v * v;
            dt = 2.0 * v * (tb - tm) / max(1.0 - fl, 1e-6);
        }
        float ds = dt / float(ATMO_N) * km;
        vec3 p = ro + rd * t;
        float r = length(p);
        vec3 rho = density(max(r - 1.0, 0.0) * km);
        vec3 ext = BETA_R * rho.x + BETA_ME * rho.y + BETA_O * rho.z;
        vec3 Ti = exp(-(od + ext * (0.5 * ds)));
        od += ext * ds;
        float mus = dot(p, u_sun) / r;
        vec3 Ts = sunT(r, mus);
        vec3 bs = BETA_R * rho.x + vec3(BETA_MS * rho.y);
        // cheap multiple scattering: isotropic, lit by sun light averaged over the sky
        vec3 Tms = transmittance(r, max(mus, 0.1)) * smoothstep(-0.35, 0.15, mus);
        L += Ti * (Ts * (BETA_R * (rho.x * pR) + vec3(BETA_MS * rho.y * pM))
                   + bs * (u_atm.w * 0.0795775) * Tms) * ds;
    }
    Tv = exp(-od);
}
#ifdef SKYREFLECT
// the same integral with REFL_N samples, for the ray reflected by the ocean
void scatterR(vec3 ro, vec3 rd, float ta, float tb, float tm, out vec3 L, out vec3 Tv) {
    float km = u_atm.y;
    float cth = dot(rd, u_sun);
    float pR = phaseR(cth);
    float pM = phaseM(cth);
    vec3 od = vec3(0.0);
    L = vec3(0.0);
    float fl = clamp((tm - ta) / max(tb - ta, 1e-6), 0.0, 1.0);
    for (int i = 0; i < REFL_N; i++) {
        float u = (float(i) + 0.5) / float(REFL_N);
        float t;
        float dt;
        if (u < fl) {
            float v = (fl - u) / fl;
            t = tm - (tm - ta) * v * v;
            dt = 2.0 * v * (tm - ta) / fl;
        } else {
            float v = (u - fl) / max(1.0 - fl, 1e-6);
            t = tm + (tb - tm) * v * v;
            dt = 2.0 * v * (tb - tm) / max(1.0 - fl, 1e-6);
        }
        float ds = dt / float(REFL_N) * km;
        vec3 p = ro + rd * t;
        float r = length(p);
        vec3 rho = density(max(r - 1.0, 0.0) * km);
        vec3 ext = BETA_R * rho.x + BETA_ME * rho.y + BETA_O * rho.z;
        vec3 Ti = exp(-(od + ext * (0.5 * ds)));
        od += ext * ds;
        float mus = dot(p, u_sun) / r;
        vec3 Ts = sunT(r, mus);
        vec3 bs = BETA_R * rho.x + vec3(BETA_MS * rho.y);
        // cheap multiple scattering: isotropic, lit by sun light averaged over the sky
        vec3 Tms = transmittance(r, max(mus, 0.1)) * smoothstep(-0.35, 0.15, mus);
        L += Ti * (Ts * (BETA_R * (rho.x * pR) + vec3(BETA_MS * rho.y * pM))
                   + bs * (u_atm.w * 0.0795775) * Tms) * ds;
    }
    Tv = exp(-od);
}
#endif
#endif
'''


# clouds (phase 3, features WEATHER and VCLOUDS) -------------------------------------
# u_weather: baked weather map (R coverage, G type, B thickness), equirectangular in
#            a frame turning with the Earth plus a slow eastward drift (u_wrot).
# u_noise:   30^3 tiling 3D noise atlas (R Perlin-Worley, G/B Worley fbm).
# u_cld    = (shell base, shell top in Earth radii above the ground,
#             extinction per Earth radius at density 1, cloud time in hours)
CLOUDS_GLSL = r'''
#ifdef WEATHER
#ifndef CLOUD_N
#define CLOUD_N 24
#endif
#ifndef CLOUD_NB
#define CLOUD_NB 8.0
#endif
#ifndef CLOUD_NMIN
#define CLOUD_NMIN 6.0
#endif
#ifndef CLOUD_NS
#define CLOUD_NS 10
#endif
#ifndef LIGHT_N
#define LIGHT_N 4
#endif
#ifndef MS_OCT
#define MS_OCT 2
#endif
#ifndef NOISE_F
#define NOISE_F 80.0
#endif
#ifndef NOISE_FD
#define NOISE_FD 650.0
#endif
#ifndef NOISE_V
#define NOISE_V 9.0
#endif
#ifndef CLOUD_GAIN
#define CLOUD_GAIN 2.0
#endif
#ifndef BASE_AMT
#define BASE_AMT 0.6
#endif
#ifndef ERODE
#define ERODE 0.15
#endif
#ifndef CELLS
#define CELLS 1.0
#endif
#ifndef CI_OPAC
#define CI_OPAC 0.7
#endif
#ifndef CI_F
#define CI_F 55.0
#endif
#ifndef CI_STRETCH
#define CI_STRETCH 3.5
#endif
#ifndef CI_WF
#define CI_WF 30.0
#endif
#ifndef CI_WARP
#define CI_WARP 18.0
#endif
#ifndef CI_LO
#define CI_LO 0.35
#endif
#ifndef DETAIL_BILLOW
#define DETAIL_BILLOW 1.0
#endif
#ifndef NOISE_W
#define NOISE_W 140.0
#endif
#ifndef WARP_AMT
#define WARP_AMT 0.022
#endif

// ECI -> weather frame (Earth rotation + cloud drift about z)
vec3 wframe(vec3 p) { return vec3(u_wrot.x * p.x + u_wrot.y * p.y, -u_wrot.y * p.x + u_wrot.x * p.y, p.z); }
vec3 weatherAt(vec3 q) {
    float lon = atan(q.y, q.x);
    float lat = asin(clamp(q.z / max(length(q), 1e-6), -1.0, 1.0));
    vec2 wuv = vec2(lon * 0.1591549 + 0.5, lat * 0.3183099 + 0.5);
    vec3 w = texture2D(u_weather, wuv).rgb;
#ifdef WEATHER2
    w = mix(w, texture2D(u_weather2, wuv).rgb, u_wmix);     // precomputed film: blend neighbouring hours
#endif
    // GlobeView(clouds=...): 0.55 shows the baked map, more or less shifts the coverage
    w.r = clamp(w.r + (u_cloud - 0.55) * 0.9, 0.0, 1.0) * smoothstep(0.0, 0.12, u_cloud);
    w.b *= smoothstep(0.0, 0.12, u_cloud) * clamp(u_cloud / 0.55, 0.0, 1.5);
    return w;
}
float remap01(float x, float a, float b) { return clamp((x - a) / max(b - a, 1e-4), 0.0, 1.0); }

#ifdef VCLOUDS
// tiling 3D noise: period 30 (texel units), 30 slices in 32 x 32 padded cells
vec3 noise3(vec3 p) {
    p -= 30.0 * floor(p * 0.0333333);
    float z0 = clamp(floor(p.z), 0.0, 29.0);
    float fz = clamp(p.z - z0, 0.0, 1.0);
    float z1 = z0 > 28.5 ? 0.0 : z0 + 1.0;
    vec2 uv = clamp(p.xy, 0.0, 30.0) + 1.5;
    float r0 = floor(z0 * 0.125);
    float r1 = floor(z1 * 0.125);
    vec2 c0 = vec2(z0 - 8.0 * r0, r0) * 32.0;
    vec2 c1 = vec2(z1 - 8.0 * r1, r1) * 32.0;
    vec3 a = texture2D(u_noise, (c0 + uv) * vec2(0.00390625, 0.0078125)).rgb;
    vec3 b = texture2D(u_noise, (c1 + uv) * vec2(0.00390625, 0.0078125)).rgb;
    return mix(a, b, fz);
}
// ragged, wind-streaked edges: the weather map is read through a noise warp,
// stretched along the latitude circles
vec3 warpW(vec3 q) {
    vec3 n = noise3(vec3(q.xy, q.z * 2.2) * NOISE_W + vec3(3.0, 7.0, 1.0)) - 0.5;
    return q + n * WARP_AMT;
}
// cloud density at weather-frame point q, normalised shell height hn, weather w
float g_near = 0.0;                       // > 0: the cloud is close (camera on the ground)
float cloudDens(vec3 q, float hn, vec3 w, float dk) {
    if (hn < 0.0 || hn > 1.0 || w.r < 0.02) return 0.0;
    float th = clamp(0.35 + 0.75 * w.r, 0.35, 1.0);                   // dense cover = deep cloud
    float top = mix(0.22, 1.0, w.g) * mix(0.55, 1.0, th);             // stratus low, convection tall
    float prof = smoothstep(0.0, 0.07, hn) * (1.0 - smoothstep(top * 0.40, top, hn));
    if (prof <= 0.0) return 0.0;
    vec3 nq = q * (NOISE_F + hn * NOISE_V) + vec3(0.37, -0.21, 0.13) * u_cld.w;
    // stratiform decks are smooth sheets, convective cloud is lumpy (type w.g)
    float base = mix(0.62, noise3(nq).r, mix(0.18, 1.0, w.g * w.g) * BASE_AMT);
    float s = remap01(base * prof, 1.0 - w.r, 1.0) * w.r;
#ifdef DETAIL
    if (dk > 0.0 && s > 0.0) {
        vec3 dn = noise3(q * NOISE_FD + vec3(11.0, 5.0, 17.0) - vec3(0.9, 0.4, -0.6) * u_cld.w);
        float f = dn.g * 0.65 + dn.b * 0.35;
        f = mix(f, 1.0 - f, clamp(hn * 6.0, 0.0, 1.0) * DETAIL_BILLOW);  // wispy bases, billowy tops
        s = remap01(s, f * ERODE * dk * (1.0 - s), 1.0);      // erode the edges, keep cores solid
        // cellular stratocumulus in the low decks: closed cells (bright cells, dark lanes)
        // where the deck is solid, open cells (cloud walls round clear centres) where broken
        float low = (1.0 - smoothstep(0.12, 0.32, w.g)) * CELLS * dk;
        if (low > 0.0) {
            float closed = smoothstep(0.30, 0.62, dn.g);
            float open = 1.0 - smoothstep(0.30, 0.55, dn.g);
            s *= mix(1.0, mix(open, closed, smoothstep(0.45, 0.80, w.r)), low);
        }
        // fair-weather cumulus: broken convective cover splits into separate puffs
        float pop = smoothstep(0.50, 0.80, w.g) * (1.0 - smoothstep(0.30, 0.65, w.r)) * dk;
        s *= mix(1.0, smoothstep(0.42, 0.72, dn.b), pop);
        if (g_near > 0.0 && s > 0.0) {
            // seen from close by: a finer octave (~1.5 km) frays edges into turrets and wisps
            vec3 dn2 = noise3(q * (NOISE_FD * 5.3) + vec3(3.0, 7.0, 1.0) - vec3(0.5, 0.2, -0.3) * u_cld.w);
            float f2 = mix(dn2.g, 1.0 - dn2.g, clamp(hn * 6.0, 0.0, 1.0) * DETAIL_BILLOW) * 0.7 + dn2.b * 0.3;
            s = remap01(s, f2 * 0.35 * g_near * (1.0 - 0.6 * s), 1.0);
        }
    }
#endif
    return s * mix(0.55, 1.0, th);
}
// optical depth (at density 1) toward the sun, cone of growing steps
float lightOD(vec3 q, vec3 sw, vec3 w, float thick) {
    float od = 0.0;
    float ds = thick * (LIGHT_N > 4 ? 0.035 : 0.06);
    vec3 p = q;
    for (int j = 0; j < LIGHT_N; j++) {
        p += sw * ds;
        float hn = (length(p) - 1.0 - u_cld.x) / thick;
        if (hn > 1.0) break;
        od += cloudDens(p, hn, w, 0.0) * ds;
        ds *= 2.0;
    }
    return od;
}
float hg(float c, float g) {
    float d = 1.0 + g * g - 2.0 * g * c;
    return (1.0 - g * g) / (12.5663706 * d * sqrt(d));
}
float ign(vec2 f) { return fract(52.9829189 * fract(dot(f, vec2(0.06711056, 0.00583715)))); }

// March the cloud shell in front of the ground (or through it, above the limb).
// Lc: cloud radiance (HDR), Tc: transmittance through the clouds.
float g_ct = 0.0;                         // opacity-weighted depth of the cloud along the ray
void cloudMarch(vec3 ro, vec3 rd, bool hitE, float tE, out vec3 Lc, out float Tc) {
    Lc = vec3(0.0);
    Tc = 1.0;
    g_ct = 0.0;
    float thick = u_cld.y - u_cld.x;
    vec2 so = sphere(ro, rd, 1.0 + u_cld.y);
    if (so.x > so.y || so.y <= 0.0) return;
    float t0 = max(so.x, 0.0);
    vec2 si = sphere(ro, rd, 1.0 + u_cld.x);
    // a camera under the cloud base (on the ground) starts where the ray leaves the base
    if (dot(ro, ro) < (1.0 + u_cld.x) * (1.0 + u_cld.x)) t0 = max(si.y, 0.0);
    float t1 = (si.x <= si.y && si.x > t0) ? si.x : so.y;
    if (hitE) t1 = min(t1, tE);
    float seg = t1 - t0;
    if (seg <= 0.0) return;
    // Rays that reach the cloud base (the disc) fall monotonically through the shell:
    // sample them at fixed heights (slab midpoints), so a cloud is sampled at the same
    // places whatever the camera does. Rays that only graze the shell above the limb
    // march evenly in t. No screen-space jitter: it made thin cloud boil on camera moves.
    bool down = hitE || (si.x <= si.y && si.x > t0);
    // from inside or under the shell: steps grow with distance (dense close by)
    bool near = !down && dot(ro, ro) < (1.0 + u_cld.y) * (1.0 + u_cld.y);

    float n = down ? float(CLOUD_NS) : floor(clamp(seg / thick * CLOUD_NB, CLOUD_NMIN, float(CLOUD_N)));
    if (near) n = float(CLOUD_N);
    float dt0 = seg / n;
    vec3 re = ro + rd * t0;                 // re-based origin: precise slab intersections
    float tprev = 0.0;
    vec3 sw = wframe(u_sun);
    vec3 qo = wframe(ro);
    vec3 qd = wframe(rd);
    float c = dot(rd, u_sun);
    // two-lobe phase (silver lining forward, soft backscatter), and the
    // multiple-scattering octaves (Wrenninge): energy 1/2, extinction 1/2, eccentricity 1/2
    float ph0 = mix(hg(c, 0.80), hg(c, -0.30), 0.5);
    float ph1 = mix(hg(c, 0.40), hg(c, -0.15), 0.5);
    float ph2 = mix(hg(c, 0.20), hg(c, -0.08), 0.5);
    float kpow = 0.30 * (0.5 - 0.5 * c);                 // powder only when looking away from the sun
    // detail erosion fades out where its cells (~0.009 Earth radii) get smaller than ~3 pixels
    float dk = 1.0 - smoothstep(0.002, 0.006, pixw(t0));
#ifdef LIGHTNING
    vec3 qe = qo + qd * t0;
    vec3 cq = qe * 55.0;
    float fl = hash(floor(cq) + floor(u_time * 6.0) * 0.731);
    vec3 fo = fract(cq) - 0.5;                                  // round, soft flash inside its cell
    float flash = smoothstep(0.9965, 1.0, fl) * exp(-dot(fo, fo) * 14.0)
                * (1.0 - smoothstep(-0.12, 0.0, dot(qe, sw) / length(qe)));
#endif
    for (int i = 0; i < CLOUD_N; i++) {
        if (float(i) >= n || Tc < 0.02) break;
        float t, dt;
        if (down) {
            float fi = float(i);
            float tb = fi + 1.0 >= n ? seg : sphere(re, rd, 1.0 + u_cld.x + thick * (1.0 - (fi + 1.0) / n)).x;
            t = t0 + sphere(re, rd, 1.0 + u_cld.x + thick * (1.0 - (fi + 0.5) / n)).x;
            dt = tb - tprev;
            tprev = tb;
        } else if (near) {
            float u = (float(i) + 0.5) / n;
            t = t0 + seg * u * u;
            dt = 2.0 * u * seg / n;
        } else {
            t = t0 + (float(i) + 0.5) * dt0;
            dt = dt0;
        }
        vec3 q = qo + qd * t;
        float r = length(q);
        float hn = (r - 1.0 - u_cld.x) / thick;
        vec3 w = weatherAt(warpW(q));
        if (w.r < 0.02) continue;                          // clear sky: skip the noise
        g_near = near ? 1.0 - smoothstep(0.0004, 0.0012, pixw(t)) : 0.0;
        float d = cloudDens(q, hn, w, near ? max(dk, 1.0 - smoothstep(0.002, 0.006, pixw(t))) : dk);
        g_near = 0.0;
        if (d <= 0.0) continue;
        float sig = u_cld.z * d;
        float od = lightOD(q, sw, w, thick) * u_cld.z;
        float mus = dot(q, sw) / r;
        // sunlight at the cloud's real altitude (1-12 km): the shell is drawn exaggerated,
        // but the light that reaches it has crossed the low, dense air (red tops at sunset)
        vec3 sunc = sunT(1.0 + (1.0 + 11.0 * hn) / u_atm.y, mus);
        float L1 = exp(-od) * ph0 + 0.5 * exp(-0.5 * od) * ph1;
#if MS_OCT > 2
        L1 += 0.25 * exp(-0.25 * od) * ph2;
#endif
        L1 *= mix(1.0, 1.0 - exp(-2.0 * od), kpow);
        float day = smoothstep(-0.20, 0.25, mus);
        vec3 amb = (vec3(0.30, 0.50, 1.0) * 0.08 * (0.35 + 0.65 * hn)
                    + vec3(0.12, 0.12, 0.10) * 0.10 * max(mus, 0.0) * (1.0 - hn)) * day / PI;
        vec3 S = (sunc * L1 * CLOUD_GAIN + amb) * u_atm.z;
#ifdef LIGHTNING
        S += vec3(0.75, 0.80, 1.0) * flash * 6.0 * smoothstep(0.7, 0.95, w.g) * (1.0 - hn);
#endif
        float a = 1.0 - exp(-sig * dt);
        Lc += Tc * S * a;
        g_ct += Tc * a * t;
        Tc *= 1.0 - a;
    }
    g_ct /= max(1.0 - Tc, 1e-4);
}
// High, thin cirrus at the top of the shell: fibrous streaks stretched along the
// latitude circles (the jet direction), amount from the weather map's B channel.
// Returns opacity; L = premultiplied radiance.
float cirrus(vec3 ro, vec3 rd, out vec3 L) {
    L = vec3(0.0);
    float rci = 1.0 + u_cld.y * 1.04;
    vec2 sc = sphere(ro, rd, rci);
    if (sc.x > sc.y || sc.y <= 0.0) return 0.0;
    if (sc.x < 0.0) {                                  // under the layer: rays into the ground see none
        vec2 sg = sphere(ro, rd, 1.0);
        if (sg.x <= sg.y && sg.y > 0.0) return 0.0;
    }
    float tci = sc.x > 0.0 ? sc.x : sc.y;
    vec3 p = ro + rd * tci;
    vec3 qc = wframe(p);
    float ci = weatherAt(warpW(qc)).b;
    if (ci < 0.02) return 0.0;
    // fibres: noise stretched ~2.5x along the latitude circles, warped by a coarser noise
    // so the strands curve and fray instead of lining up
    vec3 wq = noise3(qc * CI_WF + vec3(5.0, 1.0, 8.0)) - 0.5;
    vec3 nq = vec3(qc.xy * CI_F, qc.z * CI_F * CI_STRETCH) + wq * CI_WARP + vec3(0.11, -0.07, 0.0) * u_cld.w;
    float fib = noise3(nq).r * 0.6 + noise3(nq * 2.3 + vec3(9.0, 3.0, 5.0)).g * 0.4;
    float tau = ci * smoothstep(0.02, 0.30, ci) * smoothstep(CI_LO, 0.95, fib) * CI_OPAC;
    float mv = abs(dot(rd, p)) / length(p);
    float a = 1.0 - exp(-tau / max(mv, 0.12));
    if (a <= 0.0) return 0.0;
    float mus = dot(p, u_sun) / length(p);
    float c = dot(rd, u_sun);
    float ph = mix(hg(c, 0.75), 0.0795775, 0.45);                     // ice: strong forward lobe
    vec3 sunc = sunT(1.0 + 11.0 / u_atm.y, mus);
    L = (sunc * ph * 2.4 + vec3(0.30, 0.50, 1.0) * 0.03 * smoothstep(-0.2, 0.25, mus)) * u_atm.z * a;
    return a;
}
// transmission of the line of sight from a ground point up through the cloud shell
// (the same density the radiance pass draws, at two heights): hides lights under cloud
float cloudViewT(vec3 nw, vec3 view, float z, float dk) {
    float thick = u_cld.y - u_cld.x;
    float zz = max(z, 0.15);
    float tau = 0.0;
    for (int k = 0; k < 2; k++) {
        float hn = 0.25 + 0.5 * float(k);
        vec3 q = wframe(nw + view * ((u_cld.x + thick * hn) / zz));
        q *= (1.0 + u_cld.x + thick * hn) / length(q);
        vec3 w = weatherAt(warpW(q));
        tau += cloudDens(q, hn, w, dk);
    }
    return exp(-tau * u_cld.z * thick * 0.5 / zz);
}
// cheap cloud opacity over a ground point (the full-resolution overlay pass)
float cloudCoverEst(vec3 nw, float slant) {
    float thick = u_cld.y - u_cld.x;
    vec3 q = wframe(nw * (1.0 + u_cld.x + 0.2 * thick));
    vec3 w = weatherAt(warpW(q));
    float d = max(cloudDens(q, 0.12, w, 0.0), cloudDens(q, 0.35, w, 0.0));
    return 1.0 - exp(-u_cld.z * d * thick * 1.5 * slant);
}
#endif

// shadow factor on the ground (direct sunlight) from the clouds above
float cloudShadow(vec3 nw, float mu, float dk) {
    if (mu <= 0.0) return 1.0;
    float thick = u_cld.y - u_cld.x;
    float ms = max(mu, 0.12);
#ifdef VCLOUDS
    // the same cloud the camera sees: density (with the same erosion) integrated through
    // the shell along the sun ray, at two heights
    float tau = 0.0;
    for (int k = 0; k < 2; k++) {
        float hn = 0.25 + 0.5 * float(k);
        vec3 q = wframe(nw + u_sun * ((u_cld.x + thick * hn) / ms));
        q *= (1.0 + u_cld.x + thick * hn) / length(q);
        vec3 w = weatherAt(warpW(q));
        tau += cloudDens(q, hn, w, dk);
    }
    tau *= u_cld.z * thick * 0.5 / ms;
#else
    vec3 q = wframe(nw + u_sun * ((u_cld.x + 0.25 * thick) / ms));
    vec3 w = weatherAt(q);
    float tau = 4.0 * w.r * w.r;
#endif
    return 0.22 + 0.78 * exp(-tau);
}
#ifndef VCLOUDS
// LOW: the weather map drawn as a lit 2D layer with fbm edges
float clouds2W(vec3 nw) {
    vec3 q = wframe(nw);
    vec3 w = weatherAt(q);
    if (w.r < 0.02) return 0.0;
    float nv = fbm(q * 14.0 + vec3(0.2, -0.1, 0.05) * u_cld.w);
    return smoothstep(1.0 - w.r - 0.10, 1.0 - w.r + 0.18, nv * 0.9 + 0.15 * w.r)
           * clamp(w.r * 1.6, 0.0, 1.0) * mix(0.75, 1.0, w.r);
}
#endif
#endif
'''

RAY_MAIN = r'''
#ifdef DO_OVL
void stationO(vec3 nw, vec4 st, float k, float cov) {
    if (dot(st.xyz, st.xyz) < 0.5) return;
    float d = acos(clamp(dot(nw, st.xyz), -1.0, 1.0)) * k;
    float on = st.w;
    over(GOLD, lineA(abs(d - 4.5), 1.0) * on * (0.6 + 0.4 * sin(u_time * 5.0)) * cov);
    over(mix(TEXTC, GOLD, on), lineA(d, 4.0) * (0.65 + 0.35 * on) * cov);
}
void link(vec2 pt, vec4 st, vec2 ps, bool satVis) {
    if (st.w < 0.5 || !satVis) return;
    if (dot(st.xyz, toCam(st.xyz)) <= 0.0) return;
    vec2 a = proj(st.xyz);
    float d = segd(pt, a, ps);
    float t = clamp(dot(pt - a, ps - a) / max(dot(ps - a, ps - a), 1e-9), 0.0, 1.0);
    float pulse = 0.55 + 0.45 * sin(t * 40.0 - u_time * 9.0);
    over(GOLD, lineA(d, 1.3) * 0.85 * pulse);
}
#endif

#ifdef ORBIT
// Closest point of the orbit circle (radius R in the e1/e2 plane) to the ray,
// by Gauss-Newton on the ray-to-circle distance (= screen distance in ortho).
// Returns the angle; C, tC (ray depth) and od (distance in points) are outputs.
float ringSolve(float phi, vec3 ro, vec3 rd, vec3 e1, float R, out vec3 C, out float tC, out float od) {
    for (int i = 0; i < 4; i++) {
        vec3 C0 = (e1 * cos(phi) + u_e2 * sin(phi)) * R;
        vec3 D0 = (-e1 * sin(phi) + u_e2 * cos(phi)) * R;
        vec3 v0 = C0 - ro;
        vec3 q0 = v0 - rd * dot(v0, rd);
        vec3 d0 = D0 - rd * dot(D0, rd);
        phi -= dot(q0, d0) / max(dot(d0, d0), 1e-6);
    }
    C = (e1 * cos(phi) + u_e2 * sin(phi)) * R;
    tC = dot(C - ro, rd);
    od = length(C - ro - rd * tC) * ppr(tC);
    return phi;
}
// colour and coverage of the ring at a solved point (trail, eclipse, far side)
float ringStyle(vec3 C, float tC, float od, float phi, float tc, out vec3 ocol) {
    bool back = tC > tc;                      // behind the limb plane (legacy cdepth < 0)
    phi = mod(phi + PI, 2.0 * PI) - PI;
    float frac = u_trail > 1e-4 ? 1.0 + phi / u_trail : -1.0;
    bool inTrail = phi <= 0.0 && frac >= 0.0;
#ifdef PENUMBRA
    vec3 oc = eclipseTint(CYAN, sunVis(C));
#else
    vec3 oc = inShadow(C) ? ECLC : CYAN;
#endif
    float ow = inTrail ? 1.1 + 1.8 * frac : 1.15;
    ocol = inTrail ? mix(oc, BGC, 0.70 * (1.0 - frac)) : mix(oc, BGC, back ? 0.62 : 0.36);
    return tC > 0.0 && od < 1e8 ? lineA(od, ow) * (inTrail ? 1.0 : 0.9) : 0.0;
}
#endif

// legacy exponential shell, evaluated from the ray's closest approach
float atmo(vec3 X, float rc, vec3 view, out vec3 atmc) {
    vec3 dir = rc > 1e-5 ? X / rc : vec3(0.0);
    float nd = dot(dir, u_sun);
    float sz = dot(u_sun, view);
    float lit = smoothstep(-0.30, 0.45, nd + 0.25 * sz);
    float twx = nd / 0.25;
    float tw = exp(-twx * twx);
    float behind = pow(max(-sz, 0.0), 2.0);
    float hgt = max(rc - 1.0, 0.0);
    float shell = exp(-hgt / 0.028);
    atmc = mix(vec3(0.28, 0.68, 1.0), vec3(1.0, 0.52, 0.26), clamp(0.6 * tw + 0.6 * behind, 0.0, 0.85));
    return min(shell * (0.08 + 0.70 * lit + 0.30 * tw + 0.85 * behind * (0.35 + 0.65 * smoothstep(-0.5, 0.9, nd))), 0.95);
}

void main() {
    acc = vec4(0.0);
    vec2 pt = (v_tex_coord - 0.5) * u_size;

    // ---- primary ray
    vec3 ro, rd;
    if (u_persp > 0.5) {
        ro = u_campos;
        rd = normalize(u_right * pt.x + u_up * pt.y - u_fwd * u_focal);
    } else {
        ro = (u_right * pt.x + u_up * pt.y) / u_er + u_fwd * 50.0;
        rd = -u_fwd;
    }
    vec3 view = -rd;
    float tc = -dot(ro, rd);                  // depth of closest approach to the centre
    if (u_persp > 0.5) tc = max(tc, 0.0);    // a ray looking away: nearest point is the eye
    vec3 X = ro + rd * tc;
    float rc = length(X);
    float pw = pixw(tc);
    bool hit = rc < 1.0 && tc > 0.0;
    float zh = sqrt(max(1.0 - rc * rc, 0.0));
    float tE = hit ? tc - zh : tc;            // Earth hit (or the limb plane)
    vec3 nw = hit ? X + view * zh : X / max(rc, 1e-6);
    float z = hit ? zh : 0.0;                 // cos(view angle) at the surface
    float cover = tc > 0.0 ? clamp((1.0 - rc) / pw + 0.5, 0.0, 1.0) : 0.0;
    bool onDisc = tc > 0.0 && rc < 1.0 + 2.0 * pw;
    float kS = ppr(tE);                       // points per Earth radius at the surface
    vec3 atmc;
    float aatm = atmo(X, rc, view, atmc);

#ifdef DO_OVL
#ifdef STARS
    if (u_persp < 0.5) {
        vec2 sp = pt + u_cam * vec2(-160.0, 160.0);
        vec2 cell = floor(sp / 7.0);
        float h = hash(vec3(cell, 3.7));
        if (h > 0.972) {
            vec2 o = (vec2(hash(vec3(cell, 1.3)), hash(vec3(cell, 8.1))) * 0.7 + 0.15) * 7.0;
            float d = length(sp - cell * 7.0 - o);
            float tw = 0.65 + 0.35 * sin(u_time * (1.0 + 3.0 * h) + h * 50.0);
            float sa = lineA(d, 0.9 + 1.4 * fract(h * 37.0)) * tw * 0.8;
#ifdef OVL_ONLY
            sa *= (1.0 - cover) * (1.0 - aatm);
#endif
            over(vec3(0.85, 0.92, 1.0), sa);
        }
    } else {
        // stars fixed on the sky: cells in (azimuth, elevation) of the ray; from the
        // ground only once the sun is well below the horizon
        float starsOn = dot(u_campos, u_campos) < 1.05 ?
            1.0 - smoothstep(-0.20, -0.08, dot(normalize(u_campos), u_sun)) : 1.0;
        float el = asin(clamp(rd.z, -1.0, 1.0));
        float ce = max(cos(el), 1e-3);
        vec2 sp = vec2(atan(rd.y, rd.x) * ce, el) * u_focal;
        vec2 cell = floor(sp / 7.0);
        float h = hash(vec3(cell, 3.7));
        if (h > 0.972) {
            vec2 o = (vec2(hash(vec3(cell, 1.3)), hash(vec3(cell, 8.1))) * 0.7 + 0.15) * 7.0;
            vec2 sa2 = (cell * 7.0 + o) / u_focal;            // star (azimuth * cos el, el)
            float se = sa2.y;
            float sz = sa2.x / max(cos(se), 1e-3);
            vec3 sdir = vec3(cos(se) * cos(sz), cos(se) * sin(sz), sin(se));
            float d = length(pt - proj(u_campos + sdir));       // round on screen
            float tw = 0.65 + 0.35 * sin(u_time * (1.0 + 3.0 * h) + h * 50.0);
            float sa = lineA(d, 0.9 + 1.4 * fract(h * 37.0)) * tw * 0.8 * starsOn;
#ifdef OVL_ONLY
            sa *= (1.0 - cover) * (1.0 - aatm);
#endif
            over(vec3(0.85, 0.92, 1.0), sa);
        }
    }
#endif
#endif

#ifdef DO_RAD
#ifdef SCATTER
    // ---- atmosphere: single scattering along the view ray, up to the ground
    vec3 Lsky = vec3(0.0);
    vec3 Tv = vec3(1.0);
    vec2 ah = sphere(ro, rd, u_atm.x);
    bool inAir = u_persp > 0.5 && dot(ro, ro) < u_atm.x * u_atm.x;
    if (ah.x <= ah.y && ah.y > 0.0) {
        float ta = max(ah.x, 0.0);
        float tb = hit ? tE : ah.y;
        float tm = hit ? tE : clamp(tc, ta, tb);
        scatter(ro, rd, ta, tb, tm, Lsky, Tv);
        Lsky *= u_atm.z * LIMB_GAIN;
    }
    if (hit) {
        // "Rayleigh correction" as in true-colour satellite imagery: keep only a
        // fraction u_haze of the air light and extinction looking straight down,
        // ramping back to the full physical value toward the limb
        float muv = clamp(-dot(normalize(ro + rd * tE), rd), 0.0, 1.0);
        float kv = 1.0 - muv;
        float hw = u_haze + (1.0 - u_haze) * kv * kv * kv;
        Lsky *= hw;
        Tv = mix(vec3(1.0), Tv, hw);
    }
    // airglow: thin 557.7 nm emission at ~95 km, seen edge-on at the night limb
    if (!hit) {
        float hc = (rc - 1.0) * u_atm.y;
        float gx = (hc - 95.0) / 6.0;
        // only visible where that air is in the Earth's shadow (by day sunlit air drowns it)
        float sx = dot(X, u_sun);
        float perp = sqrt(max(rc * rc - sx * sx, 0.0));
        float night = (1.0 - smoothstep(-0.04, 0.02, sx)) * (1.0 - smoothstep(0.985, 1.0, perp));
        // 630 nm red oxygen line: a broad, dim band directly above it (~120-300 km)
        float rx = (hc - 165.0) / 60.0;
        Lsky += (vec3(0.20, 0.85, 0.30) * 0.025 * exp(-gx * gx)
               + vec3(0.90, 0.16, 0.06) * 0.007 * exp(-rx * rx)) * u_atm.z * night;
    }
    float Tcl = 1.0;
#ifdef VCLOUDS
    // clouds: air in front of the deck still scatters, the air behind it is hidden
    vec3 Lcl;
    cloudMarch(ro, rd, hit, tE, Lcl, Tcl);
    // air between the eye and the cloud: from space, half the path (sqrt); from inside
    // the air (a camera on the ground) the fraction of the path up to the cloud
    vec3 Tcv = hit ? Tv : sqrt(Tv);
    if (inAir) Tcv = pow(max(Tv, vec3(1e-6)), vec3(clamp(g_ct / max(ah.y, 1e-6), 0.0, 1.0)));
    // the air light in front of the cloud stays: about half from orbit; from the ground,
    // the share of the path up to the cloud
    vec3 front = inAir ? (1.0 - Tcv) / max(1.0 - Tv, vec3(1e-4)) : vec3(0.55);
    Lsky = Lsky * (Tcl + (1.0 - Tcl) * clamp(front, 0.0, 1.0)) + Lcl * Tcv;
    vec3 Lci;
    float aci = cirrus(ro, rd, Lci);
    Lsky = Lsky * (1.0 - aci) + Lci * (inAir ? sqrt(Tv) : Tcv);
    Tcl *= 1.0 - aci;
#endif
    if (inAir && !hit) {
        // the sun's disc seen from inside the air (drawn 2x its true 0.27 degree radius),
        // reddened by the air it crosses and hidden by cloud
        vec3 upo = normalize(ro);
        float cs = dot(rd, u_sun);
        float sd = smoothstep(0.999955, 0.999975, cs);
#ifdef REFRACTION
        // seen through the air: lifted, and squashed (the lower limb is lifted more)
        float et = degrees(asin(clamp(dot(u_sun, upo), -1.0, 1.0)));
        float hk = (length(ro) - 1.0) * 6371.0;
        float ea = et + refrDeg(et, hk);
        float sq = 1.0 + (refrDeg(et + 0.25, hk) - refrDeg(et - 0.25, hk)) / 0.5;   // d(apparent)/d(true)
        vec3 hz = normalize(u_sun - upo * dot(u_sun, upo));
        vec3 sa = hz * cos(radians(ea)) + upo * sin(radians(ea));
        float del = asin(clamp(dot(rd, upo), -1.0, 1.0)) - radians(ea);          // up/down from its centre
        vec3 rh = rd - upo * dot(rd, upo);
        float daz = length(rh - hz * dot(rh, hz));                                  // sideways
        float ang = sqrt(daz * daz + (del / max(sq, 0.2)) * (del / max(sq, 0.2)));
        sd = 1.0 - smoothstep(0.0090, 0.0095, ang);
        cs = dot(rd, sa);
#endif
        float glow = exp((cs - 1.0) * 1500.0) * 0.03;
        Lsky += sunT(length(ro), dot(upo, rd)) * (sd * 40.0 + glow) * u_atm.z * Tcl;
    }
    vec3 skyc = tonemapL(Lsky);
    overP(skyc, max(1.0 - Tcl, max(skyc.r, max(skyc.g, skyc.b))));
#else
#ifdef ATMOSPHERE
    over(tonemap(atmc), aatm);
#endif
#endif
#endif

    float R = max(length(u_sat), 1.0001);
    vec3 e1 = length(u_sat) > 0.5 ? u_sat / length(u_sat) : vec3(1.0, 0.0, 0.0);
#ifdef DO_OVL
#ifdef ORBIT
    // Exact ring. Where the projected ellipse is thin (orbit seen near edge-on)
    // a pixel lies near two sheets of the circle, the near and the far one, so
    // solve from the plane hit and from its mirror image and draw both.
    vec3 onrm = cross(e1, u_e2);
    float dn = dot(rd, onrm);
    vec3 Q0 = abs(dn) > 1e-4 ? ro - rd * (dot(ro, onrm) / dn) : X;
    float phiA = atan(dot(Q0, u_e2), dot(Q0, e1));
    vec3 dip = rd - onrm * dn;                         // view direction within the plane
    vec3 ax = length(dip) > 1e-5 ? cross(onrm, dip) : e1;
    float phiAx = atan(dot(ax, u_e2), dot(ax, e1));
    vec3 Ca, Cb;
    float tCa, tCb, oda, odb;
    phiA = ringSolve(phiA, ro, rd, e1, R, Ca, tCa, oda);
    float phiB = ringSolve(2.0 * phiAx - phiA, ro, rd, e1, R, Cb, tCb, odb);
    if (length(Ca - Cb) * ppr(tCa) < 1.0) odb = 1e9;   // both starts found the same point
    // a = the nearer of the two along the ray, b = the farther
    if (tCb < tCa) {
        vec3 tv = Ca; Ca = Cb; Cb = tv;
        float tt = tCa; tCa = tCb; tCb = tt;
        tt = oda; oda = odb; odb = tt;
        tt = phiA; phiA = phiB; phiB = tt;
    }
    vec3 ocolA, ocolB;
    float oaA = ringStyle(Ca, tCa, oda, phiA, tc, ocolA);
    float oaB = ringStyle(Cb, tCb, odb, phiB, tc, ocolB);
    bool backA = tCa > tc;
    bool backB = tCb > tc;
#ifdef OVL_ONLY
    if (backB) over(ocolB, oaB * (1.0 - cover));
    if (backA) over(ocolA, oaA * (1.0 - cover));
#else
    if (backB) over(ocolB, oaB);
    if (backA) over(ocolA, oaA);
#endif
#endif
#endif

    if (onDisc) {
        vec3 ne = vec3(u_rot.x * nw.x + u_rot.y * nw.y, -u_rot.y * nw.x + u_rot.x * nw.y, nw.z);
        float lon = atan(ne.y, ne.x);
        float lat = asin(clamp(ne.z, -1.0, 1.0));
        float alat = abs(lat) * 57.2958;
        // C1 texel interpolation: magnified up close (astronaut view) the lights stay
        // round instead of bilinear diamonds; the same single fetch
        vec2 tuv = vec2(lon / (2.0 * PI) + 0.5, lat / PI + 0.5) * vec2(TEX_W, TEX_H) + 0.5;
        vec2 tf = fract(tuv);
        tuv = (floor(tuv) + tf * tf * (3.0 - 2.0 * tf) - 0.5) / vec2(TEX_W, TEX_H);
        vec3 tx = texture2D(u_texture, tuv).rgb;
        float land = tx.r;
        float coast = tx.g;
        float lights = tx.b;
        float mu = dot(nw, u_sun);
#if defined(SURFACE) && defined(DO_RAD)
        // NASA Blue Marble colour (sea ice added); u_texture is the matching data texture
        // (land, coast and Black Marble lights); aux: R height sqrt(h / 9 km), 0 = water,
        // G sea-ice concentration
        vec4 ax = texture2D(u_aux, tuv);
        vec3 bm = texture2D(u_surf, tuv).rgb;
        float seaice = ax.g;
        float mur = mu;                       // sun on the relief-tilted ground
#ifdef RELIEF
        // slopes from the height texture (two more fetches), exaggerated
        vec2 tst = vec2(1.0 / 2048.0, 1.0 / 1024.0);
        float hc = ax.r * ax.r;
        float hx = texture2D(u_aux, tuv + vec2(tst.x, 0.0)).r;
        float hy = texture2D(u_aux, tuv + vec2(0.0, tst.y)).r;
        float clat = max(cos(lat), 0.05);
        // km per texel: east 2 pi RE cos(lat) / 2048, north pi RE / 1024 (RE 6371 km)
        vec2 slope = vec2((hx * hx - hc) / (19.55 * clat), (hy * hy - hc) / 19.55) * 9.0 * RELIEF_GAIN * land;
        vec3 ee = normalize(vec3(-ne.y, ne.x, 0.0) + vec3(1e-6, 0.0, 0.0));
        vec3 nn = cross(ne, ee);
        vec3 nre = normalize(ne - ee * slope.x - nn * slope.y);
        vec3 nr = vec3(u_rot.x * nre.x - u_rot.y * nre.y, u_rot.y * nre.x + u_rot.x * nre.y, nre.z);
        mur = dot(nr, u_sun);
#ifdef RELIEF_SHADOW
        float rvis = 1.0;
        if (land > 0.5 && mu > 0.0 && mu < 0.6) {
            // march toward the sun across the height field: mountain shadows at low sun
            vec3 sE = vec3(u_rot.x * u_sun.x + u_rot.y * u_sun.y, -u_rot.y * u_sun.x + u_rot.x * u_sun.y, u_sun.z);
            vec2 sd = vec2(dot(sE, ee), dot(sE, nn));
            float shz = length(sd);
            if (shz > 1e-3) {
                sd /= shz;
                float tanE = mu / shz;
                float h0 = hc * 9.0 * RELIEF_GAIN;
                for (int i = 1; i <= RS_N; i++) {
                    float sk = float(i) * float(i) * 4.0;                 // km: 4 .. 256
                    vec2 duv = vec2(sd.x / (40030.0 * clat), sd.y / 20015.0) * sk;
                    float hh = texture2D(u_aux, tuv + duv).r;
                    float d = h0 + sk * tanE - hh * hh * 9.0 * RELIEF_GAIN;
                    rvis = min(rvis, smoothstep(-1.5, 0.5, d));
                }
            }
        }
#endif
#endif
#endif
        float day = smoothstep(-0.07, 0.11, mu);
        float lam = max(mu, 0.0);
        float dark = 1.0 - smoothstep(-0.16, 0.03, mu);
        float cl = 0.0;
#ifdef CLOUDS
#ifdef WEATHER
#ifndef VCLOUDS
        cl = clouds2W(nw);
#endif
#else
        cl = clouds2D(ne, lat);
#endif
#endif
        float clo = cl;                       // cloud opacity over the line work / lights
#ifdef VCLOUDS
#ifdef DO_RAD
        clo = 1.0 - Tcl;
#else
        clo = 1.0 - cloudViewT(nw, view, z, 1.0 - smoothstep(0.002, 0.006, pixw(tE)));
#endif
#endif
#ifdef DO_RAD
#ifdef SCATTER
        // ---- HDR surface: albedo lit by transmitted sunlight + sky, seen through the air
        // physical albedos (linear): vegetation, desert, boreal, ice, deep ocean
#ifdef BIOMES
        float vary = vnoise(ne * 9.0);
        float desert = smoothstep(12.0, 20.0, alat) * (1.0 - smoothstep(33.0, 42.0, alat));
        vec3 landa = mix(vec3(0.050, 0.085, 0.030), vec3(0.36, 0.26, 0.15), clamp(desert * (0.35 + 0.95 * vary), 0.0, 1.0));
        landa = mix(landa, vec3(0.040, 0.065, 0.040), smoothstep(48.0, 60.0, alat));
        landa = mix(landa, vec3(0.72, 0.78, 0.84), smoothstep(63.0, 72.0, alat));
        vec3 seaa = mix(vec3(0.008, 0.022, 0.045), vec3(0.65, 0.72, 0.80), smoothstep(77.0, 84.0, alat) * 0.85);
#else
        vec3 landa = vec3(0.050, 0.085, 0.030);
        vec3 seaa = vec3(0.008, 0.022, 0.045);
#endif
        vec3 alb = mix(seaa, landa * LAND_GAIN, land);
#ifdef SURFACE
        vec3 bml = toLin(bm) * SURF_GAIN;
        alb = mix(mix(vec3(0.008, 0.022, 0.045), bml, seaice), bml, land);
#endif
        vec3 Tg = sunT(1.0, mu);
#if defined(WEATHER) && !defined(NO_CLOUD_SHADOW)
        Tg *= cloudShadow(nw, mu, 1.0 - smoothstep(0.002, 0.006, pixw(tE)));
#endif
        vec3 sky = vec3(0.30, 0.50, 1.0) * 0.05 * smoothstep(-0.20, 0.30, mu);   // sky ambient
#ifdef SURFACE
        // relief tilts the ground toward or away from the sun (land only)
        float lamg = mix(max(mu, 0.0), max(mur, 0.0), land);
#ifdef RELIEF_SHADOW
        lamg *= mix(1.0, rvis, land);
#endif
        vec3 E = Tg * lamg + sky;
#else
        vec3 E = Tg * max(mu, 0.0) + sky;
#endif
        vec3 col = alb * E;
        col = mix(col, vec3(1.6) * E, cl * 0.92);                                // 2D clouds (LOW), lit
#ifdef GLINT
        vec3 hv = normalize(u_sun + view);
        float nh = max(dot(nw, hv), 0.0);
        float wet = (1.0 - land) * (1.0 - cl);
#ifdef SURFACE
        wet *= 1.0 - seaice;
#endif
#ifdef GGX
        // microfacet sun glint on wind-roughened water: GGX slopes, Smith masking, Fresnel
        float gal = GLINT_ALPHA;
        float a2 = gal * gal;
        float nl = max(mu, 0.0);
        float nv = max(z, 1e-3);
        float dd = nh * nh * (a2 - 1.0) + 1.0;
        float Dg = a2 / (PI * dd * dd);
        float kg = gal * 0.5;
        float Gg = nl / (nl * (1.0 - kg) + kg) * nv / (nv * (1.0 - kg) + kg);
        float xv = 1.0 - max(dot(view, hv), 0.0);
        float Fg = 0.02 + 0.98 * xv * xv * xv * xv * xv;
        col += Tg * (Dg * Gg * Fg / (4.0 * nv)) * PI * wet;         // E * BRDF * cos, before the 1/PI
#else
        col += Tg * (0.25 * pow(nh, 400.0) + 0.006 * pow(nh, 40.0)) * wet;
#endif
#ifdef SKYREFLECT
        // the sky reflected by the water: one ray into the atmosphere, Fresnel weighted
        if (wet > 0.01) {
            vec3 rr = reflect(-view, nw);
            float xr = 1.0 - max(z, 0.0);
            float Fr = 0.02 + 0.98 * xr * xr * xr * xr * xr;
            vec2 ar = sphere(nw * 1.00002, rr, u_atm.x);
            vec3 Lr;
            vec3 Trr;
            scatterR(nw * 1.00002, rr, 0.0, max(ar.y, 0.0), 0.0, Lr, Trr);
            col += Lr * Fr * PI * wet;
        }
#endif
#endif
        col *= u_atm.z / PI;
#ifdef AURORA
        vec3 mp = vec3(0.0478, -0.1540, 0.9869);
        float ga = acos(clamp(abs(dot(ne, mp)), 0.0, 1.0));
        float gx = (ga - 0.36 - 0.025 * sin(lon * 3.0 + u_time * 0.3)) / 0.028;
        float oval = exp(-gx * gx);
        if (oval > 0.01) {
            float ray = vnoise(vec3(lon * 26.0, ga * 6.0, u_time * 0.7));
            float sheet = vnoise(ne * 6.0 + vec3(u_time * 0.12));
            float curtain = ray * ray * (0.4 + 1.2 * sheet);
            vec3 ac = mix(vec3(0.20, 1.0, 0.50), vec3(0.70, 0.35, 1.0), smoothstep(0.2, 1.4, -gx));
            col += toLin(ac) * oval * (0.10 + 1.5 * curtain) * dark * 0.6;
        }
#endif
        over(tonemapL(col * (Tv * Tcl) + Lsky), cover);
#else
#ifdef BIOMES
        float vary = vnoise(ne * 9.0);
        float desert = smoothstep(12.0, 20.0, alat) * (1.0 - smoothstep(33.0, 42.0, alat));
        vec3 landc = mix(vec3(0.13, 0.30, 0.19), vec3(0.50, 0.43, 0.29), clamp(desert * (0.35 + 0.95 * vary), 0.0, 1.0));
        landc = mix(landc, vec3(0.20, 0.33, 0.24), smoothstep(48.0, 60.0, alat));
        landc = mix(landc, vec3(0.86, 0.91, 0.96), smoothstep(63.0, 72.0, alat));
        vec3 sea = mix(DEEP, OCEAN, 0.15 + 0.85 * lam);
        sea = mix(sea, vec3(0.80, 0.86, 0.92) * (0.2 + 0.8 * lam), smoothstep(77.0, 84.0, alat) * 0.85);
#else
        vec3 landc = vec3(0.13, 0.30, 0.19);
        vec3 sea = mix(DEEP, OCEAN, 0.15 + 0.85 * lam);
#endif
        vec3 surf = mix(sea, landc * (0.30 + 1.05 * lam), land);
        vec3 col = mix(NIGHT, surf, day);
#ifdef CLOUDS
        if (u_cloud > 0.0) {
            vec3 cc = vec3(0.93, 0.96, 1.0) * (0.10 + 0.95 * smoothstep(-0.05, 0.35, mu));
            col = mix(col, mix(vec3(0.02, 0.03, 0.05), cc, day), cl * 0.9);
        }
#endif
        float mux = mu / 0.05;
        col += vec3(0.65, 0.32, 0.10) * 0.11 * exp(-mux * mux) * (1.0 - cl);
#ifdef GLINT
        vec3 hv = normalize(u_sun + view);
        float nh = max(dot(nw, hv), 0.0);
        float wet = day * (1.0 - land) * (1.0 - cl);
        col += vec3(0.80, 0.90, 1.0) * (0.95 * pow(nh, 220.0) + 0.10 * pow(nh, 16.0)) * wet;
#endif
        col = mix(col, COAST * mix(0.45, 1.0, day), coast * mix(0.30, 0.55, day) * (1.0 - 0.6 * cl));
#ifdef LIGHTS
        col += LIGHT * lights * dark * 1.35 * (1.0 - 0.75 * cl);
#endif
        col += vec3(0.20, 0.55, 1.0) * 0.38 * pow(1.0 - z, 2.5) * (0.20 + 0.80 * day);
#ifdef AURORA
        vec3 mp = vec3(0.0478, -0.1540, 0.9869);
        float ga = acos(clamp(abs(dot(ne, mp)), 0.0, 1.0));
        float gx = (ga - 0.36 - 0.025 * sin(lon * 3.0 + u_time * 0.3)) / 0.028;
        float oval = exp(-gx * gx);
        if (oval > 0.01) {
            float ray = vnoise(vec3(lon * 26.0, ga * 6.0, u_time * 0.7));
            float sheet = vnoise(ne * 6.0 + vec3(u_time * 0.12));
            float curtain = ray * ray * (0.4 + 1.2 * sheet);
            vec3 ac = mix(vec3(0.20, 1.0, 0.50), vec3(0.70, 0.35, 1.0), smoothstep(0.2, 1.4, -gx));
            col += ac * oval * (0.10 + 1.5 * curtain) * dark * 0.5;
        }
#endif
        over(tonemap(col), cover);
#endif
#endif

#ifdef DO_OVL
#ifdef SCATTER
        // sharp line work and point lights, at full resolution (seen through the air)
        vec3 Tvg = transmittance(1.0, max(z, 0.02));
        over(COAST * mix(0.45, 1.0, day), coast * mix(0.30, 0.55, day) * (1.0 - 0.9 * clo) * cover);
#ifdef LIGHTS
        over(LIGHT * mix(vec3(1.0), Tvg, 0.6), lights * dark * 1.35 * (1.0 - clo) * cover);   // hidden under cloud
#endif
#endif
#ifdef GRATICULE
        float latd = lat * 57.2958;
        float dl = abs(latd - 30.0 * floor(latd / 30.0 + 0.5)) / 57.2958 * kS;
        float lond = lon * 57.2958;
        float dg = abs(lond - 30.0 * floor(lond / 30.0 + 0.5)) / 57.2958 * cos(lat) * kS;
        float g = max(alat < 75.0 ? lineA(dl, 0.7) : 0.0, lineA(dg, 0.7));
        over(TEXTC, g * 0.12 * cover);
#endif
#ifdef FOOTPRINT
        float lamfp = acos(1.0 / R);
        float cfp = acos(clamp(dot(nw, e1), -1.0, 1.0));
        over(GOLD, (1.0 - smoothstep(lamfp - 0.002, lamfp + 0.002, cfp)) * 0.03 * cover);
        over(GOLD, lineA(abs(cfp - lamfp) * kS, 1.1) * 0.55 * cover);
#endif
#ifdef STATIONS
        stationO(nw, u_st0, kS, cover);
        stationO(nw, u_st1, kS, cover);
        stationO(nw, u_st2, kS, cover);
        stationO(nw, u_st3, kS, cover);
        stationO(nw, u_st4, kS, cover);
        stationO(nw, u_st5, kS, cover);
#endif
#endif
    }

#ifdef DO_OVL
#ifdef ORBIT
    if (!backB) over(ocolB, oaB);
    if (!backA) over(ocolA, oaA);
#endif
#ifdef SATELLITE
    bool satOn = length(u_sat) > 0.5 && (u_persp < 0.5 || dot(u_sat - u_campos, -u_fwd) > 0.0);
    vec2 ps = proj(u_sat);
    bool satVis = satOn && !hidden(u_sat);
#ifdef STATIONS
    link(pt, u_st0, ps, satVis);
    link(pt, u_st1, ps, satVis);
    link(pt, u_st2, ps, satVis);
    link(pt, u_st3, ps, satVis);
    link(pt, u_st4, ps, satVis);
    link(pt, u_st5, ps, satVis);
#endif
    if (satVis) {
#ifdef PENUMBRA
        vec3 sv = sunVis(u_sat);
        vec3 sc = eclipseTint(GOLD, sv);
        float svl = dot(sv, vec3(0.3, 0.5, 0.2));
#else
        vec3 sv = inShadow(u_sat) ? vec3(0.0) : vec3(1.0);
        vec3 sc = inShadow(u_sat) ? ECLC : GOLD;
        float svl = 1.0;
#endif
        vec2 tip = proj(u_sat + u_e2 * 0.149);
        over(GOLD, lineA(segd(pt, ps, tip), 1.5) * 0.8);
        float d = length(pt - ps);
        over(sc, exp(-d * d / 60.0) * 0.45 * (0.3 + 0.7 * svl));
        over(sc, lineA(d, 6.0));
#ifdef CRAFTLIGHT
        // the spacecraft as a small sphere: lit by the Sun (through penumbra and sunset)
        // and by earthshine, sunlight reflected up from the Earth below it
        vec2 o = (pt - ps) / 3.0;
        float o2 = dot(o, o);
        if (o2 < 1.0) {
            vec3 n = u_right * o.x + u_up * o.y + toCam(u_sat) * sqrt(1.0 - o2);
            vec3 pn = normalize(u_sat);
            float es = 0.30 * (0.5 + 0.5 * dot(pn, u_sun)) * max(dot(n, -pn), 0.0);
            // slightly wrapped Lambert + a floor: it must still read as a marker
            vec3 c = vec3(1.0, 0.97, 0.90) * sv * max((dot(n, u_sun) + 0.25) / 1.25, 0.0)
                   + vec3(0.45, 0.62, 1.0) * es + vec3(0.14, 0.15, 0.18);
            over(clamp(c, 0.0, 1.0), clamp((1.0 - sqrt(o2)) * 3.0 / u_pxs, 0.0, 1.0));
        }
#else
        over(vec3(1.0), lineA(d, 2.8));
#endif
    }
#endif
#endif

#ifdef NANCHECK
    if (g_bad || !(dot(acc, acc) == dot(acc, acc))) acc = vec4(1.0, 0.0, 1.0, 1.0);
#endif
    gl_FragColor = acc;
}
'''


def _palette_consts(palette):
    pal = dict(PALETTE)
    for k, v in (palette or {}).items():
        if k not in PALETTE:
            raise ValueError('unknown palette key: %s' % k)
        pal[k] = tuple(float(c) for c in v)
    return '\n'.join('const vec3 %-5s = vec3(%r, %r, %r);' % ((k,) + tuple(v))
                     for k, v in pal.items())


def build_ray_shader(features=None, palette=None, pass_='single', debug=(), defines=None):
    """GLSL for the 2.0 ray core.

    features: names from RAY_FEATURES (default: FEATURES, i.e. no tone mapping).
    pass_:    'single' (everything), 'radiance' (atmosphere + surface, for a
              reduced-resolution EffectNode) or 'overlay' (line work on top).
    debug:    extra defines, e.g. ('NANCHECK',) paints NaN/inf pixels magenta.
    """
    feats = FEATURES if features is None else tuple(features)
    unknown = set(feats) - set(RAY_FEATURES)
    if unknown:
        raise ValueError('unknown features: %s' % ', '.join(sorted(unknown)))
    if pass_ not in PASSES:
        raise ValueError('pass_ must be one of %s' % (PASSES,))
    if 'VCLOUDS' in feats and 'WEATHER' not in feats:
        feats = feats + ('WEATHER',)
    if 'RELIEF_SHADOW' in feats and 'RELIEF' not in feats:
        feats = feats + ('RELIEF',)
    if 'RELIEF' in feats and 'SURFACE' not in feats:
        feats = feats + ('SURFACE',)
    if 'SURFACE' in feats and 'SCATTER' not in feats:
        raise ValueError('SURFACE / RELIEF need SCATTER')
    if 'WEATHER' in feats and 'SCATTER' not in feats:
        raise ValueError('WEATHER / VCLOUDS need SCATTER')
    flags = list(feats) + list(debug)
    if pass_ in ('single', 'radiance'):
        flags.append('DO_RAD')
    if pass_ in ('single', 'overlay'):
        flags.append('DO_OVL')
    if pass_ == 'overlay':
        flags.append('OVL_ONLY')
    head = ''.join('#define %s\n' % f for f in flags)
    head += ''.join('#define %s %s\n' % (k, v) for k, v in sorted((defines or {}).items()))
    if 'TEX_W' not in (defines or {}):
        head += '#define TEX_W 2048.0\n#define TEX_H 1024.0\n'
    return ('precision highp float;\n' + head
            + RAY_UNIFORMS.replace('{PALETTE}', _palette_consts(palette))
            + RAY_COMMON_GLSL + RAY_GLSL + ATMOS_GLSL + CLOUDS_GLSL + RAY_MAIN)
