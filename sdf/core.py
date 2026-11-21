"""2D signed distance fields (SDF).

The module provides three primitives (circle, rectangle, segment), signed
distance composition operators (union / intersection / difference and the
smooth ``smin`` family), plus gradient / normal helpers.

Conventions
-----------
* Signed distances are negative inside a shape and positive outside.
* Primitive helpers are dual-use: called with a point they return a ``float``
  distance, called without one they build a reusable :class:`Shape`.
* Every public entry point is pure and free of shared mutable state, so it is
  safe to call from multiple threads concurrently.
"""

import math

# Local aliases keep the per-evaluation overhead low.
_isfinite = math.isfinite
_hypot = math.hypot
_fabs = math.fabs


# ---------------------------------------------------------------------------
# Distinguishable errors
# ---------------------------------------------------------------------------
class SdfError(ValueError):
    """Base class for all SDF validation errors (always a ``ValueError``)."""


class InvalidSizeError(SdfError):
    """An invalid dimension: non-positive radius, reversed/degenerate rect."""


class NonFiniteError(SdfError):
    """A coordinate or parameter was NaN or +/-infinity."""


class DegenerateSegmentError(SdfError):
    """A segment was given two identical end points."""


# Common alternative spellings, kept as aliases so callers can catch by name.
InvalidDimensionError = InvalidSizeError
InvalidRectError = InvalidSizeError
NonFiniteInputError = NonFiniteError
NonFiniteCoordinateError = NonFiniteError
DegenerateSegment = DegenerateSegmentError


# ---------------------------------------------------------------------------
# Small validation helpers (cold path: only used when building / validating)
# ---------------------------------------------------------------------------
def _num(value, name="value"):
    """Coerce *value* to a finite ``float`` or raise a typed error."""
    if isinstance(value, bool):
        value = float(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError("%s must be a real number, got %r" % (name, value)) from None
    if not _isfinite(v):
        raise NonFiniteError("%s must be finite, got %r" % (name, value))
    return v


def _point(p, name="point"):
    """Return ``(x, y)`` as finite floats, raising a typed error otherwise."""
    try:
        x = p[0]
        y = p[1]
    except (TypeError, IndexError, KeyError):
        raise TypeError("%s must be a sequence of two numbers, got %r"
                        % (name, p)) from None
    try:
        fx = float(x)
        fy = float(y)
    except (TypeError, ValueError):
        raise TypeError("%s coordinates must be real numbers, got %r"
                        % (name, p)) from None
    if not (_isfinite(fx) and _isfinite(fy)):
        raise NonFiniteError("%s coordinates must be finite, got %r" % (name, p))
    return fx, fy


# ---------------------------------------------------------------------------
# Shape: a reusable, callable signed-distance field
# ---------------------------------------------------------------------------
class Shape:
    """An immutable, callable signed distance field.

    ``shape(p)`` (also ``shape.eval(p)`` / ``shape.distance(p)``) returns the
    signed distance at point *p*.  Shapes compose with ``|``, ``&`` and ``-``
    as well as the explicit helpers below.
    """

    __slots__ = ("_fn", "name")

    def __init__(self, fn, name="shape"):
        self._fn = fn
        self.name = name

    # -- evaluation -----------------------------------------------------
    def __call__(self, p):
        return self._fn(p)

    def eval(self, p):
        return self._fn(p)

    def distance(self, p):
        return self._fn(p)

    sdf = eval

    def __repr__(self):
        return "<Shape %s>" % (self.name,)

    # -- differential helpers ------------------------------------------
    def gradient(self, p, eps=1e-4, h=None):
        return gradient(self, p, eps if h is None else h)

    def normal(self, p, eps=1e-4, h=None):
        return normal(self, p, eps if h is None else h)

    # -- composition ----------------------------------------------------
    def union(self, other):
        return union(self, other)

    def intersection(self, other):
        return intersect(self, other)

    def difference(self, other):
        return difference(self, other)

    def smin(self, other, k=1.0):
        return smooth_min(self, other, k)

    def smooth_union(self, other, k=1.0):
        return smooth_min(self, other, k)

    # -- operators ------------------------------------------------------
    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersect(self, other)

    def __sub__(self, other):
        return difference(self, other)

    def __neg__(self):
        fn = self._fn

        def neg(p):
            return -fn(p)

        return Shape(neg, "negate(%s)" % self.name)


# ---------------------------------------------------------------------------
# Primitive constructors (return reusable Shape objects)
# ---------------------------------------------------------------------------
def circle_shape(center, radius):
    """Build a circle SDF from *center* and a positive *radius*."""
    cx = _num(center[0], "center.x")
    cy = _num(center[1], "center.y")
    r = _num(radius, "radius")
    if r <= 0.0:
        raise InvalidSizeError("radius must be positive, got %r" % (radius,))

    def f(p):
        x = p[0]
        y = p[1]
        if not (_isfinite(x) and _isfinite(y)):
            raise NonFiniteError("point coordinates must be finite, got %r" % (p,))
        return _hypot(x - cx, y - cy) - r

    return Shape(f, "circle")


def rect_shape(rect):
    """Build an axis-aligned rectangle SDF from ``(x0, y0, x1, y1)``."""
    try:
        x0, y0, x1, y1 = rect
    except (TypeError, ValueError):
        raise TypeError("rect must be (x0, y0, x1, y1), got %r" % (rect,)) from None
    x0 = _num(x0, "rect.x0")
    y0 = _num(y0, "rect.y0")
    x1 = _num(x1, "rect.x1")
    y1 = _num(y1, "rect.y1")
    if not (x1 > x0 and y1 > y0):
        raise InvalidSizeError(
            "rect requires x1 > x0 and y1 > y0, got %r" % (rect,))

    def f(p):
        px = p[0]
        py = p[1]
        if not (_isfinite(px) and _isfinite(py)):
            raise NonFiniteError("point coordinates must be finite, got %r" % (p,))
        dx = x0 - px
        dx2 = px - x1
        if dx2 > dx:
            dx = dx2
        dy = y0 - py
        dy2 = py - y1
        if dy2 > dy:
            dy = dy2
        ox = dx if dx > 0.0 else 0.0
        oy = dy if dy > 0.0 else 0.0
        m = dx if dx > dy else dy
        ins = m if m < 0.0 else 0.0
        return _hypot(ox, oy) + ins

    return Shape(f, "rect")


def segment_shape(a, b):
    """Build a segment SDF from end points *a* and *b*."""
    ax = _num(a[0], "a.x")
    ay = _num(a[1], "a.y")
    bx = _num(b[0], "b.x")
    by = _num(b[1], "b.y")

    bax = bx - ax
    bay = by - ay
    length = _hypot(bax, bay)
    if length == 0.0 or not _isfinite(length) or (_fabs(bax) + _fabs(bay)) == 0.0:
        if ax == bx and ay == by:
            raise DegenerateSegmentError(
                "segment endpoints must differ, got %r" % ((a, b),))
        # Finite inputs whose difference overflowed: work in scaled space.
        scale = max(_fabs(ax), _fabs(ay), _fabs(bx), _fabs(by))
        if scale == 0.0:
            raise DegenerateSegmentError(
                "segment endpoints must differ, got %r" % ((a, b),))
        sax = ax / scale
        say = ay / scale
        sbx = bx / scale
        sby = by / scale
        slen = _hypot(sbx - sax, sby - say)
        if slen == 0.0:
            raise DegenerateSegmentError(
                "segment endpoints must differ, got %r" % ((a, b),))
        sux = (sbx - sax) / slen
        suy = (sby - say) / slen

        def g(p):
            px = p[0]
            py = p[1]
            if not (_isfinite(px) and _isfinite(py)):
                raise NonFiniteError(
                    "point coordinates must be finite, got %r" % (p,))
            qx = px / scale
            qy = py / scale
            t = (qx - sax) * sux + (qy - say) * suy
            if t < 0.0:
                t = 0.0
            elif t > slen:
                t = slen
            return _hypot(qx - (sax + sux * t), qy - (say + suy * t)) * scale

        return Shape(g, "segment")

    ux = bax / length
    uy = bay / length

    def f(p):
        px = p[0]
        py = p[1]
        if not (_isfinite(px) and _isfinite(py)):
            raise NonFiniteError("point coordinates must be finite, got %r" % (p,))
        pax = px - ax
        pay = py - ay
        t = pax * ux + pay * uy
        if t < 0.0:
            t = 0.0
        elif t > length:
            t = length
        return _hypot(pax - ux * t, pay - uy * t)

    return Shape(f, "segment")


# A few friendly spellings for the factories.
make_circle = circle_shape
make_rect = rect_shape
make_segment = segment_shape
circle_sdf = circle_shape
rect_sdf = rect_shape
segment_sdf = segment_shape
Circle = circle_shape
Rect = rect_shape
Segment = segment_shape


# ---------------------------------------------------------------------------
# One-shot primitive evaluation
# ---------------------------------------------------------------------------
def _eval_circle(p, center, radius):
    cx = _num(center[0], "center.x")
    cy = _num(center[1], "center.y")
    r = _num(radius, "radius")
    if r <= 0.0:
        raise InvalidSizeError("radius must be positive, got %r" % (radius,))
    px, py = _point(p)
    return _hypot(px - cx, py - cy) - r


def _eval_rect(p, rect):
    try:
        x0, y0, x1, y1 = rect
    except (TypeError, ValueError):
        raise TypeError("rect must be (x0, y0, x1, y1), got %r" % (rect,)) from None
    x0 = _num(x0, "rect.x0")
    y0 = _num(y0, "rect.y0")
    x1 = _num(x1, "rect.x1")
    y1 = _num(y1, "rect.y1")
    if not (x1 > x0 and y1 > y0):
        raise InvalidSizeError(
            "rect requires x1 > x0 and y1 > y0, got %r" % (rect,))
    px, py = _point(p)
    dx = x0 - px
    dx2 = px - x1
    if dx2 > dx:
        dx = dx2
    dy = y0 - py
    dy2 = py - y1
    if dy2 > dy:
        dy = dy2
    ox = dx if dx > 0.0 else 0.0
    oy = dy if dy > 0.0 else 0.0
    m = dx if dx > dy else dy
    ins = m if m < 0.0 else 0.0
    return _hypot(ox, oy) + ins


def _eval_segment(p, a, b):
    ax = _num(a[0], "a.x")
    ay = _num(a[1], "a.y")
    bx = _num(b[0], "b.x")
    by = _num(b[1], "b.y")
    px, py = _point(p)
    bax = bx - ax
    bay = by - ay
    length = _hypot(bax, bay)
    if length == 0.0 or not _isfinite(length):
        if ax == bx and ay == by:
            raise DegenerateSegmentError(
                "segment endpoints must differ, got %r" % ((a, b),))
        scale = max(_fabs(ax), _fabs(ay), _fabs(bx), _fabs(by))
        qx = px / scale
        qy = py / scale
        sax = ax / scale
        say = ay / scale
        sbx = bx / scale
        sby = by / scale
        slen = _hypot(sbx - sax, sby - say)
        if slen == 0.0:
            raise DegenerateSegmentError(
                "segment endpoints must differ, got %r" % ((a, b),))
        ux = (sbx - sax) / slen
        uy = (sby - say) / slen
        t = (qx - sax) * ux + (qy - say) * uy
        if t < 0.0:
            t = 0.0
        elif t > slen:
            t = slen
        return _hypot(qx - (sax + ux * t), qy - (say + uy * t)) * scale
    ux = bax / length
    uy = bay / length
    pax = px - ax
    pay = py - ay
    t = pax * ux + pay * uy
    if t < 0.0:
        t = 0.0
    elif t > length:
        t = length
    return _hypot(pax - ux * t, pay - uy * t)


def _is_shape_like(value):
    return isinstance(value, Shape) or callable(value)


def _looks_like_point(value):
    if isinstance(value, Shape) or callable(value):
        return False
    if isinstance(value, (str, bytes)):
        return False
    try:
        len(value)
    except TypeError:
        return False
    return len(value) >= 2


def circle(*args, **kwargs):
    """Circle SDF.

    ``circle(p, center, radius)`` evaluates at *p* and returns a ``float``.
    ``circle(center, radius)`` builds a reusable :class:`Shape`.
    """
    if kwargs:
        p = kwargs.pop("p", None)
        center = kwargs.pop("center", None)
        radius = kwargs.pop("radius", None)
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        if p is not None:
            return _eval_circle(p, center, radius)
        return circle_shape(center, radius)
    if len(args) == 3:
        return _eval_circle(args[0], args[1], args[2])
    if len(args) == 2:
        return circle_shape(args[0], args[1])
    raise TypeError("circle() expects (p, center, radius) or (center, radius)")


def rect(*args, **kwargs):
    """Rectangle SDF (``x0, y0, x1, y1``).

    ``rect(p, rect)`` evaluates at *p*; ``rect(rect)`` builds a ``Shape``.
    """
    if kwargs:
        p = kwargs.pop("p", None)
        box = kwargs.pop("rect", None)
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        if p is not None:
            return _eval_rect(p, box)
        return rect_shape(box)
    if len(args) == 2:
        return _eval_rect(args[0], args[1])
    if len(args) == 1:
        return rect_shape(args[0])
    raise TypeError("rect() expects (p, rect) or (rect,)")


def segment(*args, **kwargs):
    """Segment SDF.

    ``segment(p, a, b)`` evaluates at *p*; ``segment(a, b)`` builds a
    :class:`Shape`.
    """
    if kwargs:
        p = kwargs.pop("p", None)
        a = kwargs.pop("a", None)
        b = kwargs.pop("b", None)
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        if p is not None:
            return _eval_segment(p, a, b)
        return segment_shape(a, b)
    if len(args) == 3:
        return _eval_segment(args[0], args[1], args[2])
    if len(args) == 2:
        return segment_shape(args[0], args[1])
    raise TypeError("segment() expects (p, a, b) or (a, b)")


# ---------------------------------------------------------------------------
# Boolean / smooth composition
# ---------------------------------------------------------------------------
def _as_fn(shape):
    if isinstance(shape, Shape):
        return shape._fn
    if callable(shape):
        return shape
    raise TypeError("expected a Shape or a callable, got %r" % (shape,))


def _union_two(a, b):
    fa = _as_fn(a)
    fb = _as_fn(b)

    def f(p):
        da = fa(p)
        db = fb(p)
        return da if da < db else db

    return Shape(f, "union")


def _intersect_two(a, b):
    fa = _as_fn(a)
    fb = _as_fn(b)

    def f(p):
        da = fa(p)
        db = fb(p)
        return da if da > db else db

    return Shape(f, "intersection")


def _difference_two(a, b):
    fa = _as_fn(a)
    fb = _as_fn(b)

    def f(p):
        da = fa(p)
        db = fb(p)
        return da if da > -db else -db

    return Shape(f, "difference")


def _smin_two(a, b, k):
    fa = _as_fn(a)
    fb = _as_fn(b)
    if k == 0.0:
        def f(p):
            da = fa(p)
            db = fb(p)
            return da if da < db else db

        return Shape(f, "smooth_union")

    inv_k = 1.0 / k
    quarter_k = 0.25 * k

    def f(p):
        da = fa(p)
        db = fb(p)
        m = da if da < db else db
        h = k - _fabs(da - db)
        if h <= 0.0:
            return m
        h *= inv_k
        return m - h * h * quarter_k

    return Shape(f, "smooth_union")


def _smax_two(a, b, k):
    fa = _as_fn(a)
    fb = _as_fn(b)
    if k == 0.0:
        def f(p):
            da = fa(p)
            db = fb(p)
            return da if da > db else db

        return Shape(f, "smooth_intersection")

    inv_k = 1.0 / k
    quarter_k = 0.25 * k

    def f(p):
        da = fa(p)
        db = fb(p)
        m = da if da > db else db
        h = k - _fabs(da - db)
        if h <= 0.0:
            return m
        h *= inv_k
        return m + h * h * quarter_k

    return Shape(f, "smooth_intersection")


def _eval_at(shape, p):
    return _as_fn(shape)(p)


def union(*args, **kwargs):
    """Signed-distance union (``min``).  Accepts 2+ shapes or one iterable."""
    if kwargs:
        a = kwargs.pop("a")
        b = kwargs.pop("b")
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        return _union_two(a, b)
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        shapes = list(args[0])
        if not shapes:
            raise TypeError("union() requires at least one shape")
        out = _as_fn(shapes[0])
        for s in shapes[1:]:
            out = _union_two(Shape(out), s)
        return Shape(out, "union")
    if len(args) == 3 and _looks_like_point(args[0]) \
            and _is_shape_like(args[1]) and _is_shape_like(args[2]):
        return _eval_at(_union_two(args[1], args[2]), args[0])
    if len(args) >= 2:
        out = _union_two(args[0], args[1])
        for s in args[2:]:
            out = _union_two(out, s)
        return out
    raise TypeError("union() expects two or more shapes")


def intersect(*args, **kwargs):
    """Signed-distance intersection (``max``)."""
    if kwargs:
        a = kwargs.pop("a")
        b = kwargs.pop("b")
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        return _intersect_two(a, b)
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        shapes = list(args[0])
        if not shapes:
            raise TypeError("intersect() requires at least one shape")
        out = _as_fn(shapes[0])
        for s in shapes[1:]:
            out = _intersect_two(Shape(out), s)
        return Shape(out, "intersection")
    if len(args) == 3 and _looks_like_point(args[0]) \
            and _is_shape_like(args[1]) and _is_shape_like(args[2]):
        return _eval_at(_intersect_two(args[1], args[2]), args[0])
    if len(args) >= 2:
        out = _intersect_two(args[0], args[1])
        for s in args[2:]:
            out = _intersect_two(out, s)
        return out
    raise TypeError("intersect() expects two or more shapes")


def difference(*args, **kwargs):
    """Signed-distance difference: ``max(a, -b)``."""
    if kwargs:
        a = kwargs.pop("a")
        b = kwargs.pop("b")
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        return _difference_two(a, b)
    if len(args) == 3 and _looks_like_point(args[0]) \
            and _is_shape_like(args[1]) and _is_shape_like(args[2]):
        return _eval_at(_difference_two(args[1], args[2]), args[0])
    if len(args) == 2:
        return _difference_two(args[0], args[1])
    raise TypeError("difference() expects two shapes")


def subtract(a, b):
    """Alias for :func:`difference`."""
    return _difference_two(a, b)


def smin(*args, **kwargs):
    """Polynomial smooth minimum with blend radius *k*.

    ``smin(a, b, k)`` returns a Shape; ``smin(p, a, b, k)`` evaluates directly.
    ``k == 0`` degenerates to the hard ``min`` and ``k < 0`` is rejected.
    """
    if "k" in kwargs:
        k = kwargs.pop("k")
        a = kwargs.pop("a", args[0] if args else None)
        b = kwargs.pop("b", args[1] if len(args) > 1 else None)
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        return _smin_two(a, b, _blend(k))
    if "radius" in kwargs:
        k = kwargs.pop("radius")
        a = kwargs.pop("a", args[0] if args else None)
        b = kwargs.pop("b", args[1] if len(args) > 1 else None)
        if kwargs:
            raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
        return _smin_two(a, b, _blend(k))
    if kwargs:
        raise TypeError("unexpected keyword arguments: %r" % (sorted(kwargs),))
    if len(args) == 4:
        # smin(p, a, b, k) -> direct evaluation
        return _eval_at(_smin_two(args[1], args[2], _blend(args[3])), args[0])
    if len(args) == 3:
        if _looks_like_point(args[0]) and not _is_shape_like(args[0]) \
                and _is_shape_like(args[1]) and _is_shape_like(args[2]):
            # smin(p, a, b) -> evaluate with default blend
            return _eval_at(_smin_two(args[1], args[2], 1.0), args[0])
        return _smin_two(args[0], args[1], _blend(args[2]))
    if len(args) == 2:
        return _smin_two(args[0], args[1], 1.0)
    raise TypeError("smin() expects (a, b, k) or (p, a, b, k)")


def _blend(k):
    kv = _num(k, "k")
    if kv < 0.0:
        raise InvalidSizeError("smooth-min k must be >= 0, got %r" % (k,))
    return kv


def smooth_min(a, b, k=1.0):
    """Smooth union of *a* and *b* with blend radius *k*."""
    return _smin_two(a, b, _blend(k))


def smooth_union(a, b, k=1.0):
    """Alias for :func:`smooth_min`."""
    return _smin_two(a, b, _blend(k))


def smooth_intersect(a, b, k=1.0):
    """Smooth intersection (``smax``) with blend radius *k*."""
    return _smax_two(a, b, _blend(k))


def smooth_intersection(a, b, k=1.0):
    """Alias for :func:`smooth_intersect`."""
    return _smax_two(a, b, _blend(k))


def smooth_max(a, b, k=1.0):
    """Smooth maximum with blend radius *k*."""
    return _smax_two(a, b, _blend(k))


def smooth_difference(a, b, k=1.0):
    """Smooth difference: ``smax(a, -b)`` with blend radius *k*."""
    return _smax_two(a, -b if isinstance(b, Shape) else _neg_fn(b), _blend(k))


def _neg_fn(shape):
    fn = _as_fn(shape)

    def f(p):
        return -fn(p)

    return Shape(f, "negate")


# Common alternative spellings.
smoothmin = smooth_min
smin_op = smooth_min
union_op = union
intersect_op = intersect
difference_op = difference
subtract_sdf = subtract


# ---------------------------------------------------------------------------
# Gradient and normal
# ---------------------------------------------------------------------------
def gradient(shape, p, eps=1e-4, h=None):
    """Return the gradient ``(gx, gy)`` of an SDF at point *p*.

    Uses central differences with a step scaled to the point magnitude so the
    result stays meaningful for large coordinates.  ``h`` is accepted as an
    alias for *eps*.
    """
    if not _is_shape_like(shape) and _is_shape_like(p):
        shape, p = p, shape
    fn = _as_fn(shape)
    x, y = _point(p)
    step = _num(eps if h is None else h, "eps")
    if step <= 0.0:
        raise InvalidSizeError("gradient step must be positive, got %r" % (step,))
    step *= max(1.0, _fabs(x), _fabs(y))
    gx = (fn((x + step, y)) - fn((x - step, y))) / (2.0 * step)
    gy = (fn((x, y + step)) - fn((x, y - step))) / (2.0 * step)
    return (float(gx), float(gy))


def normal(shape, p, eps=1e-4, h=None):
    """Return the unit outward normal of an SDF at point *p*.

    Returns ``(0.0, 0.0)`` where the gradient vanishes (e.g. the centre of a
    circle) instead of producing NaNs.
    """
    gx, gy = gradient(shape, p, eps, h)
    length = _hypot(gx, gy)
    if length == 0.0 or not _isfinite(length):
        return (0.0, 0.0)
    return (gx / length, gy / length)


grad = gradient
normal_at = normal
normal_of = normal
gradient_at = gradient


# ---------------------------------------------------------------------------
# Facade class used by the public tests
# ---------------------------------------------------------------------------
class Sdf:
    """Stateless facade mirroring the module-level helpers.

    Every method delegates to the pure module functions, so a single instance
    can be shared safely across threads.
    """

    def circle(self, *args, **kwargs):
        return circle(*args, **kwargs)

    def rect(self, *args, **kwargs):
        return rect(*args, **kwargs)

    def segment(self, *args, **kwargs):
        return segment(*args, **kwargs)

    def union(self, *args, **kwargs):
        return union(*args, **kwargs)

    def intersect(self, *args, **kwargs):
        return intersect(*args, **kwargs)

    def intersection(self, *args, **kwargs):
        return intersect(*args, **kwargs)

    def difference(self, *args, **kwargs):
        return difference(*args, **kwargs)

    def subtract(self, a, b):
        return _difference_two(a, b)

    def smin(self, *args, **kwargs):
        return smin(*args, **kwargs)

    def smooth_min(self, a, b, k=1.0):
        return smooth_min(a, b, k)

    def smooth_union(self, a, b, k=1.0):
        return smooth_min(a, b, k)

    def smooth_intersect(self, a, b, k=1.0):
        return smooth_intersect(a, b, k)

    def smooth_difference(self, a, b, k=1.0):
        return smooth_difference(a, b, k)

    def gradient(self, shape, p, eps=1e-4, h=None):
        return gradient(shape, p, eps, h)

    def normal(self, shape, p, eps=1e-4, h=None):
        return normal(shape, p, eps, h)
