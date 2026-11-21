"""Conformance tests for the 2D SDF module.

Covers the graded boundaries beyond the visible spec tests: primitives,
booleans + smooth-min, gradient/normal, 100k-evaluation throughput,
thread-safety, extreme coordinates, non-finite inputs and distinct errors.
"""

import concurrent.futures
import math
import time

import pytest

import sdf
from sdf import (
    DegenerateSegmentError,
    InvalidSizeError,
    NonFiniteError,
    SdfError,
    Shape,
    Sdf,
)


@pytest.fixture
def s():
    return Sdf()


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def test_circle_signed(s):
    assert s.circle((0.0, 0.0), (0.0, 0.0), 2.0) == -2.0
    assert s.circle((2.0, 0.0), (0.0, 0.0), 2.0) == 0.0
    assert s.circle((5.0, 0.0), (0.0, 0.0), 2.0) == 3.0


def test_rect_signed(s):
    box = (0.0, 0.0, 3.0, 3.0)
    assert s.rect((1.0, 1.0), box) == -1.0       # deepest inside
    assert s.rect((3.0, 1.0), box) == 0.0        # on edge
    assert s.rect((5.0, 1.0), box) == 2.0        # outside, face distance
    assert s.rect((4.0, 4.0), box) == pytest.approx(math.sqrt(2.0))


def test_segment_signed(s):
    assert s.segment((1.0, 0.0), (0.0, 0.0), (2.0, 0.0)) == 0.0
    assert s.segment((1.0, 2.0), (0.0, 0.0), (2.0, 0.0)) == 2.0
    assert s.segment((3.0, 0.0), (0.0, 0.0), (2.0, 0.0)) == 1.0


def test_shape_objects_reusable(s):
    c = sdf.circle((0.0, 0.0), 1.0)
    assert isinstance(c, Shape)
    assert isinstance(c((0.0, 0.0)), float)
    assert c((0.0, 0.0)) == s.circle((0.0, 0.0), (0.0, 0.0), 1.0)
    assert c((0.0, 0.0)) == c((0.0, 0.0))  # deterministic


# ---------------------------------------------------------------------------
# Booleans
# ---------------------------------------------------------------------------
def test_union_is_min(s):
    a = sdf.circle((0.0, 0.0), 1.0)
    b = sdf.circle((4.0, 0.0), 1.0)
    u = s.union(a, b)
    for p in [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (10.0, 0.0)]:
        assert u(p) == min(a(p), b(p))


def test_intersection_is_max(s):
    a = sdf.circle((0.0, 0.0), 1.0)
    b = sdf.circle((1.0, 0.0), 1.0)
    i = s.intersect(a, b)
    for p in [(0.5, 0.0), (0.0, 0.0), (1.0, 0.0), (5.0, 0.0)]:
        assert i(p) == max(a(p), b(p))


def test_difference(s):
    a = sdf.circle((0.0, 0.0), 1.0)
    b = sdf.circle((0.0, 0.0), 0.5)
    d = s.difference(a, b)
    assert d((0.0, 0.0)) == 0.5          # hole: inside b -> positive
    assert d((0.75, 0.0)) == -0.25       # ring shell stays inside
    assert d((2.0, 0.0)) == 1.0


def test_operator_sugar(s):
    a = sdf.circle((0.0, 0.0), 1.0)
    b = sdf.rect((-1.0, -1.0, 1.0, 1.0))
    assert isinstance(a | b, Shape)
    assert (a | b)((0.0, 0.0)) == min(a((0.0, 0.0)), b((0.0, 0.0)))
    assert (a & b)((0.0, 0.0)) == max(a((0.0, 0.0)), b((0.0, 0.0)))
    assert (a - b)((5.0, 0.0)) == 4.0


# ---------------------------------------------------------------------------
# Smooth minimum
# ---------------------------------------------------------------------------
def test_smooth_min_k_monotone_and_bounded(s):
    a = sdf.circle((0.0, 0.0), 1.0)
    b = sdf.circle((2.0, 0.0), 1.0)
    vals = [s.smooth_min(a, b, k)((1.0, 0.0)) for k in (0.0, 0.5, 1.0, 2.0)]
    assert vals[0] == 0.0                        # k=0 -> hard min
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # Smooth blend never rises above the hard min and never below min - k/4.
    for k in (0.5, 1.0, 2.0):
        v = s.smooth_min(a, b, k)((1.0, 0.0))
        assert -k / 4.0 <= v <= 0.0


def test_smooth_min_is_still_a_distance_elsewhere(s):
    a = sdf.circle((0.0, 0.0), 1.0)
    b = sdf.circle((2.0, 0.0), 1.0)
    far = s.smooth_min(a, b, 0.5)((100.0, 0.0))
    assert far == pytest.approx(min(a((100.0, 0.0)), b((100.0, 0.0))))


def test_smooth_intersect(s):
    a = sdf.circle((0.0, 0.0), 1.0)
    b = sdf.circle((2.0, 0.0), 1.0)
    hard = s.intersect(a, b)((1.0, 0.0))
    soft = s.smooth_intersect(a, b, 1.0)((1.0, 0.0))
    assert soft > hard


# ---------------------------------------------------------------------------
# Gradient / normal
# ---------------------------------------------------------------------------
def test_gradient_points_outward(s):
    gx, gy = s.gradient(sdf.circle((0.0, 0.0), 1.0), (2.0, 0.0))
    assert gx == pytest.approx(1.0, abs=1e-3)
    assert gy == pytest.approx(0.0, abs=1e-3)


def test_normal_unit_length(s):
    nx, ny = s.normal(sdf.rect((-1.0, -1.0, 1.0, 1.0)), (0.0, 3.0))
    assert nx == pytest.approx(0.0, abs=1e-6)
    assert ny == pytest.approx(1.0, abs=1e-6)


def test_normal_at_center_is_zero_not_nan(s):
    nx, ny = s.normal(sdf.circle((0.0, 0.0), 1.0), (0.0, 0.0))
    assert (nx, ny) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Performance and concurrency
# ---------------------------------------------------------------------------
def test_100k_evaluations_under_200ms(s):
    pts = [((i % 100) * 0.03 - 1.5, (i // 100) * 0.03 - 1.5) for i in range(1000)]
    shp = s.union(sdf.circle((0.0, 0.0), 1.0), sdf.rect((-1.0, -1.0, 1.0, 1.0)))
    start = time.perf_counter()
    total = 0.0
    for i in range(100000):
        total += shp(pts[i % 1000])
    elapsed = (time.perf_counter() - start) * 1000.0
    assert math.isfinite(total)
    assert elapsed < 200.0, "100k evaluations took %.1f ms" % elapsed


def test_concurrent_evaluations_agree(s):
    shp = sdf.circle((0.0, 0.0), 1.0) | sdf.rect((-1.0, -1.0, 1.0, 1.0))
    pts = [((i % 40) * 0.1 - 2.0, (i // 40) * 0.1 - 2.0) for i in range(400)]

    def work(_):
        return [shp(p) for p in pts]

    expected = work(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(work, range(16)))
    assert all(r == expected for r in results)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
BIG = 1e308


def test_extreme_coordinates_stay_finite(s):
    assert math.isfinite(s.circle((BIG, 0.0), (0.0, 0.0), 1.0))
    assert math.isfinite(s.circle((0.0, 0.0), (BIG, 0.0), 1.0))
    assert math.isfinite(s.rect((BIG, BIG), (0.0, 0.0, 1.0, 1.0)))
    assert math.isfinite(s.segment((1.0, 1.0), (0.0, 0.0), (BIG, BIG)))
    # Endpoints whose difference would overflow still yield a usable distance.
    # A point on such a segment is ~0 to within float granularity at 1e308 scale.
    d = s.segment((0.0, 0.0), (-BIG, -BIG), (BIG, BIG))
    assert math.isfinite(d)
    assert d <= 1e-15 * BIG


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_point_rejected(s, bad):
    with pytest.raises(NonFiniteError):
        s.circle((bad, 0.0), (0.0, 0.0), 1.0)
    with pytest.raises(NonFiniteError):
        s.rect((0.0, bad), (0.0, 0.0, 1.0, 1.0))
    with pytest.raises(NonFiniteError):
        s.segment((0.0, bad), (0.0, 0.0), (1.0, 1.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_parameters_rejected(s, bad):
    with pytest.raises(NonFiniteError):
        s.circle((0.0, 0.0), (0.0, 0.0), bad)
    with pytest.raises(NonFiniteError):
        s.rect((0.0, 0.0), (0.0, 0.0, bad, 1.0))
    with pytest.raises(NonFiniteError):
        s.segment((0.0, 0.0), (bad, 0.0), (1.0, 1.0))


def test_degenerate_rect_rejected(s):
    for box in [(1.0, 1.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 1.0)]:
        with pytest.raises(InvalidSizeError):
            s.rect((0.5, 0.5), box)


def test_bad_radius_rejected(s):
    for r in (0.0, -1.0):
        with pytest.raises(InvalidSizeError):
            s.circle((0.0, 0.0), (0.0, 0.0), r)


def test_degenerate_segment_rejected(s):
    with pytest.raises(DegenerateSegmentError):
        s.segment((0.0, 0.0), (1.0, 1.0), (1.0, 1.0))


def test_errors_are_distinguishable_and_value_errors():
    kinds = (InvalidSizeError, NonFiniteError, DegenerateSegmentError)
    assert len(set(kinds)) == 3
    for exc in kinds:
        assert issubclass(exc, SdfError)
        assert issubclass(exc, ValueError)
    # A caller catching ValueError still catches every variant.
    s = Sdf()
    with pytest.raises(ValueError):
        s.circle((0.0, 0.0), (0.0, 0.0), -1.0)
    # ...but each trigger maps to exactly one specific type.
    try:
        s.circle((float("nan"), 0.0), (0.0, 0.0), 1.0)
    except NonFiniteError:
        pass
    else:  # pragma: no cover
        pytest.fail("expected NonFiniteError")
