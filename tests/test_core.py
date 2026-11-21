import math

import pytest

import sdf

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return sdf.Sdf()


def _check_circle_center():
    return _s().circle((0.0, 0.0), (0.0, 0.0), 2.0) == -2.0


def _check_circle_edge():
    return _s().circle((2.0, 0.0), (0.0, 0.0), 2.0) == 0.0


def _check_circle_out():
    return _s().circle((5.0, 0.0), (0.0, 0.0), 2.0) == 3.0


def _check_rect_inside():
    return _s().rect((1.0, 1.0), (0.0, 0.0, 3.0, 3.0)) < 0.0


def _check_rect_outside():
    return _s().rect((5.0, 1.0), (0.0, 0.0, 3.0, 3.0)) == 2.0


def _check_rect_edge():
    return _s().rect((3.0, 1.0), (0.0, 0.0, 3.0, 3.0)) == 0.0


def _check_seg_on():
    return _s().segment((1.0, 0.0), (0.0, 0.0), (2.0, 0.0)) == 0.0


def _check_seg_dist():
    return _s().segment((1.0, 2.0), (0.0, 0.0), (2.0, 0.0)) == 2.0


def _check_seg_end():
    return _s().segment((3.0, 0.0), (0.0, 0.0), (2.0, 0.0)) == 1.0


def _check_bad_radius():
    return _raises(lambda: _s().circle((0.0, 0.0), (0.0, 0.0), 0.0))


def _check_bad_rect():
    return _raises(lambda: _s().rect((0.0, 0.0), (1.0, 1.0, 0.0, 2.0)))


def _check_det():
    return _s().circle((1.0, 1.0), (0.0, 0.0), 1.0) == _s().circle((1.0, 1.0), (0.0, 0.0), 1.0)


def _check_type():
    return isinstance(_s().circle((0.0, 0.0), (0.0, 0.0), 1.0), float)


def _check_sign():
    return _s().circle((0.5, 0.0), (0.0, 0.0), 1.0) < 0.0


def _check_seg_zero_len():
    return _raises(lambda: _s().segment((0.0, 0.0), (1.0, 1.0), (1.0, 1.0)))

CASES = [("circle_center", "_check_circle_center"), ("circle_edge", "_check_circle_edge"), ("circle_out", "_check_circle_out"), ("rect_inside", "_check_rect_inside"), ("rect_outside", "_check_rect_outside"), ("rect_edge", "_check_rect_edge"), ("seg_on", "_check_seg_on"), ("seg_dist", "_check_seg_dist"), ("seg_end", "_check_seg_end"), ("bad_radius", "_check_bad_radius"), ("bad_rect", "_check_bad_rect"), ("det", "_check_det"), ("type", "_check_type"), ("sign", "_check_sign"), ("seg_zero_len", "_check_seg_zero_len")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
