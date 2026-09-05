"""The shared converter's camera matrix (R-EXP-08).

`norm10` replaces each fork's own converter with this one, so any difference
in what it writes is charged to the neighbour count in the author->norm10
delta. The principal point is where that went wrong once already.
"""

import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "src" / "eval"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

import colmap2mvsnet_acm_perf as conv  # noqa: E402


def _camera(model, width, height, params):
    return conv.Camera(id=0, model=model, width=width, height=height, params=params)


def test_the_principal_point_is_absolute_not_offset_by_half_the_image():
    """pipes at width 1600: `PINHOLE 1600 1065 882.38 881.94 802.37 529.22`.
    Adding width/2 put the principal point at 1602 px in a 1600 px image and
    took ACMH's F1 from 0.531 to 0.0002."""
    matrix = conv.camera_intrinsic(
        _camera("PINHOLE", 1600, 1065, [882.3845659163986, 881.9439628109153,
                                        802.3665594855305, 529.2209007486115])
    )
    assert matrix[0, 2] == pytest.approx(802.3665594855305)
    assert matrix[1, 2] == pytest.approx(529.2209007486115)
    assert matrix[0, 0] == pytest.approx(882.3845659163986)
    assert matrix[1, 1] == pytest.approx(881.9439628109153)


def test_a_single_focal_model_fills_both_axes():
    matrix = conv.camera_intrinsic(_camera("SIMPLE_PINHOLE", 100, 80, [50.0, 49.0, 39.0]))
    assert (matrix[0, 0], matrix[1, 1]) == (50.0, 50.0)
    assert (matrix[0, 2], matrix[1, 2]) == (49.0, 39.0)


def test_it_matches_what_the_shipped_acm_converter_writes():
    """The value ACMH's own `colmap2mvsnet_acm.py` writes for the same camera
    is `[[fx,0,cx],[0,fy,cy],[0,0,1]]` -- nothing added."""
    params = [3430.27, 3429.23, 3119.2, 2057.75]
    matrix = conv.camera_intrinsic(_camera("PINHOLE", 6220, 4141, params))
    assert matrix.tolist() == [
        [3430.27, 0, 3119.2],
        [0, 3429.23, 2057.75],
        [0, 0, 1],
    ]
