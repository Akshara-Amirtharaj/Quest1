from fractions import Fraction


def test_pts_timestamp_conversion_rule() -> None:
    pts = 9009
    time_base = Fraction(1, 90000)
    assert float(pts * time_base) == 0.1001
