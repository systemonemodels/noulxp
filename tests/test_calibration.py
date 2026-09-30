import pytest

from noulxp.calibration import Calibration
from noulxp.errors import PackageError
from noulxp.export import decider as decider_export
from noulxp.export import laya as laya_export

TYPED_DECISIONS = {
    "temperature": [1.0148, 1.0374, 1.0575],
    "temperature_by_options": {
        "choice:3-5": 1.76,
        "choice:6-10": 1.00002,
        "score:3-5": 1.2514,
        "noul:2": 1.9834,
        "choice:11+": 0.1006,
        "choice:2": 1.9064,
    },
}


def test_lookup_order():
    cal = Calibration(
        {
            "temperature": {"choice": 2.0, "noul": 3.0},
            "by_option_count": [
                {"type": "choice", "min": 3, "max": 5, "temperature": 1.5},
                {"type": "choice", "min": 11, "max": None, "temperature": 0.7},
            ],
        }
    )
    assert cal.temperature("choice", 2) == 2.0
    assert cal.temperature("choice", 4) == 1.5
    assert cal.temperature("choice", 30) == 0.7
    assert cal.temperature("noul", 2) == 3.0
    assert cal.temperature("score", 5) == 1.0


@pytest.mark.parametrize(
    "data",
    [
        {"temperature": {"choice": 0}},
        {"temperature": {"choice": "hot"}},
        {"temperature": {"maybe": 1.0}},
        {"by_option_count": [{"type": "choice", "min": 5, "max": 2, "temperature": 1}]},
        {"by_option_count": [{"type": "choice", "min": 2, "temperature": -1}]},
    ],
)
def test_invalid_calibration(data):
    with pytest.raises(PackageError):
        Calibration(data)


def test_laya_calibration_clamps_and_buckets():
    data = laya_export.calibration(TYPED_DECISIONS)
    cal = Calibration(data)
    assert cal.temperature("choice", 2) == pytest.approx(1.9064)
    assert cal.temperature("choice", 4) == pytest.approx(1.76)
    assert cal.temperature("choice", 12) == 0.5  # 0.1006 clamped to laya's floor
    assert cal.temperature("score", 7) == pytest.approx(1.0374)  # no bucket: per type
    assert cal.temperature("noul", 2) == pytest.approx(1.9834)
    assert data["source"]["clamped"] == ["choice:11+: 0.1006 -> 0.5"]
    assert laya_export.clamp(99) == 5.0 and laya_export.clamp(float("nan")) == 1.0


def test_decider_calibration_by_type_with_fallback():
    cfg = {"temperature": 1.145, "temperature_by_type": {"choice": 1.164, "noul": 1.624}}
    cal = Calibration(decider_export.calibration(cfg))
    assert cal.temperature("choice", 3) == 1.164
    assert cal.temperature("score", 5) == 1.145
