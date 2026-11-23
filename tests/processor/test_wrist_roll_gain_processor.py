import pytest

from lerobot.processor.wrist_roll_gain_processor import WristRollGainStep, wrap_deg


def feed(step, rolls):
    """依次喂入一串手腕角度，返回每一帧输出的 wrist_roll 目标。"""
    return [step.action({"wrist_roll.pos": r})["wrist_roll.pos"] for r in rolls]


def test_wrap_deg():
    assert wrap_deg(150) == 150  # 不跨接缝，原样返回
    assert wrap_deg(358) == -2  # 例如 -179 → +179，其实是往负方向走了 2°
    assert wrap_deg(-358) == 2


def test_direct_mapping_follows_input():
    step = WristRollGainStep(mode="direct")
    assert feed(step, [0, 10, 20, 15]) == pytest.approx([0, 10, 20, 15])


def test_fixed_gain_doubles_increments():
    step = WristRollGainStep(mode="fixed", k_max=2.0)
    assert feed(step, [0, 10, 20, 15]) == pytest.approx([0, 20, 40, 30])


def test_target_is_clamped_to_range():
    step = WristRollGainStep(mode="fixed", k_max=2.0, limit_deg=155.0)
    assert feed(step, [0, 50, 100]) == pytest.approx([0, 100, 155])
