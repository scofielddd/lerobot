from collections import deque
from dataclasses import dataclass, field

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, TransitionKey

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep

KEY = "wrist_roll.pos"


def wrap_deg(d: float) -> float:
    """把角度差折回 [-180, 180]。"""
    if d > 180:
        d -= 360
    elif d < -180:
        d += 360
    return d


def dnr_gain(s: float, s_min: float, s_max: float, k_max: float) -> float:
    """根据 s 在 [s_min, s_max] 的位置，返回一个 0~k_max 的增益。"""
    if s <= s_min:
        return 0.0
    elif s >= s_max:
        return k_max
    else:
        return k_max * (s - s_min) / (s_max - s_min)


@ProcessorStepRegistry.register("wrist_roll_gain")
@dataclass
class WristRollGainStep(RobotActionProcessorStep):
    mode: str = "direct"  # "direct" | "fixed" | "dnr"
    k_max: float = 1.8
    limit_deg: float = 155.0
    fps: float = 30.0
    s_min: float = 30.0  # deg/s
    s_max: float = 90.0  # deg/s
    window: int = 15  # 帧

    # 运行时状态：不是构造参数，所以 init=False
    _prev_input: float | None = field(default=None, init=False, repr=False)
    _target: float | None = field(default=None, init=False, repr=False)
    _recent: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self):
        self._recent = deque(maxlen=self.window)

    def action(self, action: RobotAction) -> RobotAction:
        roll = action[KEY]
        if self._target is None:
            # 第一帧：起点 = 机械臂当前位置；拿不到观测就用输入
            obs = (
                self._current_transition.get(TransitionKey.OBSERVATION) if self._current_transition else None
            )
            # obs 里有 KEY 就用 obs[KEY]，否则用 roll
            self._target = obs[KEY] if obs and KEY in obs else roll
        else:
            d = wrap_deg(roll - self._prev_input)
            # target 加上 k × d，再限制在 [-limit_deg, +limit_deg]
            self._recent.append(d)
            self._target += self._gain() * d
            self._target = max(-self.limit_deg, min(self.limit_deg, self._target))
        self._prev_input = roll
        action[KEY] = self._target
        return action

    def _gain(self) -> float:
        # direct 返回 1.0，fixed 返回 k_max；其他 mode 就 raise ValueError
        if self.mode == "direct":
            return 1.0
        elif self.mode == "fixed":
            return self.k_max
        elif self.mode == "dnr":
            dt = 1 / self.fps
            s_avg = abs(sum(self._recent)) / (len(self._recent) * dt)
            return dnr_gain(s_avg, self.s_min, self.s_max, self.k_max)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def reset(self) -> None:
        self._prev_input = None
        self._target = None
        self._recent.clear()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features  # 只改值，不增删键，所以原样返回
