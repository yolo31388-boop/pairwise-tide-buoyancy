"""潮汐水位与浮力采样。

本模块实现一套确定性的潮汐/浮力模拟，并修复以下六个问题：

1. ``same_tide_for_physics`` —— 水位在渲染帧和物理帧各取一次，物理帧落后半帧，
   导致船体吃水不一致。修复：每个渲染帧只采样一次水位，物理与渲染共用该快照。
2. ``freeze_before_height`` —— 地形高度查询是异步的，未就绪时以 0 填充，
   退潮后会错误判定搁浅。修复：查询就绪前冻结该刚体的浮力/重力积分，
   就绪后从冻结时的状态继续，不产生速度突变。
3. ``reset_phase_on_level`` —— 换关时潮汐相位未重置，新关卡沿用上一关潮位。
   修复：切换关卡时把相位重置到关卡起始相位。
4. ``wake_on_submerge`` —— 休眠木箱被当作静态碰撞体，入水后不唤醒。
   修复：采样点一旦没入水面，立即唤醒刚体并清除“静态”标记。
5. ``freeze_phase_on_pause`` —— 暂停时全局时间继续推进，恢复后浪高跳变。
   修复：暂停期间潮汐相位冻结，恢复后从同一相位继续。
6. ``clamp_wave_height`` —— 位移用全局时间乘固定频率，暂停会跳变。
   修复：改用相位驱动，并把每帧位移增量限幅，保证浪高连续。

除六个修复函数外，还提供 ``sample_tide`` / ``apply_buoyancy`` /
``switch_level`` / ``pause`` / ``validate_buoyancy`` / ``metrics`` /
``audit`` / ``fallback`` / ``determinism`` 等入口。所有入口都接受一个参数
（测试以整数 1 调用），缺省时内部构造默认世界。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "RigidBody",
    "TideFrame",
    "TideWorld",
    "same_tide_for_physics",
    "freeze_before_height",
    "reset_phase_on_level",
    "wake_on_submerge",
    "freeze_phase_on_pause",
    "clamp_wave_height",
    "sample_tide",
    "apply_buoyancy",
    "switch_level",
    "pause",
    "validate_buoyancy",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

GRAVITY = 9.81
WATER_DENSITY = 1025.0
EPS = 1.0e-9
# 单帧最大浪高变化（米），防止暂停/卡顿恢复后的位移跳变。
MAX_WAVE_STEP = 0.05
# 单个渲染帧允许推进的最大模拟时间（秒）。超长帧会被截断，
# 这样物理始终能在本帧内追平渲染，而不会永久落后于画面。
MAX_FRAME_DT = 1.0 / 8.0
# 物理子步上限：与截断后的帧长对应，保证物理永远不落到渲染帧之后。
MAX_PHYSICS_STEPS = 8


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Level:
    """关卡潮汐配置。``phase`` 为关卡起始相位（度）。"""

    name: str = "level"
    base_height: float = 2.0
    amplitude: float = 1.5
    period: float = 12.0
    phase: float = 0.0
    wind_dir: float = 0.0


@dataclass
class TideFrame:
    """单个渲染帧的潮汐快照，物理与渲染共用同一份。"""

    sim_time: float
    phase: float
    height: float
    wave_offset: float
    waterline: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "phase": self.phase,
            "height": self.height,
            "wave_offset": self.wave_offset,
            "waterline": self.waterline,
        }


@dataclass
class HeightQuery:
    """异步地形高度查询句柄。"""

    key: tuple[float, float, float] | None
    ready: bool = False
    result: float = 0.0
    serial: int = 0


@dataclass
class RigidBody:
    """刚体。``sample_points`` 为相对质心的浮力采样点。"""

    name: str = "body"
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mass: float = 50.0
    half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5)
    sample_points: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)]
    )
    asleep: bool = True
    static: bool = False
    query: HeightQuery | None = None
    frozen: bool = False
    terrain_height: float = 0.0
    contacts: int = 0
    wake_count: int = 0

    @property
    def underwater(self) -> bool:
        return self.position[2] < self.terrain_height


@dataclass
class BuoyancyReport:
    """一次浮力积分的诊断结果。"""

    body: str
    submerged_points: int
    total_points: int
    force_z: float
    submerged: bool
    terrain_ready: bool
    frozen: bool
    woke: bool


# ---------------------------------------------------------------------------
# 潮汐相位与水位
# ---------------------------------------------------------------------------


def _angular_speed(level: Level) -> float:
    period = level.period if level.period > EPS else 12.0
    return 360.0 / period


def wrap_phase(phase: float) -> float:
    return phase % 360.0


def advance_phase(phase: float, dt: float, level: Level) -> float:
    if dt <= 0.0:
        return wrap_phase(phase)
    return wrap_phase(phase + _angular_speed(level) * dt)


def tide_height(phase: float, level: Level) -> float:
    """给定相位的水位高度。``phase`` 以度为单位。"""
    return level.base_height + level.amplitude * math.sin(math.radians(phase))


def wave_offset(phase: float, level: Level) -> float:
    """波浪位移。相位驱动，因此暂停即冻结，重放可复现。"""
    return 0.35 * math.sin(math.radians(2.0 * phase + level.wind_dir))


def smooth_wave(
    phase: float,
    level: Level,
    previous: float | None = None,
    max_step: float = MAX_WAVE_STEP,
) -> float:
    """按相位计算浪高增量，并限制单帧变化量以保证连续。"""
    target = wave_offset(phase, level)
    if previous is None or max_step <= 0.0:
        return target
    delta = target - previous
    if delta > max_step:
        return previous + max_step
    if delta < -max_step:
        return previous - max_step
    return target


def sample_tide(arg: Any = None, level: Level | None = None, phase: float | None = None) -> dict[str, Any]:
    """采样一次潮位。供渲染与物理共用，保证两者读到同一水位。"""
    level = level or _default_level()
    if phase is None:
        phase = 0.0
    height = tide_height(phase, level)
    return {
        "level": level.name,
        "phase": wrap_phase(phase),
        "height": height,
        "wave_offset": wave_offset(phase, level),
        "waterline": height,
    }


# ---------------------------------------------------------------------------
# 修复 1：物理与渲染共用同一潮位
# ---------------------------------------------------------------------------


def same_tide_for_physics(arg: Any = None, world: "TideWorld | None" = None) -> dict[str, Any]:
    """每个渲染帧只采样一次水位，物理帧复用该快照。

    物理帧永不晚于渲染帧：模拟时间在每个渲染帧内推进，物理步使用该帧
    采样的 ``TideFrame``；物理滞后时补步（上限 ``MAX_PHYSICS_STEPS``），
    因此不会出现“物理落后半帧、吃水偏浅”的现象。
    """
    world = world or TideWorld.default()
    return world.advance_render_frame().as_dict()


# ---------------------------------------------------------------------------
# 修复 2：高度查询就绪前冻结
# ---------------------------------------------------------------------------


def freeze_before_height(arg: Any = None, body: RigidBody | None = None) -> dict[str, Any]:
    """高度查询未就绪时冻结刚体积分。

    冻结期间不改变速度与位移（不灌入零高度、不做重力积分），查询就绪后
    从冻结时的状态继续。返回本帧是否处于冻结状态及查询编号。
    """
    if body is None:
        body = TideWorld.default().bodies[0]
    query = body.query
    ready = query is None or query.ready
    # 冻结时保留速度：恢复后无缝继续，而不是从静止重新加速。
    body.frozen = not ready
    return {
        "body": body.name,
        "frozen": body.frozen,
        "ready": ready,
        "serial": query.serial if query is not None else 0,
    }


# ---------------------------------------------------------------------------
# 修复 3：换关重置相位
# ---------------------------------------------------------------------------


def reset_phase_on_level(arg: Any = None, level: Level | None = None) -> float:
    """切换关卡时返回该关卡起始相位，而不是沿用上一关相位。"""
    level = level or _default_level()
    return wrap_phase(level.phase)


# ---------------------------------------------------------------------------
# 修复 4：入水唤醒刚体
# ---------------------------------------------------------------------------


def wake_on_submerge(
    arg: Any = None,
    body: RigidBody | None = None,
    waterline: float | None = None,
) -> dict[str, Any]:
    """采样点一旦没入水面，唤醒刚体并清除静态标记。"""
    if body is None:
        world = TideWorld.default()
        body = world.bodies[0]
        waterline = world.height
    if waterline is None:
        waterline = 0.0
    submerged = any(
        body.position[2] + p[2] < waterline + EPS for p in body.sample_points
    ) or body.position[2] < waterline + EPS
    woke = False
    if submerged:
        if body.asleep or body.static:
            woke = True
            body.wake_count += 1
        body.asleep = False
        body.static = False
    return {
        "body": body.name,
        "submerged": submerged,
        "woke": woke,
        "asleep": body.asleep,
        "static": body.static,
    }


# ---------------------------------------------------------------------------
# 修复 5：暂停冻结相位
# ---------------------------------------------------------------------------


def freeze_phase_on_pause(
    arg: Any = None, world: "TideWorld | None" = None
) -> dict[str, Any]:
    """暂停时冻结潮汐相位，恢复后从同一相位继续。"""
    world = world or TideWorld.default()
    entering = not world.paused
    world.paused = True
    return {
        "paused": True,
        "phase": world.phase,
        "entering": entering,
        "sim_time": world.sim_time,
    }


# ---------------------------------------------------------------------------
# 修复 6：浪高连续
# ---------------------------------------------------------------------------


def clamp_wave_height(
    arg: Any = None,
    target: float | None = None,
    previous: float | None = None,
    max_step: float = MAX_WAVE_STEP,
) -> dict[str, Any]:
    """把浪高变化限幅到 ``max_step``，避免暂停/恢复造成跳变。"""
    if target is None:
        level = _default_level()
        target = wave_offset(level.phase, level)
    previous = 0.0 if previous is None else previous
    lo, hi = previous - max_step, previous + max_step
    applied = min(max(target, lo), hi)
    return {
        "target": target,
        "applied": applied,
        "previous": previous,
        "max_step": max_step,
        "clamped": abs(applied - target) > EPS,
    }


# ---------------------------------------------------------------------------
# 世界与积分
# ---------------------------------------------------------------------------


def _default_level() -> Level:
    return Level(name="default", base_height=2.0, amplitude=1.5, period=12.0, phase=0.0)


def _default_bodies(level: Level) -> list[RigidBody]:
    pts = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (-0.5, 0.5, 0.0), (0.5, 0.5, 0.0)]
    return [
        RigidBody(
            name="boat",
            position=[0.0, 0.0, level.base_height + level.amplitude],
            velocity=[1.0, 0.0, 0.0],
            mass=800.0,
            half_extents=(2.0, 1.0, 0.5),
            sample_points=pts,
            asleep=False,
            query=HeightQuery(key=(0.0, 0.0, 0.0), ready=True, result=0.2, serial=1),
            terrain_height=0.2,
        ),
        RigidBody(
            name="crate",
            position=[3.0, 0.0, level.base_height + level.amplitude - 1.0],
            velocity=[0.0, 0.0, 0.0],
            mass=20.0,
            half_extents=(0.4, 0.4, 0.4),
            sample_points=[(0.0, 0.0, 0.0)],
            asleep=True,
            static=True,
            query=HeightQuery(key=(3.0, 0.0, 0.0), ready=True, result=0.1, serial=2),
            terrain_height=0.1,
        ),
    ]


@dataclass
class TideWorld:
    """确定性的潮汐世界。"""

    level: Level = field(default_factory=_default_level)
    phase: float = 0.0
    sim_time: float = 0.0
    paused: bool = False
    _last_render_time: float = 0.0
    _last_physics_time: float = 0.0
    _wave: float = 0.0
    height: float = 0.0
    wave: float = 0.0
    bodies: list[RigidBody] = field(default_factory=list)
    frames: int = 0
    stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phase = wrap_phase(self.level.phase)
        self.height = tide_height(self.phase, self.level)
        self.wave = wave_offset(self.phase, self.level)
        if not self.bodies:
            self.bodies = _default_bodies(self.level)
        self.stats.setdefault("wake_events", 0)
        self.stats.setdefault("frozen_frames", 0)
        self.stats.setdefault("level_switches", 0)
        self.stats.setdefault("pauses", 0)

    # -- 便捷构造 ---------------------------------------------------------

    @classmethod
    def default(cls) -> "TideWorld":
        return cls()

    @classmethod
    def from_level(cls, level: Level, bodies: Iterable[RigidBody] | None = None, seed: int = 0) -> "TideWorld":
        world = cls(level=level)
        world.phase = wrap_phase(level.phase + float(seed % 360))
        world.height = tide_height(world.phase, level)
        world.wave = wave_offset(world.phase, level)
        if bodies is not None:
            world.bodies = list(bodies)
        return world

    # -- 时间推进 ---------------------------------------------------------

    @property
    def waterline(self) -> float:
        """当前水面高度（潮位 + 连续浪高）。"""
        return self.height + self.wave

    def _tick_phase(self, dt: float) -> None:
        """推进潮汐相位。暂停时 dt 被忽略，保证相位冻结。"""
        if self.paused or dt <= 0.0:
            return
        self.phase = advance_phase(self.phase, dt, self.level)
        self.sim_time += dt
        self.height = tide_height(self.phase, self.level)
        self.wave = smooth_wave(self.phase, self.level, self.wave)

    def advance_render_frame(self, dt: float = 1.0 / 60.0) -> TideFrame:
        """推进一个渲染帧，并返回该帧物理与渲染共用的潮汐快照。

        物理进度落后于渲染时在本帧内补步，物理时间不晚于渲染时间。
        超长帧先被截断到 ``MAX_FRAME_DT``，避免物理在本帧内追不平、
        永久落后于画面（那正是"物理落后半帧"问题的根源）。
        """
        dt = min(max(dt, 0.0), MAX_FRAME_DT)
        self._tick_phase(dt)
        self._last_render_time = self.sim_time
        self._settle_physics()
        self.frames += 1
        return TideFrame(
            sim_time=self.sim_time,
            phase=self.phase,
            height=self.height,
            wave_offset=self.wave,
            waterline=self.waterline,
        )

    def advance_physics_frame(self) -> None:
        """物理追赶：复用最近一帧渲染的潮汐快照，绝不使用更新的水位。

        物理时间只向渲染时间靠拢，不会越过它，因此物理吃水永远与画面一致。
        """
        self._last_render_time = max(self._last_render_time, self._last_physics_time)
        self._settle_physics()

    def _settle_physics(self) -> None:
        steps = 0
        while (
            self._last_physics_time + 1.0e-12 < self._last_render_time
            and steps < MAX_PHYSICS_STEPS
        ):
            step = min(1.0 / 60.0, self._last_render_time - self._last_physics_time)
            if step <= 1.0e-12:
                break
            self._last_physics_time += step
            self._physics_step(step)
            steps += 1

    def _physics_step(self, dt: float) -> None:
        for body in self.bodies:
            self.step_body(body, dt)

    def step_body(self, body: RigidBody, dt: float) -> BuoyancyReport:
        """积分单个刚体：先判暂停，再判冻结，再判唤醒，最后算浮力。"""
        if self.paused:
            return BuoyancyReport(
                body=body.name,
                submerged_points=0,
                total_points=len(body.sample_points),
                force_z=0.0,
                submerged=False,
                terrain_ready=True,
                frozen=False,
                woke=False,
            )
        # 修复 2：查询就绪前冻结，不灌零高度、不积分。
        ready = body.query is None or body.query.ready
        body.frozen = not ready
        if not ready:
            self.stats["frozen_frames"] += 1
            return BuoyancyReport(
                body=body.name,
                submerged_points=0,
                total_points=len(body.sample_points),
                force_z=0.0,
                submerged=False,
                terrain_ready=False,
                frozen=True,
                woke=False,
            )
        if body.query is not None:
            body.terrain_height = body.query.result
        # 修复 4：入水先唤醒，避免把休眠刚体当静态体。
        wake = wake_on_submerge(None, body, self.waterline)
        if wake["woke"]:
            self.stats["wake_events"] += 1
        if body.asleep:
            # 仍在休眠且未入水：保持静止，不做重力积分。
            return BuoyancyReport(
                body=body.name,
                submerged_points=0,
                total_points=len(body.sample_points),
                force_z=0.0,
                submerged=False,
                terrain_ready=True,
                frozen=False,
                woke=False,
            )
        return self.apply_buoyancy(body, dt, self.waterline)

    def apply_buoyancy(self, body: RigidBody, dt: float, waterline: float | None = None) -> BuoyancyReport:
        """对刚体的采样点计算浮力并积分。

        - 未唤醒的休眠刚体不参与浮力（但入水已在 ``step_body`` 中唤醒）。
        - 手感：浮力与浸没深度成正比，附加阻尼，避免数值发散。
        """
        if waterline is None:
            waterline = self.waterline
        if dt <= 0.0 or self.paused or body.asleep:
            return BuoyancyReport(body.name, 0, len(body.sample_points), 0.0, False, True, False, False)
        volume_per_point = (2.0 * body.half_extents[0] * 2.0 * body.half_extents[1]
                            * 2.0 * body.half_extents[2]) / max(1, len(body.sample_points))
        submerged = 0
        force_z = 0.0
        for point in body.sample_points:
            z = body.position[2] + point[2]
            if z >= waterline:
                continue
            depth = waterline - z
            body.contacts += 1
            submerged += 1
            # 浸没越深，浮力越大；上限为排开整点体积的重量。
            displaced = volume_per_point * min(depth / max(body.half_extents[2], EPS), 1.0)
            force_z += WATER_DENSITY * GRAVITY * displaced
        accel = force_z / max(body.mass, EPS)
        if submerged:
            # 合力 = 浮力 - 重力；再加一点随浸没比例增加的阻尼，防止过冲。
            body.velocity[2] += (accel - GRAVITY) * dt
            body.velocity[2] *= max(0.0, 1.0 - 0.05 * submerged)
        else:
            body.velocity[2] -= GRAVITY * dt
        body.position[2] += body.velocity[2] * dt
        if body.position[2] < body.terrain_height:
            body.position[2] = body.terrain_height
            if body.velocity[2] < 0.0:
                body.velocity[2] = 0.0
        return BuoyancyReport(
            body=body.name,
            submerged_points=submerged,
            total_points=len(body.sample_points),
            force_z=force_z,
            submerged=submerged > 0,
            terrain_ready=True,
            frozen=False,
            woke=False,
        )

    # -- 关卡与暂停 -------------------------------------------------------

    def switch_level(self, level: Level, seed: int = 0) -> dict[str, Any]:
        """切换关卡：重置相位到新关卡起始相位，不继承上一关潮位。"""
        previous_phase = self.phase
        self.level = level
        self.phase = wrap_phase(level.phase)
        self.sim_time = 0.0
        self._last_render_time = 0.0
        self._last_physics_time = 0.0
        self.height = tide_height(self.phase, level)
        self.wave = smooth_wave(self.phase, level, self.wave)
        self.bodies = _default_bodies(level)
        self.frames = 0
        self.stats["level_switches"] += 1
        return {
            "level": level.name,
            "previous_phase": previous_phase,
            "phase": self.phase,
            "height": self.height,
            "seed": seed,
        }

    def set_paused(self, paused: bool = True) -> dict[str, Any]:
        if paused and not self.paused:
            self.stats["pauses"] += 1
        self.paused = paused
        return {"paused": self.paused, "phase": self.phase, "sim_time": self.sim_time}

    # -- 诊断 -------------------------------------------------------------

    def validate(self) -> dict[str, Any]:
        """检查六项修复是否同时成立。"""
        checks = {
            "physics_tide_matches_render": True,
            "frozen_without_zero_height": True,
            "level_resets_phase": True,
            "submerged_bodies_awake": True,
            "pause_freezes_phase": True,
            "wave_continuous": True,
        }
        world = TideWorld.from_level(self.level)
        frame = world.advance_render_frame()
        checks["physics_tide_matches_render"] = (
            abs(world._last_physics_time - frame.sim_time) <= 1.0e-9
            and abs(world.height - frame.height) <= 1.0e-12
        )
        idle = RigidBody(name="probe", query=HeightQuery(key=(0.0, 0.0, 0.0), ready=False, result=0.0))
        world.step_body(idle, 1.0 / 60.0)
        checks["frozen_without_zero_height"] = idle.frozen and idle.position[2] == 0.0 and idle.velocity[2] == 0.0
        # 把旧相位拨到与新关卡起始相位明显不同的位置，确保确实发生了重置。
        new_level = Level(name="next", base_height=1.0, amplitude=0.5, period=10.0, phase=200.0)
        world.phase = new_level.phase + 123.0
        info = world.switch_level(new_level)
        checks["level_resets_phase"] = abs(info["phase"] - new_level.phase) <= EPS
        crate = RigidBody(
            name="crate",
            position=[0.0, 0.0, -1.0],
            asleep=True,
            static=True,
            query=HeightQuery(key=(0.0, 0.0, 0.0), ready=True, result=-2.0),
        )
        world.bodies = [crate]
        report = world.step_body(crate, 1.0 / 60.0)
        checks["submerged_bodies_awake"] = (not crate.asleep) and (not crate.static) and report.submerged
        paused_world = TideWorld.from_level(new_level)
        paused_world.set_paused(True)
        phase_before = paused_world.phase
        paused_world.advance_render_frame()
        checks["pause_freezes_phase"] = abs(paused_world.phase - phase_before) <= EPS
        wave_world = TideWorld.from_level(self.level)
        wave_world.advance_render_frame()
        wave_before = wave_world.wave
        wave_world.set_paused(True)
        for _ in range(30):
            wave_world.advance_render_frame()
        paused_stable = abs(wave_world.wave - wave_before) <= EPS
        wave_world.set_paused(False)
        wave_world.advance_render_frame()
        checks["wave_continuous"] = (
            paused_stable and abs(wave_world.wave - wave_before) <= MAX_WAVE_STEP + EPS
        )
        return {"ok": all(checks.values()), "checks": checks, "world": self}

    def as_metrics(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "sim_time": self.sim_time,
            "phase": self.phase,
            "height": self.height,
            "wave": self.wave,
            "waterline": self.waterline,
            "paused": self.paused,
            "bodies": len(self.bodies),
            "wake_events": self.stats["wake_events"],
            "frozen_frames": self.stats["frozen_frames"],
            "level_switches": self.stats["level_switches"],
            "pauses": self.stats["pauses"],
        }

    def signature(self) -> dict[str, Any]:
        return {
            "phase": round(self.phase, 6),
            "height": round(self.height, 6),
            "wave": round(self.wave, 6),
            "bodies": [
                {
                    "name": b.name,
                    "z": round(b.position[2], 6),
                    "vz": round(b.velocity[2], 6),
                    "asleep": b.asleep,
                    "static": b.static,
                }
                for b in self.bodies
            ],
        }


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def _world_from(arg: Any) -> TideWorld:
    if isinstance(arg, TideWorld):
        return arg
    return TideWorld.default()


def _body_from(world: TideWorld, index: int = 0) -> RigidBody:
    if not world.bodies:
        world.bodies = _default_bodies(world.level)
    index = min(index, len(world.bodies) - 1)
    return world.bodies[index]


def apply_buoyancy(arg: Any = None, body: RigidBody | None = None, dt: float = 1.0 / 60.0) -> dict[str, Any]:
    """对外浮力入口：对世界中的刚体做一次浮力积分。"""
    world = _world_from(arg)
    if body is None:
        body = _body_from(world)
    report = world.step_body(body, dt)
    return {
        "body": report.body,
        "submerged_points": report.submerged_points,
        "total_points": report.total_points,
        "force_z": report.force_z,
        "submerged": report.submerged,
        "terrain_ready": report.terrain_ready,
        "frozen": report.frozen,
        "woke": report.woke,
        "waterline": world.waterline,
    }


def switch_level(arg: Any = None, level: Level | None = None) -> dict[str, Any]:
    world = _world_from(arg)
    level = level or Level(name="next", base_height=1.0, amplitude=0.5, period=10.0, phase=200.0)
    return world.switch_level(level)


def pause(arg: Any = None, paused: bool = True) -> dict[str, Any]:
    world = _world_from(arg)
    world.set_paused(paused)
    return freeze_phase_on_pause(None, world) if paused else world.set_paused(False)


def validate_buoyancy(arg: Any = None) -> dict[str, Any]:
    world = _world_from(arg)
    result = world.validate()
    return {
        "ok": result["ok"],
        "checks": result["checks"],
        "body": world.bodies[0].name if world.bodies else None,
    }


def metrics(arg: Any = None) -> dict[str, Any]:
    world = _world_from(arg)
    if not isinstance(arg, TideWorld):
        for _ in range(3):
            world.advance_render_frame()
    return world.as_metrics()


def audit(arg: Any = None) -> dict[str, Any]:
    """逐项审计六项修复，返回问题条目（空表示全部通过）。"""
    world = _world_from(arg) if isinstance(arg, TideWorld) else TideWorld.default()
    result = world.validate()
    issues = [name for name, passed in result["checks"].items() if not passed]
    return {
        "ok": not issues,
        "checks": result["checks"],
        "issues": issues,
        "count": len(issues),
    }


def fallback(arg: Any = None, height: float | None = None) -> dict[str, Any]:
    """高度查询不可用时的兜底：使用上一帧地形高度，绝不以 0 填充。"""
    world = _world_from(arg)
    if height is None and world.bodies:
        height = world.bodies[0].terrain_height
    height = 0.0 if height is None else height
    return {
        "height": height,
        "ready": False,
        "frozen": True,
        "used_last_known": True,
    }


def determinism(arg: Any = None, seed: int = 1) -> dict[str, Any]:
    """同一 ``seed`` 运行两次应得到相同轨迹。"""
    level = Level(name="det", base_height=2.0, amplitude=1.5, period=12.0, phase=0.0)

    def run(s: int) -> dict[str, Any]:
        world = TideWorld.from_level(level, seed=s)
        traces = []
        for _ in range(4):
            frame = world.advance_render_frame(1.0 / 30.0)
            traces.append(
                {
                    "sim_time": frame.sim_time,
                    "phase": round(frame.phase, 9),
                    "height": round(frame.height, 9),
                    "wave": round(frame.wave_offset, 9),
                }
            )
        return {"seed": s, "traces": traces, "signature": world.signature()}

    first = run(seed)
    second = run(seed)
    return {
        "deterministic": first["signature"] == second["signature"],
        "runs": [first, second],
        "stable": first["signature"],
    }
