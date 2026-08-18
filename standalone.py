
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import robotic as ry


# ---------------------------------------------------------------------------
# Defaults matching main.py / keyframes.py
# ---------------------------------------------------------------------------

DEFAULT_JSON = "JSON/own_examples/260804_FoC_demo.json"
DEFAULT_MODEL_ROOT = "src/models"
DEFAULT_RADIUS = 0.005
DEFAULT_SCALE = 0.0011
DEFAULT_ATTEMPTS = 100
DEFAULT_RANDOM_MULTIPLIER = 3.0
DEFAULT_RANDOM_OFFSET = -1.5

ALL_CONSTRAINT_GROUPS = (
    "joint_limits",
    "collisions",
    "continuing_supports",
    "releasable_supports",
    "candidate_hold_during_support",
    "main_grasp_positions",
    "main_grasp_orientations",
    "main_stable_switch",
    "old_support_away",
    "new_support_grasp",
    "new_support_stable_switch",
    "new_support_rod_pose",
    "pickup",
)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timings:
    def __init__(self):
        self.rows: list[tuple[str, float]] = []

    @contextmanager
    def measure(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.rows.append((label, duration))
            print(f"[time] {label:<42} {duration:>10.6f} s")

    def total_matching(self, prefix: str) -> float:
        return sum(value for label, value in self.rows if label.startswith(prefix))

    def print_summary(self):
        if not self.rows:
            return

        print("\nTiming summary")
        print("--------------")
        for label, duration in self.rows:
            print(f"{label:<46} {duration:>10.6f} s")


# ---------------------------------------------------------------------------
# Truss loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrussData:
    nodes: dict[int, tuple[float, float, float]]
    elements: dict[int, tuple[int, int]]
    grounded_rods: frozenset[int]
    couplers: frozenset[tuple[int, int]]

    @classmethod
    def from_json(cls, path: Path) -> "TrussData":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        nodes = {
            int(node["node_id"]): (
                float(node["point"]["X"]),
                float(node["point"]["Y"]),
                float(node["point"]["Z"]),
            )
            for node in data["node_list"]
        }

        elements: dict[int, tuple[int, int]] = {}
        grounded: set[int] = set()

        for rod in data["rod_list"]:
            rod_id = int(rod["rod_id"])
            elements[rod_id] = tuple(int(value) for value in rod["end_node_ids"])
            if rod.get("grounded", 0) == 1:
                grounded.add(rod_id)

        couplers = {
            tuple(sorted(int(value) for value in coupler["rod_ids"]))
            for coupler in data.get("coupler_list", [])
        }

        return cls(
            nodes=nodes,
            elements=elements,
            grounded_rods=frozenset(grounded),
            couplers=frozenset(couplers),
        )


# ---------------------------------------------------------------------------
# Geometry helpers copied from utils.py / rods.py
# ---------------------------------------------------------------------------


def quaternion_from_z_to_vector(direction: Iterable[float]) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    dot = np.dot(z_axis, direction)

    if axis_norm < 1e-8:
        if dot > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([0.0, 1.0, 0.0, 0.0])

    axis = axis / axis_norm
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return np.concatenate(
        (
            [np.cos(angle / 2.0)],
            axis * np.sin(angle / 2.0),
        )
    )


class RodManager:
    def __init__(
        self,
        config: ry.Config,
        truss: TrussData,
        radius: float,
        scale: float,
    ):
        self.C = config
        self.truss = truss
        self.radius = radius
        self.scale = scale

    def get_rod_endpoints(self, rod_id: int) -> tuple[np.ndarray, np.ndarray]:
        node_1, node_2 = self.truss.elements[rod_id]
        point_1 = np.asarray(self.truss.nodes[node_1], dtype=float) * self.scale
        point_2 = np.asarray(self.truss.nodes[node_2], dtype=float) * self.scale
        return point_1, point_2

    def get_rod_length(self, rod_id: int) -> float:
        point_1, point_2 = self.get_rod_endpoints(rod_id)
        return float(np.linalg.norm(point_2 - point_1) - 0.03)

    def get_goal_pose(self, rod_id: int) -> tuple[np.ndarray, np.ndarray]:
        point_1, point_2 = self.get_rod_endpoints(rod_id)
        center = 0.5 * (point_1 + point_2) + np.array([0.0, 0.0, 0.1])
        quaternion = quaternion_from_z_to_vector(point_2 - point_1)
        return center, quaternion

    def create_rod_at_goal_pose(self, rod_id: int):
        center, quaternion = self.get_goal_pose(rod_id)
        length = self.get_rod_length(rod_id)

        if length < 1e-10:
            raise ValueError(f"Rod {rod_id} has zero length")

        quaternion = quaternion / np.linalg.norm(quaternion)
        self.C.addFrame(f"rod_{rod_id}") \
            .setShape(ry.ST.cylinder, [length, self.radius]) \
            .setColor([0.5, 1.0, 0.0]) \
            .setPosition(center) \
            .setQuaternion(quaternion) \
            .setContact(1)

    def create_dual_arm_grasp_frames(
        self,
        rod_id: int,
        d1_from_end: float = 0.12,
        d12_between_arms: float = 0.8,
    ) -> tuple[str, str]:
        rod_name = f"rod_{rod_id}"
        length = self.get_rod_length(rod_id)
        d2_from_end = d1_from_end + d12_between_arms

        if d1_from_end < 0.0 or d2_from_end > length:
            raise ValueError(
                "Invalid grasp distances: "
                f"d1={d1_from_end}, d2={d2_from_end}, rod length={length}"
            )

        z_1 = -0.5 * length + d1_from_end
        z_2 = -0.5 * length + d2_from_end
        grasp_1 = f"rod_{rod_id}_grasp_a1"
        grasp_2 = f"rod_{rod_id}_grasp_a2"

        self.C.addFrame(grasp_1, rod_name).setRelativePosition([0.0, 0.0, z_1])
        self.C.addFrame(grasp_2, rod_name).setRelativePosition([0.0, 0.0, z_2])
        return grasp_1, grasp_2

    def create_support_grasp_frame_at_fraction(
        self,
        rod_id: int,
        fraction: float,
    ) -> str:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Support fraction must be between 0 and 1")

        rod_name = f"rod_{rod_id}"
        length = self.get_rod_length(rod_id)
        z_position = -0.5 * length + fraction * length
        frame_name = f"rod_{rod_id}_support_grasp_{fraction:.2f}"
        self.C.addFrame(frame_name, rod_name) \
            .setRelativePosition([0.0, 0.0, z_position])
        return frame_name


# ---------------------------------------------------------------------------
# Scene construction copied from scene.py
# ---------------------------------------------------------------------------


class SceneBuilder:
    def __init__(self, config: ry.Config, model_root: Path):
        self.C = config
        self.model_root = model_root

    def ensure_table(self):
        if self.C.getFrame("table") is not None:
            return

        self.C.addFrame("table") \
            .setPosition([0.0, 0.0, 0.0]) \
            .setShape(ry.ST.box, size=[20.0, 20.0, 0.02, 0.005]) \
            .setColor([0.9, 0.9, 0.9]) \
            .setContact(1)

    def check_model_files(self):
        required = (
            self.model_root / "husky" / "husky.g",
            self.model_root / "ur5" / "ur5.g",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Could not find the RAI model files:\n  " + "\n  ".join(missing)
            )

    def import_main_husky(self):
        self.ensure_table()
        husky_path = str(self.model_root / "husky" / "husky.g")
        ur5_path = str(self.model_root / "ur5" / "ur5.g")

        self.C.addFrame("husky_base_XYPhi_joint") \
            .setParent(self.C.getFrame("world")) \
            .setJoint(
                ry.JT.transXYPhi,
                limits=np.array([-30, 30, -30, 30, -3.14, 3.14]),
            ) \
            .setJointState([-1.0, 0.0, 0.0])

        self.C.addFile(husky_path, namePrefix="husky_coll_") \
            .setParent(self.C.getFrame("husky_base_XYPhi_joint")) \
            .setRelativePosition([0.0, 0.0, 0.16])

        self.C.addFile(ur5_path, namePrefix="a1_") \
            .setParent(self.C.getFrame("husky_coll_right_arm_bulkhead_joint")) \
            .setRelativePosition([0.0, 0.0, 0.0]) \
            .setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])

        self.C.addFile(ur5_path, namePrefix="a2_") \
            .setParent(self.C.getFrame("husky_coll_left_arm_bulkhead_joint")) \
            .setRelativePosition([0.0, 0.0, 0.0]) \
            .setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])

    def import_support_husky(
        self,
        name: str,
        base_q: tuple[float, float, float],
    ):
        self.ensure_table()
        husky_path = str(self.model_root / "husky" / "husky.g")
        ur5_path = str(self.model_root / "ur5" / "ur5.g")
        base_frame = f"{name}_base_XYPhi_joint"
        husky_prefix = f"{name}_husky_"

        self.C.addFrame(base_frame) \
            .setParent(self.C.getFrame("world")) \
            .setJoint(
                ry.JT.transXYPhi,
                limits=np.array([-30, 30, -30, 30, -3.14, 3.14]),
            ) \
            .setJointState(list(base_q))

        self.C.addFile(husky_path, namePrefix=husky_prefix) \
            .setParent(self.C.getFrame(base_frame)) \
            .setRelativePosition([0.0, 0.0, 0.16])

        self.C.addFile(ur5_path, namePrefix=f"{name}_a1_") \
            .setParent(self.C.getFrame(f"{husky_prefix}right_arm_bulkhead_joint")) \
            .setRelativePosition([0.0, 0.0, 0.0]) \
            .setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])

    def import_all_robots(self):
        self.check_model_files()
        self.import_main_husky()
        self.import_support_husky("h1", (3.0, -3.0, 0.0))
        self.import_support_husky("h2", (-3.0, -3.0, 0.0))


# ---------------------------------------------------------------------------
# Complete KOMO construction
# ---------------------------------------------------------------------------


class PhaseSchedule:
    def __init__(self):
        self.names: list[str] = []

    def add(self, name: str) -> float:
        self.names.append(name)
        return float(len(self.names))

    @property
    def n_phases(self) -> int:
        return len(self.names)


@dataclass
class ProblemContext:
    komo: ry.KOMO
    q0: np.ndarray
    phases: PhaseSchedule
    phase_info: dict


class FullRemovalProblem:
    def __init__(
        self,
        config: ry.Config,
        rods: RodManager,
        timings: Timings,
        disabled_groups: set[str],
        enable_control: bool,
    ):
        self.C = config
        self.rods = rods
        self.timings = timings
        self.disabled_groups = disabled_groups
        self.enable_control = enable_control

    def enabled(self, group: str) -> bool:
        return group not in self.disabled_groups

    def copy_frame_pose(self, source_name: str, target_name: str):
        source = self.C.getFrame(source_name)
        target = self.C.getFrame(target_name)
        if source is None or target is None:
            raise RuntimeError(f"Missing pose-copy frame: {source_name} -> {target_name}")
        target.setPosition(source.getPosition())
        target.setQuaternion(source.getQuaternion())

    def make_pose_target(self, source_name: str, target_name: str):
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, "world") \
                .setShape(ry.ST.marker, [0.06]) \
                .setContact(0)
        self.copy_frame_pose(source_name, target_name)

    def add_group(self, name: str, callback: Callable[[], None]):
        if not self.enabled(name):
            print(f"[skip] constraint group: {name}")
            return
        with self.timings.measure(f"add constraints: {name}"):
            callback()

    def build(
        self,
        rod_id: int,
        continuing_supports: dict[str, int],
        releasable_supports: dict[str, int],
        new_support_assignments: dict[str, int],
        old_support_gripper: str | None,
        support_fraction: float,
    ) -> ProblemContext:
        rod = f"rod_{rod_id}"
        q0 = self.C.getJointState().copy()
        candidate_is_supported = bool(releasable_supports)
        old_support_is_reused = (
            old_support_gripper is not None
            and old_support_gripper in new_support_assignments
        )

        with self.timings.measure("create helper and target frames"):
            pickup_name = f"rod_{rod_id}_pickup_target"
            self.C.addFrame(pickup_name, "world") \
                .setPosition([-3.0, -1.0, 1.0]) \
                .setQuaternion([0.5, 0.0, 0.5, 0.70710678])

            candidate_hold_target = f"rod_{rod_id}_hold_target"
            self.C.addFrame(candidate_hold_target, "world")
            self.copy_frame_pose(rod, candidate_hold_target)

            grasp_1, grasp_2 = self.rods.create_dual_arm_grasp_frames(
                rod_id,
                d1_from_end=0.12,
                d12_between_arms=0.8,
            )

            continuing_gripper_targets: dict[str, str] = {}
            continuing_rod_targets: dict[str, str] = {}
            for support_gripper, supported_rod_id in continuing_supports.items():
                gripper_target = f"{support_gripper}_stay_target"
                rod_target = f"rod_{supported_rod_id}_stay_target"
                self.make_pose_target(support_gripper, gripper_target)
                self.make_pose_target(f"rod_{supported_rod_id}", rod_target)
                continuing_gripper_targets[support_gripper] = gripper_target
                continuing_rod_targets[support_gripper] = rod_target

            releasable_gripper_targets: dict[str, str] = {}
            for support_gripper, supported_rod_id in releasable_supports.items():
                if supported_rod_id == rod_id:
                    target = f"{support_gripper}_pre_release_stay_target"
                    self.make_pose_target(support_gripper, target)
                    releasable_gripper_targets[support_gripper] = target

        with self.timings.measure("construct phase schedule"):
            phases = PhaseSchedule()
            t_grasp = phases.add("main_grasp")

            t_old_support_away = None
            old_support_safe_name = None
            if candidate_is_supported and not old_support_is_reused:
                t_old_support_away = phases.add("old_support_away")
                old_support_safe_name = f"{old_support_gripper}_safe"
                self.C.addFrame(old_support_safe_name, "world") \
                    .setPosition([1.5, 1.5, 1.0]) \
                    .setShape(ry.ST.sphere, size=[0.04]) \
                    .setColor([1.0, 0.0, 0.0]) \
                    .setContact(0)

            support_phase_by_gripper: dict[str, float] = {}
            support_grasp_by_gripper: dict[str, str] = {}
            support_rod_by_gripper: dict[str, str] = {}
            support_target_by_gripper: dict[str, str] = {}

            for support_gripper, support_rod_id in new_support_assignments.items():
                support_phase_by_gripper[support_gripper] = phases.add(
                    f"support_rod_{support_rod_id}"
                )
                support_rod = f"rod_{support_rod_id}"
                support_rod_by_gripper[support_gripper] = support_rod
                support_grasp_by_gripper[support_gripper] = (
                    self.rods.create_support_grasp_frame_at_fraction(
                        support_rod_id,
                        support_fraction,
                    )
                )

                support_target = f"rod_{support_rod_id}_support_target"
                self.C.addFrame(support_target, "world")
                self.copy_frame_pose(support_rod, support_target)
                support_target_by_gripper[support_gripper] = support_target

            t_pickup = phases.add("move_to_pickup")

        with self.timings.measure("construct KOMO object"):
            komo = ry.KOMO(
                self.C,
                phases=phases.n_phases,
                slicesPerPhase=1,
                kOrder=1,
                enableCollisions=self.enabled("collisions"),
            )

        if self.enable_control:
            with self.timings.measure("add control objective"):
                komo.addControlObjective([], 1, 1e-2)

        self.add_group(
            "joint_limits",
            lambda: komo.addObjective(
                [], ry.FS.jointLimits, [], ry.OT.ineq, [1e0]
            ),
        )

        self.add_group(
            "collisions",
            lambda: komo.addObjective(
                [], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1e1]
            ),
        )

        def add_continuing_supports():
            for support_gripper, supported_rod_id in continuing_supports.items():
                _supported_rod = f"rod_{supported_rod_id}"
                gripper_target = continuing_gripper_targets[support_gripper]
                _rod_target = continuing_rod_targets[support_gripper]
                komo.addObjective(
                    [t_grasp, t_pickup],
                    ry.FS.positionDiff,
                    [support_gripper, gripper_target],
                    ry.OT.eq,
                    [1e2],
                )
                komo.addObjective(
                    [t_grasp, t_pickup],
                    ry.FS.quaternionDiff,
                    [support_gripper, gripper_target],
                    ry.OT.eq,
                    [1e2],
                )

        self.add_group("continuing_supports", add_continuing_supports)

        def add_releasable_supports():
            for support_gripper, gripper_target in releasable_gripper_targets.items():
                komo.addObjective(
                    [t_grasp],
                    ry.FS.positionDiff,
                    [support_gripper, gripper_target],
                    ry.OT.eq,
                    [1e2],
                )
                komo.addObjective(
                    [t_grasp],
                    ry.FS.quaternionDiff,
                    [support_gripper, gripper_target],
                    ry.OT.eq,
                    [1e2],
                )

        self.add_group("releasable_supports", add_releasable_supports)

        def add_candidate_hold():
            for t_support in support_phase_by_gripper.values():
                komo.addObjective(
                    [t_support],
                    ry.FS.positionDiff,
                    [rod, candidate_hold_target],
                    ry.OT.eq,
                    [1e2],
                )
                komo.addObjective(
                    [t_support],
                    ry.FS.quaternionDiff,
                    [rod, candidate_hold_target],
                    ry.OT.eq,
                    [1e1],
                )

        self.add_group("candidate_hold_during_support", add_candidate_hold)

        def add_main_grasp_positions():
            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.positionDiff,
                ["a1_ur_gripper_center", grasp_1],
                ry.OT.eq,
                [1e2],
            )
            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.positionDiff,
                ["a2_ur_gripper_center", grasp_2],
                ry.OT.eq,
                [1e2],
            )

        self.add_group("main_grasp_positions", add_main_grasp_positions)

        def add_main_grasp_orientations():
            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.scalarProductXZ,
                ["a1_ur_gripper_center", rod],
                ry.OT.eq,
                [1e2],
                [1.0],
            )
            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.scalarProductXZ,
                ["a2_ur_gripper_center", rod],
                ry.OT.eq,
                [1e2],
                [1.0],
            )
            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.scalarProductYY,
                ["a1_ur_gripper_center", "a2_ur_gripper_center"],
                ry.OT.eq,
                [1e1],
                [1.0],
            )

        self.add_group("main_grasp_orientations", add_main_grasp_orientations)

        self.add_group(
            "main_stable_switch",
            lambda: komo.addModeSwitch(
                [t_grasp, t_pickup],
                ry.SY.stable,
                ["a1_ur_gripper_center", rod],
                True,
            ),
        )

        def add_old_support_away():
            if t_old_support_away is not None:
                komo.addObjective(
                    [t_old_support_away],
                    ry.FS.positionDiff,
                    [old_support_gripper, old_support_safe_name],
                    ry.OT.eq,
                    [1e1],
                )

        self.add_group("old_support_away", add_old_support_away)

        def add_new_support_grasp():
            for support_gripper in new_support_assignments:
                t_support = support_phase_by_gripper[support_gripper]
                support_grasp = support_grasp_by_gripper[support_gripper]
                support_rod = support_rod_by_gripper[support_gripper]
                komo.addObjective(
                    [t_support, t_pickup],
                    ry.FS.positionDiff,
                    [support_gripper, support_grasp],
                    ry.OT.eq,
                    [1e2],
                )
                komo.addObjective(
                    [t_support, t_pickup],
                    ry.FS.scalarProductXZ,
                    [support_gripper, support_rod],
                    ry.OT.eq,
                    [1e1],
                    [-1.0],
                )

        self.add_group("new_support_grasp", add_new_support_grasp)

        def add_new_support_switches():
            for support_gripper in new_support_assignments:
                komo.addModeSwitch(
                    [support_phase_by_gripper[support_gripper], t_pickup],
                    ry.SY.stable,
                    [support_gripper, support_rod_by_gripper[support_gripper]],
                    True,
                )

        self.add_group("new_support_stable_switch", add_new_support_switches)

        def add_new_support_rod_pose():
            for support_gripper in new_support_assignments:
                t_support = support_phase_by_gripper[support_gripper]
                support_rod = support_rod_by_gripper[support_gripper]
                support_target = support_target_by_gripper[support_gripper]
                komo.addObjective(
                    [t_support, t_pickup],
                    ry.FS.positionDiff,
                    [support_rod, support_target],
                    ry.OT.eq,
                    [1e2],
                )
                komo.addObjective(
                    [t_support, t_pickup],
                    ry.FS.quaternionDiff,
                    [support_rod, support_target],
                    ry.OT.eq,
                    [1e2],
                )

        self.add_group("new_support_rod_pose", add_new_support_rod_pose)

        def add_pickup():
            komo.addObjective(
                [t_pickup],
                ry.FS.positionDiff,
                [rod, pickup_name],
                ry.OT.eq,
                [1e2],
            )
            komo.addObjective(
                [t_pickup],
                ry.FS.scalarProductZZ,
                [rod, pickup_name],
                ry.OT.eq,
                [1e2],
                [1.0],
            )

        self.add_group("pickup", add_pickup)

        phase_info = {
            "names": list(phases.names),
            "main_grasp_segment": int(t_grasp) - 1,
            "old_support_away_segment": (
                int(t_old_support_away) - 1
                if t_old_support_away is not None
                else None
            ),
            "new_support_segments": {
                gripper: int(t_support) - 1
                for gripper, t_support in support_phase_by_gripper.items()
            },
            "pickup_segment": int(t_pickup) - 1,
        }

        return ProblemContext(
            komo=komo,
            q0=q0,
            phases=phases,
            phase_info=phase_info,
        )


# ---------------------------------------------------------------------------
# Solving and evaluation
# ---------------------------------------------------------------------------


def initialization_for_attempt(
    mode: str,
    q0: np.ndarray,
    attempt: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "current":
        return q0.copy()

    if mode == "random":
        return (
            rng.random(len(q0)) * DEFAULT_RANDOM_MULTIPLIER
            + DEFAULT_RANDOM_OFFSET
        )

    if mode == "staged":
        if attempt == 0:
            return q0.copy()
        if attempt < 5:
            return q0 + rng.normal(0.0, 0.05, size=len(q0))
        if attempt < 9:
            return q0 + rng.normal(0.0, 0.20, size=len(q0))
        return (
            rng.random(len(q0)) * DEFAULT_RANDOM_MULTIPLIER
            + DEFAULT_RANDOM_OFFSET
        )

    raise ValueError(f"Unknown initialization mode: {mode}")


def print_problem_dimensions(context: ProblemContext, config: ry.Config, nlp):
    feature_types = nlp.getFeatureTypes()
    print("\nProblem dimensions")
    print("------------------")
    print(f"Frames:             {len(config.getFrameNames())}")
    print(f"Configuration DOFs: {len(config.getJointState())}")
    print(f"Phases:             {context.phases.n_phases}")
    print(f"Phase names:        {context.phases.names}")
    print(f"NLP variables:      {nlp.getDimension()}")
    print(f"NLP feature rows:   {len(feature_types)}")


def evaluate_nlp(
    nlp,
    repeats: int,
    timings: Timings,
):
    with timings.measure("create NLP initialization sample"):
        sample = nlp.getInitializationSample()

    phi = None
    jacobian = None
    durations = []

    for index in range(repeats):
        start = time.perf_counter()
        phi, jacobian = nlp.evaluate(sample)
        duration = time.perf_counter() - start
        durations.append(duration)
        timings.rows.append((f"NLP evaluate {index + 1}", duration))
        print(f"[time] NLP evaluate {index + 1:<28} {duration:>10.6f} s")

    print("\nNLP evaluation")
    print("--------------")
    print(f"Evaluations:        {repeats}")
    print(f"Mean time:          {np.mean(durations):.6f} s")
    print(f"Minimum time:       {np.min(durations):.6f} s")
    print(f"Maximum time:       {np.max(durations):.6f} s")
    print(f"phi shape:          {np.shape(phi)}")
    print(f"Jacobian shape:     {np.shape(jacobian)}")


def solve_problem(
    context: ProblemContext,
    attempts: int,
    initialization: str,
    seed: int,
    stop_evals: int | None,
    view_solution: bool,
    timings: Timings,
):
    rng = np.random.default_rng(seed)
    komo = context.komo

    for attempt in range(attempts):
        print(f"\nKOMO attempt {attempt + 1}/{attempts}")
        print("-" * 30)

        with timings.measure(f"attempt {attempt + 1}: initialization"):
            q_initial = initialization_for_attempt(
                initialization,
                context.q0,
                attempt,
                rng,
            )
            komo.initWithConstant(q_initial)

        with timings.measure(f"attempt {attempt + 1}: create NLP"):
            nlp = komo.nlp()

        with timings.measure(f"attempt {attempt + 1}: create solver"):
            solver = ry.NLP_Solver(nlp, verbose=0)
            if stop_evals is not None:
                solver.setOptions(stopEvals=stop_evals)

        try:
            with timings.measure(f"attempt {attempt + 1}: solve"):
                result = solver.solve()
        except RuntimeError as error:
            message = str(error)
            if "checkNan" in message or "inconsistent number" in message:
                print(f"Attempt crashed with a numerical error: {message}")
                continue
            raise

        result_dict = result.dict()
        print("Solver return:")
        print(result_dict)

        if result_dict["feasible"]:
            with timings.measure("extract feasible KOMO path"):
                keyframes = komo.getPath()

            print(f"Feasible path shape: {np.shape(keyframes)}")
            if view_solution:
                komo.view(True, "Standalone feasible KOMO solution")
            return keyframes, result_dict

    print("\nFAILED to find a feasible solution")
    return None, None


# ---------------------------------------------------------------------------
# CLI and experiment setup
# ---------------------------------------------------------------------------


def parse_assignment(text: str) -> tuple[str, int]:
    try:
        gripper, rod_text = text.split("=", maxsplit=1)
        return gripper, int(rod_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "Support assignments must have the form GRIPPER=ROD"
        ) from error


def assignments(values: list[tuple[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for gripper, rod_id in values:
        if gripper in result:
            raise ValueError(f"Support gripper assigned twice: {gripper}")
        result[gripper] = rod_id
    return result


def resolve_path(path_text: str, script_directory: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = script_directory / path
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--remove", type=int, required=True)
    parser.add_argument(
        "--active",
        type=int,
        nargs="*",
        default=None,
        help="Active rods; defaults to every rod in the JSON.",
    )
    parser.add_argument(
        "--continuing-support",
        type=parse_assignment,
        action="append",
        default=[],
        metavar="GRIPPER=ROD",
    )
    parser.add_argument(
        "--releasable-support",
        type=parse_assignment,
        action="append",
        default=[],
        metavar="GRIPPER=ROD",
    )
    parser.add_argument(
        "--new-support",
        type=parse_assignment,
        action="append",
        default=[],
        metavar="GRIPPER=ROD",
    )
    parser.add_argument(
        "--old-support-gripper",
        default=None,
        help="Defaults to the first releasable support gripper.",
    )
    parser.add_argument(
        "--q-start",
        default=None,
        help="Optional .npy file containing a full parent joint state.",
    )
    parser.add_argument("--support-fraction", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initialization",
        choices=("random", "current", "staged"),
        default="random",
        help="'random' exactly matches the current keyframes.py behavior.",
    )
    parser.add_argument(
        "--stop-evals",
        type=int,
        default=None,
        help="Optional per-attempt NLP evaluation limit.",
    )
    parser.add_argument(
        "--disable",
        action="append",
        choices=ALL_CONSTRAINT_GROUPS,
        default=[],
        metavar="GROUP",
        help=(
            "Disable one constraint group. Repeat for multiple groups. "
            "Groups: " + ", ".join(ALL_CONSTRAINT_GROUPS)
        ),
    )
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="Add order-1 control regularization; disabled in the current code.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Time NLP feature/Jacobian evaluation without solving.",
    )
    parser.add_argument("--eval-repeats", type=int, default=5)
    parser.add_argument("--report-problem", action="store_true")
    parser.add_argument("--view-scene", action="store_true")
    parser.add_argument("--view-solution", action="store_true")
    return parser


def validate_inputs(
    args,
    truss: TrussData,
    config: ry.Config,
    active_rods: frozenset[int],
    continuing: dict[str, int],
    releasable: dict[str, int],
    new_supports: dict[str, int],
):
    if args.remove not in active_rods:
        raise ValueError(f"Candidate rod {args.remove} is not active")

    unknown_rods = active_rods - truss.elements.keys()
    if unknown_rods:
        raise ValueError(f"Unknown active rods: {sorted(unknown_rods)}")

    all_frame_names = set(config.getFrameNames())
    all_assignments = {
        **continuing,
        **releasable,
        **new_supports,
    }

    for gripper, rod_id in all_assignments.items():
        if gripper not in all_frame_names:
            raise ValueError(f"Unknown support gripper frame: {gripper}")
        if rod_id not in active_rods:
            raise ValueError(f"Support target rod {rod_id} is not active")

    for gripper, rod_id in releasable.items():
        if rod_id != args.remove:
            raise ValueError(
                f"Releasable support {gripper} must hold candidate rod {args.remove}, "
                f"not rod {rod_id}"
            )

    if set(continuing) & set(releasable):
        raise ValueError("A gripper cannot be continuing and releasable")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if args.eval_repeats <= 0:
        parser.error("--eval-repeats must be positive")

    script_directory = Path(__file__).resolve().parent
    json_path = resolve_path(args.json, script_directory)
    model_root = resolve_path(args.model_root, script_directory)
    timings = Timings()

    print("Standalone full RAI removal profiler")
    print("====================================")
    print(f"JSON:           {json_path}")
    print(f"Model root:     {model_root}")
    print(f"Candidate rod:  {args.remove}")
    print(f"Disabled:       {args.disable or 'none'}")
    print(f"Initialization: {args.initialization}")

    with timings.measure("load truss JSON"):
        truss = TrussData.from_json(json_path)

    with timings.measure("create empty RAI configuration"):
        config = ry.Config()
        config.addFrame("world")

    scene = SceneBuilder(config, model_root)
    with timings.measure("load main and support robot models"):
        scene.import_all_robots()

    support_grippers = sorted(
        name
        for name in config.getFrameNames()
        if name.startswith("h") and name.endswith("gripper_center")
    )
    print(f"Support grippers detected: {support_grippers}")

    active_rods = (
        frozenset(truss.elements)
        if args.active is None
        else frozenset(args.active)
    )
    rods = RodManager(config, truss, radius=args.radius, scale=args.scale)

    with timings.measure("create and attach active rods"):
        for rod_id in sorted(active_rods):
            rods.create_rod_at_goal_pose(rod_id)
            config.attach("table", f"rod_{rod_id}")

    continuing = assignments(args.continuing_support)
    releasable = assignments(args.releasable_support)
    new_supports = assignments(args.new_support)

    validate_inputs(
        args,
        truss,
        config,
        active_rods,
        continuing,
        releasable,
        new_supports,
    )

    if args.q_start is not None:
        q_start_path = resolve_path(args.q_start, script_directory)
        with timings.measure("load and apply q_start"):
            q_start = np.load(q_start_path)
            expected_shape = config.getJointState().shape
            if q_start.shape != expected_shape:
                raise ValueError(
                    f"q_start has shape {q_start.shape}; expected {expected_shape}"
                )
            config.setJointState(q_start)

    supported_before = dict(continuing)
    supported_before.update(releasable)
    with timings.measure("restore existing support attachments"):
        for support_gripper, supported_rod in supported_before.items():
            config.attach(support_gripper, f"rod_{supported_rod}")

    old_support_gripper = args.old_support_gripper
    if old_support_gripper is None and releasable:
        old_support_gripper = next(iter(releasable))

    if args.view_scene:
        config.view(True, "Standalone RAI scene before KOMO")

    problem = FullRemovalProblem(
        config=config,
        rods=rods,
        timings=timings,
        disabled_groups=set(args.disable),
        enable_control=args.enable_control,
    )

    with timings.measure("build complete removal problem"):
        context = problem.build(
            rod_id=args.remove,
            continuing_supports=continuing,
            releasable_supports=releasable,
            new_support_assignments=new_supports,
            old_support_gripper=old_support_gripper,
            support_fraction=args.support_fraction,
        )

    with timings.measure("create NLP for inspection"):
        inspection_nlp = context.komo.nlp()

    print_problem_dimensions(context, config, inspection_nlp)

    if args.report_problem:
        print("\nKOMO problem report")
        print("-------------------")
        print(context.komo.reportProblem())

    if args.eval_only:
        evaluate_nlp(
            inspection_nlp,
            repeats=args.eval_repeats,
            timings=timings,
        )
        timings.print_summary()
        return

    solve_problem(
        context=context,
        attempts=args.attempts,
        initialization=args.initialization,
        seed=args.seed,
        stop_evals=args.stop_evals,
        view_solution=args.view_solution,
        timings=timings,
    )
    timings.print_summary()


if __name__ == "__main__":
    main()