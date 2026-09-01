# Stuff related to finding the keyframes

import numpy as np
import robotic as ry
from time import perf_counter
import time
from . import ur5e_ssik
from itertools import product


class PhaseSchedule:
    def __init__(self):
        self.names = []

    def add(self, name):
        self.names.append(name)
        return float(len(self.names))

    def get(self, name):
        return float(self.names.index(name) + 1)

    @property
    def n_phases(self):
        return len(self.names)
        
class KeyframePlanner:
    DEFAULT_BASE_CIRCLE_RADIUS = 0.4
    
    ARM_JOINT_SUFFIXES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    ARM_SPECS = {
        "a1": {
            "base_frame": "a1_base",
            "base_joint": "husky_base_XYPhi_joint",
            "joint_prefix": "a1_",
        },
        "a2": {
            "base_frame": "a2_base",
            "base_joint": "husky_base_XYPhi_joint",
            "joint_prefix": "a2_",
        },
        "h1_a1": {
            "base_frame": "h1_a1_base",
            "base_joint": "h1_base_XYPhi_joint",
            "joint_prefix": "h1_a1_",
        },
        "h2_a1": {
            "base_frame": "h2_a1_base",
            "base_joint": "h2_base_XYPhi_joint",
            "joint_prefix": "h2_a1_",
        },
    }

    def __init__(self, C, rod_manager):
        self.C = C
        self.rods = rod_manager
        

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        """Return world-Z yaw from a [w, x, y, z] quaternion."""
        w, x, y, z = np.asarray(quaternion, dtype=float)

        return np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def _mobile_base_joint_names(self):
        return [
            frame_name
            for frame_name in self.C.getFrameNames()
            if (
                frame_name == "husky_base_XYPhi_joint"
                or frame_name.endswith("_base_XYPhi_joint")
            )
        ]

    def _base_joint_for_gripper(self, gripper):
        if gripper.startswith(("a1_", "a2_")):
            base_joint = "husky_base_XYPhi_joint"

        else:
            robot_prefix = gripper.split("_", maxsplit=1)[0]
            base_joint = f"{robot_prefix}_base_XYPhi_joint"

        if self.C.getFrame(base_joint) is None:
            raise RuntimeError(
                f"Could not find mobile base {base_joint} "
                f"for gripper {gripper}"
            )

        return base_joint

    def _sample_bases_around_targets(
        self,
        x_init,
        q0,
        base_target_positions,
        rng,
        radius,
    ):
        """
        Keep randomly sampled arm joints, but replace the mobile-base
        configurations with samples on circles around the grasp targets.
        """
        if radius <= 0.0:
            raise ValueError("base circle radius must be positive")

        q_saved = self.C.getJointState().copy()

        try:
            # Obtain the current base configurations from q0.
            self.C.setJointState(q0)

            base_q0 = {}

            for base_joint in self._mobile_base_joint_names():
                frame = self.C.getFrame(base_joint)

                position = np.asarray(
                    frame.getPosition(),
                    dtype=float,
                )

                yaw = self._yaw_from_quaternion(
                    frame.getQuaternion()
                )

                base_q0[base_joint] = np.array([
                    position[0],
                    position[1],
                    yaw,
                ])

            # Apply the fully random initialization first.
            self.C.setJointState(x_init)

            # Restore every base to its current pose. Therefore idle robots
            # are not randomly moved.
            for base_joint, base_state in base_q0.items():
                self.C.setJointState(
                    base_state,
                    [base_joint],
                )

            # Override bases of robots that need to grasp a new rod.
            for base_joint, target_position in (
                base_target_positions.items()
            ):
                center = np.asarray(
                    target_position,
                    dtype=float,
                )

                circle_angle = rng.uniform(
                    -np.pi,
                    np.pi,
                )

                base_x = (
                    center[0]
                    + radius * np.cos(circle_angle)
                )

                base_y = (
                    center[1]
                    + radius * np.sin(circle_angle)
                )

                # The Husky's local +X direction is treated as its front.
                yaw_to_center = np.arctan2(
                    center[1] - base_y,
                    center[0] - base_x,
                )

                # Your base limits use [-3.14, 3.14].
                yaw_to_center = np.clip(
                    yaw_to_center,
                    -3.14,
                    3.14,
                )

                self.C.setJointState(
                    [
                        base_x,
                        base_y,
                        yaw_to_center,
                    ],
                    [base_joint],
                )

            return self.C.getJointState().copy()

        finally:
            # Do not leave the actual configuration at the sampled state.
            self.C.setJointState(q_saved)

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion):
        w, x, y, z = np.asarray(
            quaternion,
            dtype=float,
        )

        return np.array([
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ])

    def _frame_transform(self, frame_name):
        frame = self.C.getFrame(frame_name)

        if frame is None:
            raise RuntimeError(
                f"Missing frame: {frame_name}"
            )

        transform = np.eye(4)

        transform[:3, :3] = (
            self._quaternion_to_rotation_matrix(
                frame.getQuaternion()
            )
        )

        transform[:3, 3] = np.asarray(
            frame.getPosition(),
            dtype=float,
        )

        return transform
    
    def solve_komo(
        self,
        komo,
        view=False,
        view_accepted=False,
        base_target_positions=None,
        ik_targets=None,
        base_circle_radius=DEFAULT_BASE_CIRCLE_RADIUS,
        circle_samples=8,
        n_phases=None,
        activation_segment_by_arm=None,
    ):
        q0 = np.asarray(
            self.C.getJointState(),
            dtype=float,
        ).copy()

        base_target_positions = dict(
            base_target_positions or {}
        )

        ik_targets = dict(
            ik_targets or {}
        )
        
        activation_segment_by_arm = dict(
            activation_segment_by_arm or {}
        )

        if n_phases is None:
            raise ValueError(
                "n_phases is required for phase-dependent SSIK initialization"
            )

        # ------------------------------------------------------------
        # Attempt zero: previous configuration.
        # ------------------------------------------------------------

        keyframes = self._solve_komo_once(
            komo=komo,
            x_init=q0,
            label="previous configuration",
            view=view,
            view_accepted=view_accepted,
        )

        if keyframes is not None:
            return keyframes

        # ------------------------------------------------------------
        # Group analytical targets by physical mobile robot.
        #
        # a1 and a2 both belong to husky_base_XYPhi_joint and are
        # therefore solved as one combined robot candidate.
        # ------------------------------------------------------------

        ik_targets_by_base = {}

        for arm_name, target_spec in ik_targets.items():
            base_joint = self.ARM_SPECS[
                arm_name
            ]["base_joint"]

            ik_targets_by_base.setdefault(
                base_joint,
                {},
            )[arm_name] = target_spec

        candidate_lists = []
        candidate_base_names = []

        for base_joint, robot_ik_targets in (
            ik_targets_by_base.items()
        ):
            if base_joint not in base_target_positions:
                raise RuntimeError(
                    f"No circle centre provided for {base_joint}"
                )

            candidates = (
                self._precompute_robot_candidates(
                    q0=q0,
                    base_joint=base_joint,
                    circle_center=(
                        base_target_positions[
                            base_joint
                        ]
                    ),
                    robot_ik_targets=robot_ik_targets,
                    radius=base_circle_radius,
                    circle_samples=circle_samples,
                )
            )

            print(
                f"{base_joint}: "
                f"{len(candidates)}/{circle_samples} "
                "circle positions have SSIK solutions"
            )

            # If one required robot has no candidate, no global combination
            # can be feasible.
            if not candidates:
                print(
                    f"No SSIK candidate for required robot "
                    f"{base_joint}"
                )
                return None

            candidate_base_names.append(base_joint)
            candidate_lists.append(candidates)

        combination_count = int(
            np.prod([
                len(candidates)
                for candidates in candidate_lists
            ])
        )

        print(
            f"Testing {combination_count} "
            "cross-robot candidate combinations"
        )

        # ------------------------------------------------------------
        # Cartesian product of independently feasible robot states.
        # KOMO now checks interactions and collisions between robots.
        # ------------------------------------------------------------

        for combination_index, combination in enumerate(
            product(*candidate_lists),
            start=1,
        ):
            x_init = self._make_phase_ssik_initialization(
                q0=q0,
                robot_candidates=combination,
                n_phases=n_phases,
                activation_segment_by_arm=(
                    activation_segment_by_arm
                ),
            )

            description = ", ".join(
                (
                    f"{base_name}:"
                    f"{candidate['circle_index']}"
                )
                for base_name, candidate in zip(
                    candidate_base_names,
                    combination,
                )
            )

            keyframes = self._solve_komo_once(
                komo=komo,
                x_init=x_init,
                label=(
                    f"combination "
                    f"{combination_index}/"
                    f"{combination_count} "
                    f"({description})"
                ),
                view=view,
                view_accepted=view_accepted,
            )

            if keyframes is not None:
                # komo.view(True)
                return keyframes

        print(
            "FAILED: all analytical robot "
            "combinations were tested"
        )

        return None

    def copy_frame_pose(self, source_frame_name, target_frame_name):
        source = self.C.getFrame(source_frame_name)
        if source is None:
            raise RuntimeError(
                f"Cannot copy pose. Source frame does not exist: {source_frame_name}"
            )

        if target_frame_name not in self.C.getFrameNames():
            raise RuntimeError(
                f"Cannot copy pose. Target frame does not exist: {target_frame_name}"
            )

        target = self.C.getFrame(target_frame_name)
        target.setPosition(source.getPosition())
        target.setQuaternion(source.getQuaternion())

        return target_frame_name

    def make_pose_target_from_frame(
        self,
        source_frame_name,
        target_frame_name,
        parent="world",
        marker_size=0.06,
    ):
        """
        Create or update a fixed target frame with the current pose of source_frame_name.

        IMPORTANT:
        Call this before constructing KOMO objectives that reference the target.
        """

        source = self.C.getFrame(source_frame_name)
        if source is None:
            raise RuntimeError(
                f"Cannot create target. Source frame does not exist: {source_frame_name}"
            )

        if target_frame_name not in self.C.getFrameNames():
            target = self.C.addFrame(target_frame_name, parent)
            target.setShape(ry.ST.marker, [marker_size])
            target.setContact(0)

        self.copy_frame_pose(source_frame_name, target_frame_name)

        return target_frame_name
 
    def _make_ssik_target_transform(
        self,
        position,
        rod_rotation,
        alignment,
        approach_direction,
        roll_offset=0.0,
    ):
        """
        Align gripper X with the rod axis and orient gripper Z
        approximately from the arm base toward the grasp point.
        """
        position = np.asarray(
            position,
            dtype=float,
        )

        rod_rotation = np.asarray(
            rod_rotation,
            dtype=float,
        )

        approach_direction = np.asarray(
            approach_direction,
            dtype=float,
        )

        # Main arm: gripper X = +rod Z.
        # Support arm: gripper X = -rod Z.
        x_axis = (
            float(alignment)
            * rod_rotation[:, 2]
        )
        x_axis /= np.linalg.norm(x_axis)

        # Project the base-to-target direction onto the plane
        # perpendicular to the rod.
        z_reference = (
            approach_direction
            - np.dot(
                approach_direction,
                x_axis,
            ) * x_axis
        )

        # If the base lies almost exactly along the rod axis,
        # use the rod's Y-axis as fallback.
        if np.linalg.norm(z_reference) < 1e-8:
            z_reference = rod_rotation[:, 1].copy()

            z_reference -= (
                np.dot(z_reference, x_axis)
                * x_axis
            )

        z_reference /= np.linalg.norm(z_reference)

        y_reference = np.cross(
            z_reference,
            x_axis,
        )
        y_reference /= np.linalg.norm(y_reference)

        # Test nearby orientations by rotating around the rod axis.
        z_axis = (
            np.cos(roll_offset) * z_reference
            + np.sin(roll_offset) * y_reference
        )
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(
            z_axis,
            x_axis,
        )
        y_axis /= np.linalg.norm(y_axis)

        target = np.eye(4)

        target[:3, :3] = np.column_stack([
            x_axis,
            y_axis,
            z_axis,
        ])

        target[:3, 3] = position

        return target

    def _apply_ssik_solution(
        self,
        arm_name,
        target_world,
    ):
        spec = self.ARM_SPECS[arm_name]

        joint_names = [
            spec["joint_prefix"] + suffix
            for suffix in self.ARM_JOINT_SUFFIXES
        ]

        q_seed = np.array(
            [
                float(
                    np.asarray(
                        self.C.getFrame(
                            joint_name
                        ).getJointState(),
                        dtype=float,
                    ).reshape(-1)[0]
                )
                for joint_name in joint_names
            ],
            dtype=float,
        )

        T_world_base = self._frame_transform(
            spec["base_frame"]
        )

        # SSIK expects gripper pose relative to its arm base.
        T_base_target = (
            np.linalg.inv(T_world_base)
            @ target_world
        )

        solutions = ur5e_ssik.solve(
            T_base_target,
            q_seed=q_seed,
            max_solutions=1,
        )

        if not solutions:
            return None

        q_arm = np.asarray(
            solutions[0].q,
            dtype=float,
        )

        self.C.setJointState(
            q_arm,
            joint_names,
        )

        return q_arm


    def _make_ssik_initialization(
        self,
        sampled_q,
        ik_targets,
        rng,
    ):
        """
        Insert analytical arm solutions into a state that already contains
        the sampled mobile-base positions.
        """
        q_saved = self.C.getJointState().copy()

        try:
            self.C.setJointState(sampled_q)

            roll_by_group = {}

            for arm_name, target_spec in ik_targets.items():
                roll_group = target_spec.get(
                    "roll_group",
                    arm_name,
                )

                # a1 and a2 use the same roll so their Y-axes are parallel.
                if roll_group not in roll_by_group:
                    roll_by_group[roll_group] = rng.uniform(
                        -np.pi,
                        np.pi,
                    )

                target_world = (
                    self._make_ssik_target_transform(
                        position=target_spec["position"],
                        rod_rotation=target_spec["rod_rotation"],
                        alignment=target_spec["alignment"],
                        roll=roll_by_group[roll_group],
                    )
                )

                if not self._apply_ssik_solution(
                    arm_name,
                    target_world,
                ):
                    return None

            return self.C.getJointState().copy()

        finally:
            self.C.setJointState(q_saved)
    
    # def get_remove_keyframes_dual_test(
    #     self,
    #     rod_id,
    #     supported=None,
    #     support_q=None,
    #     candidate_is_supported=False,
    #     old_support_gripper=None,
    #     continuing_supports=None,
    #     releasable_supports=None,
    #     new_support_assignments=None,
    #     support_fraction=0.5,
    # ):
    #     """Deterministic synthetic replacement for get_remove_keyframes_dual()."""

    #     supported = dict(supported or {})
    #     support_q = dict(support_q or {})
    #     continuing_supports = dict(continuing_supports or {})
    #     releasable_supports = dict(releasable_supports or {})
    #     new_support_assignments = dict(new_support_assignments or {})

    #     q0 = self.C.getJointState().copy()

    #     # Main arm 2 is deliberately excluded.
    #     moving_grippers = [
    #         "a1_ur_gripper_center",
    #         "h1_a1_ur_gripper_center",
    #         "h2_a1_ur_gripper_center",
    #     ]

    #     missing_grippers = [
    #         gripper
    #         for gripper in moving_grippers
    #         if self.C.getFrame(gripper) is None
    #     ]

    #     if missing_grippers:
    #         raise RuntimeError(
    #             f"Missing test grippers: {missing_grippers}"
    #         )

    #     # ------------------------------------------------------------
    #     # Deterministic target positions
    #     # ------------------------------------------------------------

    #     structure_points = (
    #         np.asarray(
    #             list(self.rods.truss.nodes.values()),
    #             dtype=float,
    #         )
    #         * self.rods.scale
    #     )

    #     structure_center_x = 0.5 * (
    #         structure_points[:, 0].min()
    #         + structure_points[:, 0].max()
    #     )

    #     structure_min_y = structure_points[:, 1].min()

    #     # Entire target cluster is placed two metres beyond the
    #     # structure's negative-Y boundary.
    #     cluster_center = np.array([
    #         structure_center_x,
    #         structure_min_y - 2.0,
    #         0.88,
    #     ])

    #     # Consecutive target frames are exactly 60 cm apart in Y.
    #     target_offsets = {
    #         "a1_ur_gripper_center": np.array([
    #             0.0,
    #             -0.20,
    #             0.0,
    #         ]),
    #         "h1_a1_ur_gripper_center": np.array([
    #             0.0,
    #             0.0,
    #             0.0,
    #         ]),
    #         "h2_a1_ur_gripper_center": np.array([
    #             0.0,
    #             0.2,
    #             0.0,
    #         ]),
    #     }

    #     target_by_gripper = {}
    #     target_positions = {}

    #     for index, gripper in enumerate(moving_grippers):
    #         target_position = (
    #             cluster_center
    #             + target_offsets[gripper]
    #         )

    #         target_name = (
    #             f"test_target_rod_{rod_id}_{index}"
    #         )

    #         if target_name not in self.C.getFrameNames():
    #             self.C.addFrame(
    #                 target_name,
    #                 "world",
    #             )

    #         gripper_frame = self.C.getFrame(gripper)
    #         target_frame = self.C.getFrame(target_name)

    #         target_frame.setPosition(
    #             target_position
    #         )

    #         # Preserve the gripper's current orientation.
    #         target_frame.setQuaternion(
    #             gripper_frame.getQuaternion()
    #         )

    #         target_frame.setShape(
    #             ry.ST.marker,
    #             [0.12],
    #         )

    #         target_frame.setContact(0)

    #         target_by_gripper[gripper] = target_name
    #         target_positions[gripper] = target_position

    #         print(
    #             f"Test target for {gripper}: "
    #             f"{target_position}"
    #         )

    #     main_h1_distance = np.linalg.norm(
    #         target_positions["a1_ur_gripper_center"]
    #         - target_positions["h1_a1_ur_gripper_center"]
    #     )

    #     h1_h2_distance = np.linalg.norm(
    #         target_positions["h1_a1_ur_gripper_center"]
    #         - target_positions["h2_a1_ur_gripper_center"]
    #     )

    #     print(
    #         f"Target distances: "
    #         f"main-h1={main_h1_distance:.2f} m, "
    #         f"h1-h2={h1_h2_distance:.2f} m"
    #     )

        
    #     # ------------------------------------------------------------
    #     # Construct synthetic KOMO problem
    #     # ------------------------------------------------------------

    #     komo = ry.KOMO(
    #         self.C,
    #         phases=1,
    #         slicesPerPhase=1,
    #         kOrder=2,
    #         enableCollisions=True,
    #     )

    #     komo.addControlObjective(
    #         [],
    #         0,
    #         1e-3,
    #     )

    #     komo.addControlObjective(
    #         [],
    #         1,
    #         1e-2,
    #     )

    #     komo.addObjective(
    #         [],
    #         ry.FS.jointLimits,
    #         [],
    #         ry.OT.ineq,
    #         [1e0],
    #     )

    #     komo.addObjective(
    #         [],
    #         ry.FS.accumulatedCollisions,
    #         [],
    #         ry.OT.ineq,
    #         [1e1],
    #     )

    #     for gripper in moving_grippers:
            
    #         target_name = target_by_gripper[gripper]

    #         komo.addObjective(
    #             [1],
    #             ry.FS.positionDiff,
    #             [gripper, target_name],
    #             ry.OT.eq,
    #             [1e2],
    #         )

    #         # komo.addObjective(
    #         #     [1],
    #         #     ry.FS.quaternionDiff,
    #         #     [gripper, target_name],
    #         #     ry.OT.eq,
    #         #     [1e1],
    #         # )
            
            
    #     # The main base circles the midpoint between its two grasp targets.
    #     base_target_positions = {
    #         "husky_base_XYPhi_joint": 0.5 * (
    #             np.asarray(
    #                 self.C.getFrame(g1).getPosition(),
    #                 dtype=float,
    #             )
    #             + np.asarray(
    #                 self.C.getFrame(g2).getPosition(),
    #                 dtype=float,
    #             )
    #         )
    #     }

    #     # Every newly deployed support robot circles its support-grasp point.
    #     for support_gripper, support_grasp in (
    #         support_grasp_by_gripper.items()
    #     ):
    #         base_joint = self._base_joint_for_gripper(
    #             support_gripper
    #         )

    #         base_target_positions[base_joint] = np.asarray(
    #             self.C.getFrame(
    #                 support_grasp
    #             ).getPosition(),
    #             dtype=float,
    #         )

    #     keyframes = self.solve_komo(
    #         komo,
    #         attempts=6,
    #         view=True,
    #         base_target_positions=base_target_positions,
    #     )

    #     # ------------------------------------------------------------
    #     # Match the original return interface
    #     # ------------------------------------------------------------

    #     new_supported = {}
    #     new_supported.update(continuing_supports)
    #     new_supported.update(new_support_assignments)
        
    #     phase_info = {
    #         "main_grasp_segment": 0,
    #         "old_support_away_segment": None,
    #         "new_support_segments": {
    #             support_gripper: 0
    #             for support_gripper in new_support_assignments
    #         },
    #         "pickup_segment": 0,
    #     }

    #     return (
    #         keyframes,
    #         q0,
    #         new_supported,
    #         phase_info,
    #     )   
    
    def _precompute_robot_candidates(
        self,
        q0,
        base_joint,
        circle_center,
        robot_ik_targets,
        radius,
        circle_samples=8,
    ):
        """
        Return at most one valid combined base/arm configuration for each
        equally spaced base position.

        For the main robot, robot_ik_targets contains both a1 and a2.
        A candidate is saved only if both arms solve.
        """
        q_saved = self.C.getJointState().copy()
        candidates = []

        # Direct base-facing orientation first, followed by nearby alternatives.
        roll_offsets = np.deg2rad([
            0.0,
            30.0,
            -30.0,
            60.0,
            -60.0,
            180.0,
        ])

        circle_center = np.asarray(
            circle_center,
            dtype=float,
        )

        circle_angles = np.linspace(
            -np.pi,
            np.pi,
            circle_samples,
            endpoint=False,
        )

        try:
            for sample_index, circle_angle in enumerate(
                circle_angles
            ):
                self.C.setJointState(q0)

                base_x = (
                    circle_center[0]
                    + radius * np.cos(circle_angle)
                )

                base_y = (
                    circle_center[1]
                    + radius * np.sin(circle_angle)
                )

                yaw = np.arctan2(
                    circle_center[1] - base_y,
                    circle_center[0] - base_x,
                )

                yaw = np.clip(
                    yaw,
                    -3.14,
                    3.14,
                )

                base_q = np.array([
                    base_x,
                    base_y,
                    yaw,
                ])

                self.C.setJointState(
                    base_q,
                    [base_joint],
                )

                q_at_sampled_base = (
                    self.C.getJointState().copy()
                )

                # For the main robot, use the centre of both targets and
                # both arm bases. This gives both arms the same orientation.
                target_center = np.mean(
                    [
                        target_spec["position"]
                        for target_spec
                        in robot_ik_targets.values()
                    ],
                    axis=0,
                )

                arm_base_center = np.mean(
                    [
                        np.asarray(
                            self.C.getFrame(
                                self.ARM_SPECS[
                                    arm_name
                                ]["base_frame"]
                            ).getPosition(),
                            dtype=float,
                        )
                        for arm_name
                        in robot_ik_targets
                    ],
                    axis=0,
                )

                approach_direction = (
                    target_center - arm_base_center
                )

                # Try the preferred orientation and a few rotations around
                # the rod. Save the first complete solution at this base.
                for roll_offset in roll_offsets:
                    self.C.setJointState(
                        q_at_sampled_base
                    )

                    arm_solutions = {}
                    sample_is_valid = True

                    for arm_name, target_spec in (
                        robot_ik_targets.items()
                    ):
                        target_world = (
                            self._make_ssik_target_transform(
                                position=target_spec[
                                    "position"
                                ],
                                rod_rotation=target_spec[
                                    "rod_rotation"
                                ],
                                alignment=target_spec[
                                    "alignment"
                                ],
                                approach_direction=(
                                    approach_direction
                                ),
                                roll_offset=roll_offset,
                            )
                        )

                        q_arm = self._apply_ssik_solution(
                            arm_name,
                            target_world,
                        )

                        if q_arm is None:
                            sample_is_valid = False
                            break

                        arm_solutions[arm_name] = (
                            q_arm.copy()
                        )

                    if sample_is_valid:
                        candidates.append({
                            "base_joint": base_joint,
                            "base_q": base_q.copy(),
                            "arm_solutions": arm_solutions,
                            "circle_index": sample_index,
                            "circle_angle": circle_angle,
                            "roll_offset": roll_offset,
                        })

                        # At most one candidate per circle position.
                        break

            return candidates

        finally:
            self.C.setJointState(q_saved)
   
    def _combine_robot_candidates(
        self,
        q0,
        robot_candidates,
    ):
        q_saved = self.C.getJointState().copy()

        try:
            self.C.setJointState(q0)

            for candidate in robot_candidates:
                self.C.setJointState(
                    candidate["base_q"],
                    [candidate["base_joint"]],
                )

                for arm_name, q_arm in (
                    candidate["arm_solutions"].items()
                ):
                    spec = self.ARM_SPECS[arm_name]

                    joint_names = [
                        spec["joint_prefix"] + suffix
                        for suffix
                        in self.ARM_JOINT_SUFFIXES
                    ]

                    self.C.setJointState(
                        q_arm,
                        joint_names,
                    )

            return self.C.getJointState().copy()

        finally:
            self.C.setJointState(q_saved)
            
    def _make_phase_ssik_initialization(
        self,
        q0,
        robot_candidates,
        n_phases,
        activation_segment_by_arm,
    ):
        """
        Build one initialization configuration per KOMO phase.

        A robot candidate is applied only from the phase in which that
        robot actually reaches its grasp.
        """
        candidate_activation_segments = []

        for candidate in robot_candidates:
            activation_segments = {
                activation_segment_by_arm[arm_name]
                for arm_name in candidate["arm_solutions"]
            }

            # All arms belonging to one physical robot candidate must become
            # active in the same phase. This is true for a1/a2 on the main robot.
            if len(activation_segments) != 1:
                raise RuntimeError(
                    "Arms belonging to the same mobile robot have "
                    f"different activation phases: {activation_segments}"
                )

            candidate_activation_segments.append(
                activation_segments.pop()
            )

        initialization_path = []

        for segment_index in range(n_phases):
            active_candidates = [
                candidate
                for candidate, activation_segment
                in zip(
                    robot_candidates,
                    candidate_activation_segments,
                )
                if segment_index >= activation_segment
            ]

            q_phase = self._combine_robot_candidates(
                q0=q0,
                robot_candidates=active_candidates,
            )

            initialization_path.append(q_phase)

        return np.asarray(
            initialization_path,
            dtype=float,
        )
            
    def _solve_komo_once(
        self,
        komo,
        x_init,
        label,
        view=False,
        view_accepted=False,
    ):
        
        x_init = np.asarray(
            x_init,
            dtype=float,
        )

        if x_init.ndim == 1:
            # First attempt: previous configuration in every phase.
            komo.initWithConstant(x_init)

        elif x_init.ndim == 2:
            # Analytical attempt: different initialization per phase.
            komo.initWithPath(x_init)

        else:
            raise ValueError(
                f"Invalid initialization shape: {x_init.shape}"
            )
        
        # komo.view(True, f"KOMO initialization: {label}")

        solver = ry.NLP_Solver(
            komo.nlp(),
            verbose=0,
        )
        
        try:
            retval = solver.solve().dict()

        except RuntimeError as error:
            print(
                f"{label}: KOMO failed with error: {error}"
            )
            return None

        print(f"{label}: {retval}")

        # view = True
        if view:
            komo.view(
                True,
                f"KOMO: {label}",
            )

        if not retval["feasible"]:
            return None

        if view_accepted:
            komo.view(
                False,
                f"Accepted: {label}",
            )

        return komo.getPath()
   
    def get_remove_keyframes_dual(
        self,
        rod_id,
        supported=None,
        support_q=None,
        candidate_is_supported=False,
        old_support_gripper=None,
        continuing_supports=None,
        releasable_supports=None,
        new_support_assignments=None,
        support_fraction=0.5,
    ):
        """
        Backward removal of rod_id.

        Desired order:
        1. Main robot grasps candidate rod X.
        2. If X was supported, old support releases/moves away.
        3. Support robots move to affected rods Y that would become unstable.
        4. Main robot removes X to pickup/staging pose.

        supported:
            support_gripper -> rod_id currently supported before this removal

        support_q:
            support_gripper -> full q at which that support robot should stay fixed
            This is kept for compatibility, but continuing supports are now locked
            by gripper/rod pose instead of qItself.

        continuing_supports:
            support_gripper -> rod_id for support robots that already hold a rod
            and must keep holding it during this removal.

        releasable_supports:
            support_gripper -> rod_id for support robots holding the candidate rod.
            These may release after the main robot grasps the candidate.

        new_support_assignments:
            support_gripper -> affected_rod_id that must be newly supported
            before removing rod_id.
        """

        rod = f"rod_{rod_id}"
        q0 = self.C.getJointState().copy()

        supported = dict(supported or {})
        support_q = dict(support_q or {})
        continuing_supports = dict(continuing_supports or {})
        releasable_supports = dict(releasable_supports or {})
        new_support_assignments = dict(new_support_assignments or {})

        # Determine whether the old support is being reused for a new affected rod.
        old_support_is_reused = (
            old_support_gripper is not None
            and old_support_gripper in new_support_assignments
        )

        # ------------------------------------------------------------
        # Helper frames
        # ------------------------------------------------------------

        # Create a fixed target frame for the candidate rod at its pickup pose.
        pickup_name = f"rod_{rod_id}_pickup_target"

        if pickup_name not in self.C.getFrameNames():
            self.C.addFrame(pickup_name, "world")

        self.C.getFrame(pickup_name) \
            .setPosition([-3, -1, 1.0]) \
            .setQuaternion([0.5, 0.0, 0.5, 0.70710678])


        # Create a fixed target frame for the candidate rod at its installed pose.
        candidate_hold_target = f"rod_{rod_id}_hold_target"

        if candidate_hold_target not in self.C.getFrameNames():
            self.C.addFrame(candidate_hold_target, "world")

        self.copy_frame_pose(rod, candidate_hold_target)

        # TODO: Give it more flexibility if only one needs to grab at connector
        rod_length = self.rods.get_rod_length(rod_id)

        # Keep both grasps away from the rod ends.
        end_margin = min(0.12, 0.20 * rod_length)

        # Use the preferred 0.8 m separation when possible,
        # otherwise use the largest separation that fits.
        grasp_separation = min(
            0.8,
            rod_length - 2.0 * end_margin,
        ) 

        if grasp_separation <= 0.0:
            raise ValueError(
                f"Rod {rod_id} is too short for a dual-arm grasp: "
                f"length={rod_length}"
            )

        g1, g2 = self.rods.create_dual_arm_grasp_frames(
            rod_id,
            d1_from_end=end_margin,
            d12_between_arms=grasp_separation,
        )

        # Fixed target frames for already-active continuing supports.
        # These targets are created before KOMO is constructed.
        continuing_gripper_target_by_gripper = {}
        continuing_rod_target_by_gripper = {}

        for support_gripper, supported_rod_id in continuing_supports.items():
            supported_rod = f"rod_{supported_rod_id}"

            gripper_target = f"{support_gripper}_stay_target"
            rod_target = f"rod_{supported_rod_id}_stay_target"

            self.make_pose_target_from_frame(
                source_frame_name=support_gripper,
                target_frame_name=gripper_target,
            )

            self.make_pose_target_from_frame(
                source_frame_name=supported_rod,
                target_frame_name=rod_target,
            )

            continuing_gripper_target_by_gripper[support_gripper] = gripper_target
            continuing_rod_target_by_gripper[support_gripper] = rod_target

        # Fixed target frames for supports that currently hold the candidate rod.
        # These supports are allowed to release only after the main robot has
        # reached the candidate at t_grasp. Until t_grasp, they must stay put.
        releasable_gripper_target_by_gripper = {}

        for support_gripper, supported_rod_id in releasable_supports.items():
            if supported_rod_id != rod_id:
                continue

            gripper_target = f"{support_gripper}_pre_release_stay_target"

            self.make_pose_target_from_frame(
                source_frame_name=support_gripper,
                target_frame_name=gripper_target,
            )

            releasable_gripper_target_by_gripper[support_gripper] = gripper_target

        # ------------------------------------------------------------
        # Phase schedule
        # ------------------------------------------------------------

        phases = PhaseSchedule()

        t_grasp = phases.add("main_grasp")

        t_old_support_away = None
        old_support_safe_name = None

        if (
            candidate_is_supported
            and not old_support_is_reused
        ):
            t_old_support_away = phases.add("old_support_away")
            old_support_safe_name = f"{old_support_gripper}_safe"

            if old_support_safe_name not in self.C.getFrameNames():
                self.C.addFrame(old_support_safe_name, "world") \
                    .setPosition([1.5, 1.5, 1.0]) \
                    .setShape(ry.ST.sphere, size=[0.04]) \
                    .setColor([1.0, 0.0, 0.0]) \
                    .setContact(0)

        support_phase_by_gripper = {}
        support_grasp_by_gripper = {}
        support_rod_frame_by_gripper = {}
        support_target_by_gripper = {}

        for support_gripper, support_rod_id in new_support_assignments.items():
            t_support = phases.add(f"support_rod_{support_rod_id}")
            support_phase_by_gripper[support_gripper] = t_support

            support_rod = f"rod_{support_rod_id}"
            support_rod_frame_by_gripper[support_gripper] = support_rod

            support_grasp = self.rods.create_support_grasp_frame_at_fraction(
                support_rod_id,
                support_fraction,
            )
            support_grasp_by_gripper[support_gripper] = support_grasp

            # Fixed target frame at the current installed pose of the support rod.
            # IMPORTANT: must exist before ry.KOMO(...) is constructed.
            support_target = f"rod_{support_rod_id}_support_target"
            support_target_by_gripper[support_gripper] = support_target

            if support_target not in self.C.getFrameNames():
                self.C.addFrame(support_target, "world")

            self.copy_frame_pose(support_rod, support_target)

        t_pickup = phases.add("move_to_pickup")

        phase_info = {
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
        
        activation_segment_by_arm = {
            "a1": phase_info["main_grasp_segment"],
            "a2": phase_info["main_grasp_segment"],
        }

        for support_gripper in support_grasp_by_gripper:
            arm_name = support_gripper.removesuffix(
                "_ur_gripper_center"
            )

            activation_segment_by_arm[arm_name] = (
                phase_info["new_support_segments"][
                    support_gripper
                ]
            )

        # ------------------------------------------------------------
        # KOMO
        # ------------------------------------------------------------

        komo = ry.KOMO(
            self.C,
            phases=phases.n_phases,
            slicesPerPhase=1,
            kOrder=1,
            enableCollisions=True,
        )

        # komo.addControlObjective([], 0, 1e-1)
        # komo.addControlObjective([], 1, 1e-1)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.ineq, [1])

        # ------------------------------------------------------------
        # Keep continuing support robots exactly in place.
        #
        # This replaces the previous qItself freeze.
        # qItself is too indirect here and can also create conflicts.
        # The actual requirement is:
        #   - this support gripper stays at its current support pose
        #   - the rod it supports stays at its installed pose
        # ------------------------------------------------------------

        for support_gripper, supported_rod_id in continuing_supports.items():
            supported_rod = f"rod_{supported_rod_id}"

            gripper_target = continuing_gripper_target_by_gripper[support_gripper]
            rod_target = continuing_rod_target_by_gripper[support_gripper]

            # Keep the support gripper at its current pose over the whole plan.
            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.positionDiff,
                [support_gripper, gripper_target],
                ry.OT.eq,
                [1e1],
            )

            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.quaternionDiff,
                [support_gripper, gripper_target],
                ry.OT.eq,
                [1e1],
            )

            # Keep the rod supported by this continuing support in its installed pose.
            # komo.addObjective(
            #     [t_grasp, t_pickup],
            #     ry.FS.positionDiff,
            #     [supported_rod, rod_target],
            #     ry.OT.eq,
            #     [1e2],
            # )

            # komo.addObjective(
            #     [t_grasp, t_pickup],
            #     ry.FS.quaternionDiff,
            #     [supported_rod, rod_target],
            #     ry.OT.eq,
            #     [1e2],
            # )

        # ------------------------------------------------------------
        # Keep support robots that hold the candidate fixed until main grasp.
        #
        # These are not continuing supports because they will release the
        # candidate rod after the main robot takes over. But they still must
        # not move before t_grasp.
        # ------------------------------------------------------------

        for support_gripper, gripper_target in releasable_gripper_target_by_gripper.items():
            komo.addObjective(
                [t_grasp],
                ry.FS.positionDiff,
                [support_gripper, gripper_target],
                ry.OT.eq,
                [1e1],
            )

            komo.addObjective(
                [t_grasp],
                ry.FS.quaternionDiff,
                [support_gripper, gripper_target],
                ry.OT.eq,
                [1e1],
            )

            # Because the candidate rod is still installed and still supported
            # before the handover, keep it in the scaffold pose at t_grasp.
            # komo.addObjective(
            #     [t_grasp],
            #     ry.FS.positionDiff,
            #     [rod, candidate_hold_target],
            #     ry.OT.eq,
            #     [1e2],
            # )

            # komo.addObjective(
            #     [t_grasp],
            #     ry.FS.quaternionDiff,
            #     [rod, candidate_hold_target],
            #     ry.OT.eq,
            #     [1e2],
            # )

        # During support phases, the candidate rod held by the main robot
        # must stay fixed in its installed scaffold pose.
        # for support_gripper, t_support in support_phase_by_gripper.items():
        #     komo.addObjective(
        #         [t_support],
        #         ry.FS.positionDiff,
        #         [rod, candidate_hold_target],
        #         ry.OT.eq,
        #         [1e2],
        #     )

        #     komo.addObjective(
        #         [t_support],
        #         ry.FS.quaternionDiff,
        #         [rod, candidate_hold_target],
        #         ry.OT.eq,
        #         [1e1],
        #     )
            
        last_installed_phase = t_pickup - 1.0

        komo.addObjective(
            [t_grasp, last_installed_phase],
            ry.FS.positionDiff,
            [rod, candidate_hold_target],
            ry.OT.eq,
            [1e1],
        )

        komo.addObjective(
            [t_grasp, last_installed_phase],
            ry.FS.quaternionDiff,
            [rod, candidate_hold_target],
            ry.OT.eq,
            [1e1],
        )

        # ------------------------------------------------------------
        # Main grasps candidate rod and keeps it until pickup.
        # ------------------------------------------------------------

        komo.addObjective(
            [t_grasp, t_pickup],
            ry.FS.positionDiff,
            ["a1_ur_gripper_center", g1],
            ry.OT.eq,
            [1e1],
        )

        komo.addObjective(
            [t_grasp, t_pickup],
            ry.FS.positionDiff,
            ["a2_ur_gripper_center", g2],
            ry.OT.eq,
            [1e1],
        )

        komo.addObjective(
            [t_grasp, t_pickup],
            ry.FS.scalarProductXZ,
            ["a1_ur_gripper_center", rod],
            ry.OT.eq,
            [1e1],
            [1.0],
        )

        komo.addObjective(
            [t_grasp, t_pickup],
            ry.FS.scalarProductXZ,
            ["a2_ur_gripper_center", rod],
            ry.OT.eq,
            [1e1],
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

        komo.addModeSwitch(
            [t_grasp, t_pickup],
            ry.SY.stable,
            ["a1_ur_gripper_center", rod],
            True,
        )

        # ------------------------------------------------------------
        # If candidate was supported, old support moves away after
        # main has grasped candidate.
        # ------------------------------------------------------------

        if t_old_support_away is not None:
            komo.addObjective(
                [t_old_support_away],
                ry.FS.positionDiff,
                [old_support_gripper, old_support_safe_name],
                ry.OT.eq,
                [1e1],
            )

        # ------------------------------------------------------------
        # Newly affected rods are supported before candidate is removed.
        # ------------------------------------------------------------

        for support_gripper, support_rod_id in new_support_assignments.items():
            t_support = support_phase_by_gripper[support_gripper]
            support_grasp = support_grasp_by_gripper[support_gripper]
            support_rod = support_rod_frame_by_gripper[support_gripper]

            support_target = support_target_by_gripper[support_gripper]

            # Support gripper moves onto the affected rod.
            komo.addObjective(
                [t_support, t_pickup],
                ry.FS.positionRel,
                [support_gripper, support_rod],
                ry.OT.eq,
                1e2 * np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]),
            )

            length = self.rods.get_rod_length(support_rod_id)
            margin = 0.04

            komo.addObjective(
                [t_support, t_pickup],
                ry.FS.positionRel,
                [support_gripper, support_rod],
                ry.OT.ineq,
                1e2 * np.array([[0.0, 0.0, 1.0]]),
                [0.0, 0.0, 0.5 * length - margin],
            )

            komo.addObjective(
                [t_support, t_pickup],
                ry.FS.positionRel,
                [support_gripper, support_rod],
                ry.OT.ineq,
                -1e2 * np.array([[0.0, 0.0, 1.0]]),
                [0.0, 0.0, -0.5 * length + margin],
            )

            # komo.addObjective(
            #     [t_support, t_pickup],
            #     ry.FS.scalarProductXZ,
            #     [support_gripper, support_rod],
            #     ry.OT.eq,
            #     [1e1],
            #     [-1.0],
            # )

            # Support robot holds the affected rod.
            komo.addModeSwitch(
                [t_support, t_pickup],
                ry.SY.stable,
                [support_gripper, support_rod],
                True,
            )

            # The affected rod must remain in its installed scaffold pose.
            # Otherwise KOMO can move the support rod together with the support robot.
            komo.addObjective(
                [t_support, t_pickup],
                ry.FS.positionDiff,
                [support_rod, support_target],
                ry.OT.eq,
                [1e1],
            )

            komo.addObjective(
                [t_support, t_pickup],
                ry.FS.quaternionDiff,
                [support_rod, support_target],
                ry.OT.eq,
                [1e1],
            )

        # ------------------------------------------------------------
        # Candidate rod moves to pickup/staging pose only at final phase.
        # ------------------------------------------------------------

        komo.addObjective(
            [t_pickup],
            ry.FS.positionDiff,
            [rod, pickup_name],
            ry.OT.eq,
            [1e1],
        )

        komo.addObjective(
            [t_pickup],
            ry.FS.scalarProductZZ,
            [rod, pickup_name],
            ry.OT.eq,
            [1e1],
            [1.0],
        )
        
        # The main base circles the midpoint between its two grasp targets.
        base_target_positions = {
            "husky_base_XYPhi_joint": 0.5 * (
                np.asarray(
                    self.C.getFrame(g1).getPosition(),
                    dtype=float,
                )
                + np.asarray(
                    self.C.getFrame(g2).getPosition(),
                    dtype=float,
                )
            )
        }

        # Every newly deployed support robot circles its support-grasp point.
        for support_gripper, support_grasp in (
            support_grasp_by_gripper.items()
        ):
            base_joint = self._base_joint_for_gripper(
                support_gripper
            )

            base_target_positions[base_joint] = np.asarray(
                self.C.getFrame(
                    support_grasp
                ).getPosition(),
                dtype=float,
            )
    
        candidate_rotation = (
            self._frame_transform(rod)[:3, :3].copy()
        )

        ik_targets = {
            "a1": {
                "position": np.asarray(
                    self.C.getFrame(g1).getPosition(),
                    dtype=float,
                ).copy(),
                "rod_rotation": candidate_rotation,
                "alignment": 1.0,
                "roll_group": "main_candidate",
            },
            "a2": {
                "position": np.asarray(
                    self.C.getFrame(g2).getPosition(),
                    dtype=float,
                ).copy(),
                "rod_rotation": candidate_rotation,
                "alignment": 1.0,
                "roll_group": "main_candidate",
            },
        }

        for support_gripper, support_grasp in (
            support_grasp_by_gripper.items()
        ):
            arm_name = support_gripper.removesuffix(
                "_ur_gripper_center"
            )

            support_rod = (
                support_rod_frame_by_gripper[
                    support_gripper
                ]
            )

            ik_targets[arm_name] = {
                "position": np.asarray(
                    self.C.getFrame(
                        support_grasp
                    ).getPosition(),
                    dtype=float,
                ).copy(),
                "rod_rotation": (
                    self._frame_transform(
                        support_rod
                    )[:3, :3].copy()
                ),
                "alignment": -1.0,
                "roll_group": arm_name,
            }

        keyframes = self.solve_komo(
            komo,
            view=False,
            base_target_positions=base_target_positions,
            ik_targets=ik_targets,
            circle_samples=8,
            base_circle_radius=0.6,
            n_phases=phases.n_phases,
            activation_segment_by_arm=(activation_segment_by_arm),
        )

        if keyframes is None:
            # Keep returned state explicit and branch-local.
            failed_supported = {}
            failed_supported.update(continuing_supports)
            failed_supported.update(new_support_assignments)
            return None, q0, failed_supported, phase_info

        # ------------------------------------------------------------
        # Update support state
        # ------------------------------------------------------------

        new_supported = {}

        # Keep support robots that were already supporting other rods.
        new_supported.update(continuing_supports)

        # Add affected rods that are now newly supported.
        new_supported.update(new_support_assignments)

        # Do not copy releasable_supports:
        # those were supporting the candidate rod, which has now been removed.

        return keyframes, q0, new_supported, phase_info