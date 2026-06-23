# Stuff related to finding the keyframes

import numpy as np
import robotic as ry
import time


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
    def __init__(self, C, rod_manager):
        self.C = C
        self.rods = rod_manager

    # based on implementation of vhartman
    def solve_komo(
        self,
        komo,
        attempts=1000,
        mult=3,
        offset=-1.5,
        view=True,
        view_accepted=False,
    ):
        for attempt in range(attempts):

            if attempt > 0:
                dim = len(self.C.getJointState())
                x_init = np.random.rand(dim) * mult + offset
                # print(x_init)
                komo.initWithConstant(x_init)
                

            solver = ry.NLP_Solver(komo.nlp(), verbose=0)

            try:
                retval = solver.solve()
            except RuntimeError as e:
                msg = str(e)
                if "checkNan" in msg or "inconsistent number" in msg:
                    print(f"KOMO attempt {attempt} crashed with NaN; skipping this restart")
                    continue
                raise
            
            retval = retval.dict()

            print(retval)

            if view:
                print(retval)
                komo.view(True, "IK solution")

            if retval["feasible"]:  # retval["ineq"] < 1 and retval["eq"] < 1 and

                if view_accepted:
                    komo.view(False, "IK solution")

                keyframes = komo.getPath()
                return keyframes

        print("FAILED to find solution")

        return None

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
        else:
            target = self.C.getFrame(target_frame_name)

        target.setPosition(source.getPosition())
        target.setQuaternion(source.getQuaternion())

        return target_frame_name

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

        old_support_is_reused = (
            old_support_gripper is not None
            and old_support_gripper in new_support_assignments
        )

        # ------------------------------------------------------------
        # Helper frames
        # ------------------------------------------------------------

        pickup_name = f"rod_{rod_id}_pickup_target"

        if pickup_name not in self.C.getFrameNames():
            self.C.addFrame(pickup_name, "world")

        self.C.getFrame(pickup_name) \
            .setPosition([-3, -1, 1.0]) \
            .setQuaternion([0.5, 0.0, 0.5, 0.70710678])

        candidate_hold_target = f"rod_{rod_id}_hold_target"

        if candidate_hold_target not in self.C.getFrameNames():
            self.C.addFrame(candidate_hold_target, "world")

        self.C.getFrame(candidate_hold_target) \
            .setPosition(self.C.getFrame(rod).getPosition()) \
            .setQuaternion(self.C.getFrame(rod).getQuaternion())

        g1, g2 = self.rods.create_dual_arm_grasp_frames(
            rod_id,
            d1_from_end=0.12,
            d12_between_arms=0.8,
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

            self.C.getFrame(support_target) \
                .setPosition(self.C.getFrame(support_rod).getPosition()) \
                .setQuaternion(self.C.getFrame(support_rod).getQuaternion())

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
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])

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
                [1e2],
            )

            komo.addObjective(
                [t_grasp, t_pickup],
                ry.FS.quaternionDiff,
                [support_gripper, gripper_target],
                ry.OT.eq,
                [1e2],
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
                [1e2],
            )

            komo.addObjective(
                [t_grasp],
                ry.FS.quaternionDiff,
                [support_gripper, gripper_target],
                ry.OT.eq,
                [1e2],
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
        for support_gripper, t_support in support_phase_by_gripper.items():
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

        # ------------------------------------------------------------
        # Main grasps candidate rod and keeps it until pickup.
        # ------------------------------------------------------------

        komo.addObjective(
            [t_grasp, t_pickup],
            ry.FS.positionDiff,
            ["a1_ur_gripper_center", g1],
            ry.OT.eq,
            [1e2],
        )

        komo.addObjective(
            [t_grasp, t_pickup],
            ry.FS.positionDiff,
            ["a2_ur_gripper_center", g2],
            ry.OT.eq,
            [1e2],
        )

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
                [1e2],
            )

            komo.addObjective(
                [t_support, t_pickup],
                ry.FS.quaternionDiff,
                [support_rod, support_target],
                ry.OT.eq,
                [1e2],
            )

        # ------------------------------------------------------------
        # Candidate rod moves to pickup/staging pose only at final phase.
        # ------------------------------------------------------------

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

        keyframes = self.solve_komo(
            komo,
            attempts=50,
            view=False,
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