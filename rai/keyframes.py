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
    def solve_komo(self, komo, attempts = 1000, mult = 3, offset = -1.5, view = False, view_accepted = False): 
        for attempt in range(attempts):
        
            if attempt > 0:
                dim = len(self.C.getJointState())
                x_init = np.random.rand(dim) * mult + offset
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=0)

            retval = solver.solve()
            retval = retval.dict()

            print(retval)

            if view:
                print(retval)
                komo.view(True, "IK solution")


            if retval["feasible"]: #retval["ineq"] < 1 and retval["eq"] < 1 and 
                
                if view_accepted:
                    komo.view(True, "IK solution")
                
                keyframes = komo.getPath()
                return keyframes
        
        print("FAILED to find solution")
        
        return None
 
    def get_remove_keyframes_dual(
        self,
        rod_id,
        supported=None,
        support_q=None,
        candidate_is_supported=False,
        old_support_gripper=None,
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

        new_support_assignments:
            support_gripper -> affected_rod_id that must be newly supported
            before removing rod_id
        """

        rod = f"rod_{rod_id}"
        q0 = self.C.getJointState().copy()
        

        supported = dict(supported or {})
        support_q = dict(support_q or {})
        new_support_assignments = dict(new_support_assignments or {})
        old_support_is_reused = (
            old_support_gripper is not None
            and old_support_gripper in new_support_assignments
        )

        # ------------------------------------------------------------
        # Helper target frames
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

        # ------------------------------------------------------------
        # Phase schedule
        # ------------------------------------------------------------

        phases = PhaseSchedule()

        t_grasp = phases.add("main_grasp")

        t_old_support_away = None
        old_support_safe_name = None

        if (
            candidate_is_supported
            and old_support_gripper is not None
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
        # Freeze support robots that are already holding rods and are
        # not being released/reused in this removal.
        # ------------------------------------------------------------
        
        # If the candidate is currently supported, keep that support robot fixed
        # until the main robot reaches/grabs the candidate.
        if candidate_is_supported and old_support_gripper is not None:
            q_fixed = np.asarray(
                support_q.get(old_support_gripper, q0),
                dtype=float,
            )

            # komo.addObjective(
            #     [t_grasp],
            #     ry.FS.qItself,
            #     [],
            #     ry.OT.eq,
            #     [1e2],
            #     q_fixed,
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

            # komo.addObjective(
            #     [t_support],
            #     ry.FS.scalarProductZZ,
            #     [rod, candidate_hold_target],
            #     ry.OT.eq,
            #     [1e2],
            #     [1.0],
            # )

            # komo.addObjective(
            #     [t_support],
            #     ry.FS.scalarProductXX,
            #     [rod, candidate_hold_target],
            #     ry.OT.eq,
            #     [1e2],
            #     [1.0],
            # )
            
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

            # Create a fixed target frame at the current installed pose of the support rod.
            support_target = f"rod_{support_rod_id}_support_target"

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

            # IMPORTANT:
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
                ry.FS.scalarProductZZ,
                [support_rod, support_target],
                ry.OT.eq,
                [1e2],
                [1.0],
            )

            komo.addObjective(
                [t_support, t_pickup],
                ry.FS.scalarProductXX,
                [support_rod, support_target],
                ry.OT.eq,
                [1e2],
                [1.0],
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
            view=True,
        )

        if keyframes is None:
            return None, q0, supported, phase_info

        # ------------------------------------------------------------
        # Update support state
        # ------------------------------------------------------------

        new_supported = dict(supported)

        # Candidate support is released after main takes over.
        if candidate_is_supported and old_support_gripper is not None:
            new_supported.pop(old_support_gripper, None)

        # Affected rods are now supported.
        for support_gripper, support_rod_id in new_support_assignments.items():
            new_supported[support_gripper] = support_rod_id

        return keyframes, q0, new_supported, phase_info   
        
        
    def get_keyframes(self, rod_id):
        
        goal_center, goal_quat = self.rods.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, 'world')

        self.C.getFrame(target_name).setPosition(goal_center)
        self.C.getFrame(target_name).setQuaternion(goal_quat)
        
        orientations = [1.0]
        
        q0 = self.C.getJointState()
        
        for orientation in orientations:
            komo = ry.KOMO(self.C, phases=2, slicesPerPhase=1, kOrder=1, enableCollisions=True)

            komo.addControlObjective([], 0, 1e-1) 
            komo.addControlObjective([], 1, 1e-1)
            # komo.addControlObjective([], 2, 1e-1)
            
            # enable collisions and respect JointLimits
            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
            
            # TODO: change constraint to allow for flexibility when deciding on grabbing position. e.g. using inequality conctraints
            komo.addObjective([1.], ry.FS.positionDiff, ['a1_ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1]) 
            # Gripper fingers are parallel to the rod center axis
            komo.addObjective([1.], ry.FS.scalarProductXZ, ['a1_ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [orientation])
            komo.addModeSwitch([1,2], ry.SY.stable, ['a1_ur_gripper_center', f"rod_{rod_id}"], True)


            # place the end effector in desired final position
            komo.addObjective([2.], ry.FS.positionDiff,
                  [f"rod_{rod_id}", target_name],
                  ry.OT.eq, [1e2])

            komo.addObjective([2.], ry.FS.scalarProductZZ,
                  [f"rod_{rod_id}", target_name],
                  ry.OT.eq, [1e2], [1.0])
            komo.addModeSwitch([2,3], ry.SY.stable, ['table', f"rod_{rod_id}"], True)

            
            # # move back to starting position
            # komo.addObjective([3., -1], ry.FS.jointState, [], ry.OT.eq, [1e0], q0)
            
            keyframes = (self.solve_komo(komo, view=False))
            

        # for t in range(keyframes.shape[0]):
        #     if t == 1:
        #         self.C.attach('a1_ur_gripper_center', f'rod_{rod_id}')
            
        #     elif t == 2:  
        #         self.C.attach('table', f'rod_{rod_id}')

        #     self.C.setJointState(keyframes[t])
        #     self.C.view(False, f'place waypoint {t}')
        #     time.sleep(5)
            
        return keyframes, q0
 
    
    
    def get_support_keyframes(
        self,
        rod_id,
        support_gripper="h2_a1_ur_gripper_center",
        main_gripper="a1_ur_gripper_center",
        grasp_fractions=(0.75, 0.5, 0.25),
        freeze_main=False,
        keep_rod_at_target=True,
    ):
        """
        Finds a keyframe where the support robot grasps the rod somewhere along it.

        Sequential assumption:
        - The main robot is already holding the rod at the target.
        - While the support robot moves, the main gripper is frozen.
        - The rod is kept at its target pose.
        """

        rod = f"rod_{rod_id}"
        q0 = self.C.getJointState()
        
        main_joint_names = [
            "husky_base_XYPhi_joint:0",
            "husky_base_XYPhi_joint:1",
            "husky_base_XYPhi_joint:2",
            "a1_shoulder_pan_joint",
            "a1_shoulder_lift_joint",
            "a1_elbow_joint",
            "a1_wrist_1_joint",
            "a1_wrist_2_joint",
            "a1_wrist_3_joint",
            "a2_shoulder_pan_joint",
            "a2_shoulder_lift_joint",
            "a2_elbow_joint",
            "a2_wrist_1_joint",
            "a2_wrist_2_joint",
            "a2_wrist_3_joint",
        ]
        
        main_q0 = q0[:15].copy()

        target_name = self.rods.create_target_frame(rod_id)

        for fraction in grasp_fractions:
            support_grasp = self.rods.create_support_grasp_frame_at_fraction(
                rod_id,
                fraction,
            )

            print(
                f"Trying support grasp for rod {rod_id} "
                f"at fraction {fraction}"
            )

            komo = ry.KOMO(
                self.C,
                phases=1,
                slicesPerPhase=1,
                kOrder=1,
                enableCollisions=True,
            )

            komo.addControlObjective([], 0, 1e-1)

            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
            
            if freeze_main:
                print("Freeze main robot is active")

                q0_full = np.asarray(q0, dtype=float)

                scale = np.zeros_like(q0_full)
                scale[:15] = 1e2   # freeze main robot only

                # komo.addObjective(
                #     [1.0],
                #     ry.FS.qItself,
                #     [],
                #     ry.OT.eq,
                #     scale,
                #     q0_full,
                # )

            # # Keep rod fixed at target while support robot approaches.
            
            # komo.addObjective(
            #     [1.0],
            #     ry.FS.positionDiff,
            #     [rod, target_name],
            #     ry.OT.eq,
            #     [1e2],
            # )

            # komo.addObjective(
            #     [1.0],
            #     ry.FS.scalarProductZZ,
            #     [rod, target_name],
            #     ry.OT.eq,
            #     [1e2],
            #     [1.0],
            # )

            # komo.addObjective(
            #     [1.0],
            #     ry.FS.scalarProductXX,
            #     [rod, target_name],
            #     ry.OT.eq,
            #     [1e2],
            #     [1.0],
            # )

            # Support gripper touches one candidate point along the rod.
            komo.addObjective(
                [1.0],
                ry.FS.positionDiff,
                [support_gripper, support_grasp],
                ry.OT.eq,
                [1e2],
            )

            # Support gripper perpendicular to rod axis.
            # If this is the wrong axis for your UR gripper, try scalarProductZZ or scalarProductYZ.
            komo.addObjective(
                [1.0],
                ry.FS.scalarProductXZ,
                [support_gripper, rod],
                ry.OT.eq,
                [1e1],
                [-1.0],
            )

            keyframes = self.solve_komo(
                komo,
                attempts=50,
                view_accepted=True,
            )

            if keyframes is not None:
                return keyframes, q0

        raise RuntimeError(f"Support keyframe failed for rod {rod_id}")
    
    def get_keyframes_with_optional_support(
        self,
        rod_id,
        support_assignments=None,
        main_gripper="a1_ur_gripper_center",
        scaffold_parent="table",
        support_fraction=0.5,
    ):
        """
        One-KOMO version.

        support_assignments:
            dict mapping support_gripper -> supported_rod_id

            Example:
            {
                "h1_ur_gripper_center": 12,
                "h2_ur_gripper_center": 17,
            }

        If support_assignments is empty, this behaves like a normal single-robot
        pick/place KOMO with fewer phases.
        """

        if support_assignments is None:
            support_assignments = {}

        rod = f"rod_{rod_id}"
        q0 = self.C.getJointState()

        target_name = self.rods.create_target_frame(rod_id)

        # ------------------------------------------------------------
        # 1. Build variable phase schedule
        # ------------------------------------------------------------
        phases = PhaseSchedule()

        t_grasp = phases.add("main_grasp")

        support_phase_names = {}
        for support_gripper in support_assignments:
            phase_name = f"support_{support_gripper}"
            support_phase_names[support_gripper] = phase_name
            phases.add(phase_name)

        t_place = phases.add("place")
        t_final = phases.add("final")

        # ------------------------------------------------------------
        # 2. Create one KOMO with the needed number of phases
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

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

        # ------------------------------------------------------------
        # 3. Main robot grasps candidate rod
        # ------------------------------------------------------------
        komo.addObjective(
            [t_grasp],
            ry.FS.positionDiff,
            [main_gripper, rod],
            ry.OT.eq,
            [1e2],
        )

        komo.addObjective(
            [t_grasp],
            ry.FS.scalarProductXZ,
            [main_gripper, rod],
            ry.OT.eq,
            [1e1],
            [1.0],
        )

        # Main robot holds candidate rod until placement
        komo.addModeSwitch(
            [t_grasp, t_place],
            ry.SY.stable,
            [main_gripper, rod],
            True,
        )

        # ------------------------------------------------------------
        # 4. Optional support robots
        # ------------------------------------------------------------
        for support_gripper, supported_rod_id in support_assignments.items():
            t_support = phases.get(support_phase_names[support_gripper])

            supported_rod = f"rod_{supported_rod_id}"

            support_grasp = self.rods.create_support_grasp_frame_at_fraction(
                supported_rod_id,
                support_fraction,
            )

            komo.addObjective(
                [t_support],
                ry.FS.positionDiff,
                [support_gripper, support_grasp],
                ry.OT.eq,
                [1e2],
            )

            komo.addObjective(
                [t_support],
                ry.FS.scalarProductXZ,
                [support_gripper, supported_rod],
                ry.OT.eq,
                [1e1],
                [-1.0],
            )

            # Support robot holds this already-installed rod until final phase
            komo.addModeSwitch(
                [t_support, t_final],
                ry.SY.stable,
                [support_gripper, supported_rod],
                True,
            )

        # ------------------------------------------------------------
        # 5. Place candidate rod at scaffold target
        # ------------------------------------------------------------
        komo.addObjective(
            [t_place],
            ry.FS.positionDiff,
            [rod, target_name],
            ry.OT.eq,
            [1e2],
        )

        komo.addObjective(
            [t_place],
            ry.FS.scalarProductZZ,
            [rod, target_name],
            ry.OT.eq,
            [1e2],
            [1.0],
        )

        komo.addObjective(
            [t_place],
            ry.FS.scalarProductXX,
            [rod, target_name],
            ry.OT.eq,
            [1e2],
            [1.0],
        )

        # Rod becomes part of scaffold after placement
        komo.addModeSwitch(
            [t_place, t_final],
            ry.SY.stable,
            [scaffold_parent, rod],
            True,
        )

        # Optional: return to start at final phase
        komo.addObjective(
            [t_final, -1],
            ry.FS.jointState,
            [],
            ry.OT.eq,
            [1e0],
            q0,
        )

        keyframes = self.solve_komo(komo, view=False)

        if keyframes is None:
            return None, q0, phases

        return keyframes, q0, phases


