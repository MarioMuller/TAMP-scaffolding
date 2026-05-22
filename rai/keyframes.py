# Stuff related to finding the keyframes

import numpy as np
import robotic as ry
import time


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
       
    def get_keyframes_dual(
        self,
        rod_id,
        d1_from_end=0.12,
        d12_between_arms=0.8,
        theta=0.0,
    ):
        goal_center, goal_quat = self.rods.get_goal_pose(rod_id)

        rod = f"rod_{rod_id}"

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, "world")

        self.C.getFrame(target_name).setPosition(goal_center)
        self.C.getFrame(target_name).setQuaternion(goal_quat)

        g1, g2 = self.rods.create_dual_arm_grasp_frames(
            rod_id,
            d1_from_end=d1_from_end,
            d12_between_arms=d12_between_arms,
        )

        q0 = self.C.getJointState()

        komo = ry.KOMO(
            self.C,
            phases=3,
            slicesPerPhase=1,
            kOrder=1,
            enableCollisions=True,
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

        print("this is actually called!")

        # Phase 1: both arms grasp rod
        # both grippers touch their respective rod positions
        komo.addObjective([1.0, 2.0], ry.FS.positionDiff,
                        ["a1_ur_gripper_center", g1],
                        ry.OT.eq, [1e2])

        komo.addObjective([1.0, 2.0], ry.FS.positionDiff,
                        ["a2_ur_gripper_center", g2],
                        ry.OT.eq, [1e2])

        # both grippers are parallel to the rod
        komo.addObjective([1.0, 2.0], ry.FS.scalarProductXZ,
                        ["a1_ur_gripper_center", rod],
                        ry.OT.eq, [1e1], [1.0])

        komo.addObjective([1.0, 2.0], ry.FS.scalarProductXZ,
                        ["a2_ur_gripper_center", rod],
                        ry.OT.eq, [1e1], [1.0])

        # same rotational angle around the rod
        komo.addObjective([1.0], ry.FS.scalarProductYY,
                        ["a1_ur_gripper_center", "a2_ur_gripper_center"],
                        ry.OT.eq, [1e1], [1.0])

        komo.addObjective([1.0], ry.FS.scalarProductZZ,
                        ["a1_ur_gripper_center", "a2_ur_gripper_center"],
                        ry.OT.eq, [1e1], [1.0])

        # RAI mode switch gives rod one parent
        # Arm 2 is constrained at the keyframe, but not attached
        komo.addModeSwitch(
            [1.0, 2.0],
            ry.SY.stable,
            ["a1_ur_gripper_center", rod],
            True,
        )
        
        # # betweeb 1 and 2 the rod needs to be carried by both
        # # Arm 1 grasp point: fixed distance from rod end
        # komo.addObjective(
        #     [1.0, 2.0],
        #     ry.FS.positionDiff,
        #     ["a1_ur_gripper_center", g1],
        #     ry.OT.eq,
        #     [1e2],
        # )

        # # Arm 2 grasp point: fixed distance from arm 1 along rod
        # komo.addObjective(
        #     [1.0, 2.0],
        #     ry.FS.positionDiff,
        #     ["a2_ur_gripper_center", g2],
        #     ry.OT.eq,
        #     [1e2],
        # )

        # Arm 1 gripper x-axis parallel to rod z-axis
        komo.addObjective(
            [1.0, 2.0],
            ry.FS.scalarProductXZ,
            ["a1_ur_gripper_center", rod],
            ry.OT.eq,
            [1e1],
            [1.0],
        )

        # Arm 2 gripper x-axis parallel to rod z-axis
        komo.addObjective(
            [1.0, 2.0],
            ry.FS.scalarProductXZ,
            ["a2_ur_gripper_center", rod],
            ry.OT.eq,
            [1e1],
            [1.0],
        )

        # Same rotational angle around the rod:
        # gripper y-axes parallel
        komo.addObjective(
            [1.0, 2.0],
            ry.FS.scalarProductYY,
            ["a1_ur_gripper_center", "a2_ur_gripper_center"],
            ry.OT.eq,
            [1e1],
            [1.0],
        )

        # Same rotational angle around the rod:
        # gripper z-axes parallel
        komo.addObjective(
            [1.0, 2.0],
            ry.FS.scalarProductZZ,
            ["a1_ur_gripper_center", "a2_ur_gripper_center"],
            ry.OT.eq,
            [1e1],
            [1.0],
        )

        # Rod is kinematically attached to arm 1 
        komo.addModeSwitch(
            [1.0, 2.0],
            ry.SY.stable,
            ["a1_ur_gripper_center", rod],
            True,
        )

        # placement position
        komo.addObjective(
            [2.0],
            ry.FS.positionDiff,
            [rod, target_name],
            ry.OT.eq,
            [1e2],
        )

        komo.addObjective(
            [2.0],
            ry.FS.scalarProductZZ,
            [rod, target_name],
            ry.OT.eq,
            [1e2],
            [1.0],
        )

        komo.addObjective(
            [2.0],
            ry.FS.scalarProductXX,
            [rod, target_name],
            ry.OT.eq,
            [1e2],
            [1.0],
        )

        # Rod becomes attached to table after placement
        komo.addModeSwitch(
            [2.0, 3.0],
            ry.SY.stable,
            ["table", rod],
            True,
        )

        # back to start
        komo.addObjective(
            [3.0, -1],
            ry.FS.jointState,
            [],
            ry.OT.eq,
            [1e0],
            q0,
        )

        keyframes = self.solve_komo(komo, view=False)

        if keyframes is None:
            raise RuntimeError("KOMO failed to find dual-arm keyframes")
        
        # for t in range(keyframes.shape[0]):
        #     if t == 1:
        #         self.C.attach('a1_ur_gripper_center', f'rod_{rod_id}')
            
        #     elif t == 2:  
        #         self.C.attach('table', f'rod_{rod_id}')

        #     self.C.setJointState(keyframes[t])
        #     self.C.view(False, f'place waypoint {t}')
        #     time.sleep(20)

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

                komo.addObjective(
                    [1.0],
                    ry.FS.qItself,
                    [],
                    ry.OT.eq,
                    scale,
                    q0_full,
                )

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
    
    
    def get_remove_keyframes_dual(
        self,
        rod_id,
        supported=None,
        candidate_is_supported=False,
        old_support_gripper=None,
    ):
        """
        Backward removal:
        - main robot goes to installed rod
        - main robot grasps rod
        - if rod was supported, support releases/moves away
        - rod is moved to pickup/staging pose
        """

        rod = f"rod_{rod_id}"
        q0 = self.C.getJointState()

        pickup_name = f"rod_{rod_id}_pickup_target"
        if self.C.getFrame(pickup_name) is None:
            self.C.addFrame(pickup_name, "world")

        self.C.getFrame(pickup_name) \
            .setPosition([-3, -1, 1.0]) \
            .setQuaternion([0.5, 0.0, 0.5, 0.70710678])

        g1, g2 = self.rods.create_dual_arm_grasp_frames(
            rod_id,
            d1_from_end=0.12,
            d12_between_arms=0.8,
        )

        komo = ry.KOMO(
            self.C,
            phases=2,
            slicesPerPhase=1,
            kOrder=1,
            enableCollisions=True,
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])

        # phase 1: main robot reaches rod in final pos
        komo.addObjective([1.0, 2.0], ry.FS.positionDiff,
                        ["a1_ur_gripper_center", g1],
                        ry.OT.eq, [1e2])

        komo.addObjective([1.0, 2.0], ry.FS.positionDiff,
                        ["a2_ur_gripper_center", g2],
                        ry.OT.eq, [1e2])

        komo.addObjective([1.0, 2.0], ry.FS.scalarProductXZ,
                        ["a1_ur_gripper_center", rod],
                        ry.OT.eq, [1e2], [1.0])

        komo.addObjective([1.0], ry.FS.scalarProductXZ,
                        ["a2_ur_gripper_center", rod],
                        ry.OT.eq, [1e2], [1.0])

        # phase 1-3: main robot carries rod
        komo.addModeSwitch(
            [1.0, 2.0],
            ry.SY.stable,
            ["a1_ur_gripper_center", rod],
            True,
        )

        # # phase 2: optional support robot moves away if it was supporting candidate
        # if candidate_is_supported and old_support_gripper is not None:
        #     # simple safe position frame
        #     safe_name = f"{old_support_gripper}_safe"
        #     if self.C.getFrame(safe_name) is None:
        #         self.C.addFrame(safe_name, "world").setPosition([1.5, 1.5, 1.0])

        #     komo.addObjective([2.0], ry.FS.positionDiff,
        #                     [old_support_gripper, safe_name],
        #                     ry.OT.eq, [1e1])

        # phase 3: removed rod goes to pickup pose
        komo.addObjective([2.0], ry.FS.positionDiff,
                        [rod, pickup_name],
                        ry.OT.eq, [1e2])

        komo.addObjective([2.0], ry.FS.scalarProductZZ,
                        [rod, pickup_name],
                        ry.OT.eq, [1e2], [1.0])

        keyframes = self.solve_komo(komo, attempts=50, view=False)

        if keyframes is None:
            return None, q0, supported

        # update branch-local support state
        new_supported = dict(supported or {})
        if candidate_is_supported and old_support_gripper is not None:
            new_supported.pop(old_support_gripper, None)

        return keyframes, q0, new_supported
    
    def get_remove_keyframes_with_support(
        self,
        rod_id,
        supported=None,
        candidate_is_supported=False,
        old_support_gripper=None,
        support_gripper="h2_a1_ur_gripper_center",
    ):
        """
        Combined support + removal KOMO.

        First simple version:
        - support robot moves to a support grasp on a remaining scaffold rod
        - main robot grasps candidate rod
        - candidate rod is moved to pickup pose
        - support robot stays fixed
        """

        supported = dict(supported or {})

        # If candidate was supported, that support robot becomes available.
        if candidate_is_supported and old_support_gripper is not None:
            supported.pop(old_support_gripper, None)
            support_gripper = old_support_gripper

        # If support robot is already busy, fail for now.
        if support_gripper in supported:
            return None, self.C.getJointState(), supported

        # TODO: choose the actually unstable rod/component.
        # For now, support any remaining rod close to candidate or simply the first non-candidate rod.
        support_rod_id = self.choose_support_rod_after_removal(rod_id)

        if support_rod_id is None:
            return None, self.C.getJointState(), supported

        rod = f"rod_{rod_id}"
        support_rod = f"rod_{support_rod_id}"

        q0 = self.C.getJointState()

        pickup_name = f"rod_{rod_id}_pickup_target"
        if self.C.getFrame(pickup_name) is None:
            self.C.addFrame(pickup_name, "world")

        self.C.getFrame(pickup_name) \
            .setPosition([-3, -1, 1.0]) \
            .setQuaternion([0.5, 0.0, 0.5, 0.70710678])

        g1, g2 = self.rods.create_dual_arm_grasp_frames(
            rod_id,
            d1_from_end=0.12,
            d12_between_arms=0.8,
        )

        support_grasp = self.rods.create_support_grasp_frame_at_fraction(
            support_rod_id,
            0.5,
        )

        komo = ry.KOMO(
            self.C,
            phases=3,
            slicesPerPhase=1,
            kOrder=1,
            enableCollisions=True,
        )

        komo.addControlObjective([], 0, 1e-1)
        komo.addControlObjective([], 1, 1e-1)
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])

        # phase 1: main robot reaches candidate rod
        komo.addObjective([1.0], ry.FS.positionDiff,
                        ["a1_ur_gripper_center", g1],
                        ry.OT.eq, [1e2])

        komo.addObjective([1.0], ry.FS.positionDiff,
                        ["a2_ur_gripper_center", g2],
                        ry.OT.eq, [1e2])

        # phase 1: support robot reaches support point
        komo.addObjective([1.0], ry.FS.positionDiff,
                        [support_gripper, support_grasp],
                        ry.OT.eq, [1e2])

        komo.addObjective([1.0], ry.FS.scalarProductXZ,
                        [support_gripper, support_rod],
                        ry.OT.eq, [1e1], [-1.0])

        # phase 1-3: main robot carries candidate rod
        komo.addModeSwitch(
            [1.0, 3.0],
            ry.SY.stable,
            ["a1_ur_gripper_center", rod],
            True,
        )

        # phase 1-3: support robot stays at support point
        komo.addObjective([1.0, 3.0], ry.FS.positionDiff,
                        [support_gripper, support_grasp],
                        ry.OT.eq, [1e2])

        # phase 3: candidate rod to pickup
        komo.addObjective([3.0], ry.FS.positionDiff,
                        [rod, pickup_name],
                        ry.OT.eq, [1e2])

        komo.addObjective([3.0], ry.FS.scalarProductZZ,
                        [rod, pickup_name],
                        ry.OT.eq, [1e2], [1.0])

        keyframes = self.solve_komo(komo, attempts=50, view=False)

        if keyframes is None:
            return None, q0, supported

        new_supported = dict(supported)
        new_supported[support_gripper] = support_rod_id

        return keyframes, q0, new_supported
        
        
    def choose_support_rod_after_removal(self, removed_rod_id):
        """
        Temporary heuristic.
        Later: choose rod from the unstable component.
        """
        for frame in self.C.getFrames():
            name = frame.name
            if name.startswith("rod_"):
                try:
                    rid = int(name.split("_")[1])
                except Exception:
                    continue

                if rid != removed_rod_id:
                    return rid

        return None