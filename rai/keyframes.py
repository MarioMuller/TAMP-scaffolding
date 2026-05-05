# Stuff related to finding the keyframes

import numpy as np
import robotic as ry
import time


class KeyframePlanner:
    def __init__(self, C, rod_manager):
        self.C = C
        self.rods = rod_manager
        
        
    # based on implementation of vhartman
    def solve_komo(self, komo, attempts = 1000, mult = 3, offset = -1.5, view = False): 
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
        
        support_fixed_pairs = self.create_fixed_pose_frames_by_prefix(
            prefixes=[
                "h2_base_XYPhi_joint",
                "h2_husky_",
                "h2_a1_",
            ],
            tag="support",
        )
        
        for orientation in orientations:
            komo = ry.KOMO(self.C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)

            komo.addControlObjective([], 0, 1e-1) 
            komo.addControlObjective([], 1, 1e-1)
            # komo.addControlObjective([], 2, 1e-1)
            
            self.add_freeze_frame_constraints(
                komo,
                support_fixed_pairs,
                times=[],
                weight=1e2,
            )
            
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

            
            # move back to starting position
            komo.addObjective([3., -1], ry.FS.jointState, [], ry.OT.eq, [1e0], q0)
            
            keyframes = (self.solve_komo(komo, view=True))
            

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
        
        support_fixed_pairs = self.create_fixed_pose_frames_by_prefix(
            prefixes=[
                "h2_base_XYPhi_joint",
                "h2_husky_",
                "h2_a1_",
            ],
            tag="support",
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
        
        self.add_freeze_frame_constraints(
            komo,
            support_fixed_pairs,
            times=[],
            weight=1e2,
        )

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])


        # Phase 1: both arms grasp rod
        # both grippers touch their respective rod positions
        komo.addObjective([1.0], ry.FS.positionDiff,
                        ["a1_ur_gripper_center", g1],
                        ry.OT.eq, [1e2])

        komo.addObjective([1.0], ry.FS.positionDiff,
                        ["a2_ur_gripper_center", g2],
                        ry.OT.eq, [1e2])

        # both grippers are parallel to the rod
        komo.addObjective([1.0], ry.FS.scalarProductXZ,
                        ["a1_ur_gripper_center", rod],
                        ry.OT.eq, [1e1], [1.0])

        komo.addObjective([1.0], ry.FS.scalarProductXZ,
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
            [1, 2],
            ry.SY.stable,
            ["a1_ur_gripper_center", rod],
            True,
        )
        
        # betweeb 1 and 2 the rod needs to be carried by both
        # Arm 1 grasp point: fixed distance from rod end
        komo.addObjective(
            [1.0, 2.0],
            ry.FS.positionDiff,
            ["a1_ur_gripper_center", g1],
            ry.OT.eq,
            [1e2],
        )

        # Arm 2 grasp point: fixed distance from arm 1 along rod
        komo.addObjective(
            [1.0, 2.0],
            ry.FS.positionDiff,
            ["a2_ur_gripper_center", g2],
            ry.OT.eq,
            [1e2],
        )

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

        keyframes = self.solve_komo(komo)

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
        grasp_fractions=(0.25, 0.5, 0.75),
        freeze_main=True,
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

        target_name = self.rods.create_target_frame(rod_id)

        fixed_main_frame = None
        if freeze_main:
            fixed_main_frame = self.create_fixed_pose_frame(
                main_gripper,
                f"fixed_{main_gripper}_for_support",
            )
            
        main_fixed_pairs = self.create_fixed_pose_frames_by_prefix(
            prefixes=[
                "husky_base_XYPhi_joint",
                "husky_coll_",
                "a1_",
                "a2_",
            ],
            tag="main",
        )

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
            
            self.add_freeze_frame_constraints(
                komo,
                main_fixed_pairs,
                times=[],
                weight=1e2,
            )

            # Freeze main gripper while support robot moves.
            if freeze_main:
                komo.addObjective(
                    [1.0],
                    ry.FS.positionDiff,
                    [main_gripper, fixed_main_frame],
                    ry.OT.eq,
                    [1e2],
                )

                komo.addObjective(
                    [1.0],
                    ry.FS.quaternionDiff,
                    [main_gripper, fixed_main_frame],
                    ry.OT.eq,
                    [1e2],
                )

            # Keep rod fixed at target while support robot approaches.
            if keep_rod_at_target:
                komo.addObjective(
                    [1.0],
                    ry.FS.positionDiff,
                    [rod, target_name],
                    ry.OT.eq,
                    [1e2],
                )

                komo.addObjective(
                    [1.0],
                    ry.FS.scalarProductZZ,
                    [rod, target_name],
                    ry.OT.eq,
                    [1e2],
                    [1.0],
                )

                komo.addObjective(
                    [1.0],
                    ry.FS.scalarProductXX,
                    [rod, target_name],
                    ry.OT.eq,
                    [1e2],
                    [1.0],
                )

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
                view=True,
            )

            if keyframes is not None:
                return keyframes, q0

        raise RuntimeError(f"Support keyframe failed for rod {rod_id}")
    
    
    def create_fixed_pose_frame(self, source_frame, fixed_name):
        if self.C.getFrame(fixed_name) is None:
            self.C.addFrame(fixed_name, "world")

        self.C.getFrame(fixed_name).setPosition(
            self.C.getFrame(source_frame).getPosition()
        )

        self.C.getFrame(fixed_name).setQuaternion(
            self.C.getFrame(source_frame).getQuaternion()
        )

        return fixed_name


    def create_fixed_pose_frames_by_prefix(self, prefixes, tag):
        fixed_pairs = []

        for frame_name in self.C.getFrameNames():
            if not any(frame_name.startswith(prefix) for prefix in prefixes):
                continue

            if frame_name.startswith("fixed_"):
                continue

            safe_name = self.safe_frame_name(frame_name)
            fixed_name = f"fixed_{tag}_{safe_name}"

            self.create_fixed_pose_frame(frame_name, fixed_name)

            if self.C.getFrame(fixed_name) is None:
                print(f"Failed to create fixed frame for {frame_name}")
                continue

            fixed_pairs.append((frame_name, fixed_name))

        print(f"Created {len(fixed_pairs)} fixed frames for {tag}")

        return fixed_pairs


    def add_freeze_frame_constraints(
        self,
        komo,
        fixed_pairs,
        times=None,
        weight=1e2,
    ):
        """
        Constrains all source frames to stay at their fixed world poses.
        Use times=[] to apply over the whole KOMO.
        """

        if times is None:
            times = []

        for source, fixed in fixed_pairs:
            komo.addObjective(
                times,
                ry.FS.positionDiff,
                [source, fixed],
                ry.OT.eq,
                [weight],
            )

            komo.addObjective(
                times,
                ry.FS.quaternionDiff,
                [source, fixed],
                ry.OT.eq,
                [weight],
            ) 
        
        
        
    def safe_frame_name(self, name):
        return (
            name.replace(">", "_")
                .replace("/", "_")
                .replace(":", "_")
                .replace(" ", "_")
                .replace(".", "_")
        )
    
    
    