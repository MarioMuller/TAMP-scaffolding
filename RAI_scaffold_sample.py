from json_import import Truss
import numpy as np
import robotic as ry
import time
import os

class RaiTrussBuilder:

    def __init__(self, truss, radius=0.0015, scale = 0.00351):
        self.truss = truss
        self.radius = radius
        self.C = ry.Config()
        self.C.addFrame("world")
        self.scale = scale 

    def quaternion_from_z_to_vector(self, direction):
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        z = np.array([0.0, 0.0, 1.0])

        # cross product gives rotation axis
        axis = np.cross(z, direction)
        axis_norm = np.linalg.norm(axis)

        # dot product gives cos(angle)
        dot = np.dot(z, direction)

        # handle parallel case
        if axis_norm < 1e-8:
            if dot > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])
            else:
                return np.array([0.0, 1.0, 0.0, 0.0])

        axis = axis / axis_norm
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        w = np.cos(angle / 2.0)
        xyz = axis * np.sin(angle / 2.0)

        return np.concatenate(([w], xyz))

    # only here as a helper to see that the calculations work properly and lead to desired final structure
    def check_end_configuration(self):
        
        for rod_id, (n1, n2) in self.truss.elements.items():

            p1 = np.array(self.truss.nodes[n1]) * self.scale
            p2 = np.array(self.truss.nodes[n2]) * self.scale
        
            length = np.linalg.norm(p2 - p1)
            center = 0.5 * (p1 + p2)
            quat = self.quaternion_from_z_to_vector(p2 - p1)

            self.C.addFrame(f"rod_{rod_id}", 'world') .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setPosition(center) .setQuaternion(quat)
        
        self.C.view()
        input("Press Enter to close...")

        return
    
    # helper function to move rods out of the way 
    def set_to_end_position(self, rod_id):
        
        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1]) * self.scale
        p2 = np.array(self.truss.nodes[n2]) * self.scale
    
        length = np.linalg.norm(p2 - p1)
        center = 0.5 * (p1 + p2)
        center[2] += 0.1
        quat = self.quaternion_from_z_to_vector(p2 - p1)

        self.C.getFrame(f"rod_{rod_id}").setPosition(center) .setQuaternion(quat)
        
        self.C.view()
        # input("Press Enter to close...")

        return
    
    def get_goal_pose(self, rod_id):

        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1]) * self.scale
        p2 = np.array(self.truss.nodes[n2]) * self.scale

        center = 0.5 * (p1 + p2)
        center = center + [0, 0, 0.1]
        quat = self.quaternion_from_z_to_vector(p2 - p1)

        return center, quat

    # creates the next required rod
    def create_rod(self, rod_id, pos = [-0.4,-0.05,0.2], ori = [0.7070, 1, 0, 0.7070]):
        
        ori = np.array(ori, dtype=float)
        ori = ori / np.linalg.norm(ori)


        n1, n2 = self.truss.elements[rod_id]

        p1 = np.array(self.truss.nodes[n1], dtype=float) * self.scale
        p2 = np.array(self.truss.nodes[n2], dtype=float) * self.scale

        length = np.linalg.norm(p2 - p1) -0.03 #-0.03 for long_beam
        if length < 1e-10:
            raise ValueError(f"Rod {rod_id} has zero length")

        self.C.addFrame(f"rod_{rod_id}") .setShape(ry.ST.cylinder, [length, self.radius]) .setColor([.5,1.,.0]) .setPosition(pos) .setQuaternion(ori) .setContact(1)
        self.C.view()
        # input("Press Enter to close...")
        return
    
    def import_ur5(self):
        self.C.addFile("/home/mario/TAMP-scaffolding/src/models/ur5/ur5.g") .setPosition([0.0,0.55,0])
        print(self.C.getFrameNames())

        self.qHome = self.C.getJointState().copy()
        self.C.view()
        return
    
    def import_husky(self):

        # sets ground plane
        table = self.C.addFrame("table").setPosition([0, 0, 0.0]).setShape(
            ry.ST.box, size=[20, 20, 0.02, 0.005]
        ).setColor([0.9, 0.9, 0.9]).setContact(1)

        # paths to the files
        husky_path = os.path.join(os.path.dirname(__file__), "src/models/husky/husky.g")    
        robot_path = os.path.join(os.path.dirname(__file__), "src/models/ur5/ur5.g")

        # Create a movable joint frame to attach the husky to
        # TODO: This should be replaced trhough a kind of accurate representation of a differential drive
        
        # self.C.addFrame("husky_base_XYPhi_joint") .setParent(self.C.getFrame("world")) .setJoint(
        #     ry.JT.phiTransXY, limits=np.array([-3.14, 3.14, -10, 10, -0.0001, 0.0001])
        # ).setJointState([0., 0, 0]) 

        self.C.addFrame("husky_base_XYPhi_joint") .setParent(self.C.getFrame("world")) .setJoint(
            ry.JT.transXYPhi, limits=np.array([-30, 30, -30, 30, -3.14, 3.14])
        ).setJointState([-1., 0, 0]) 

        self.C.addFile(husky_path, namePrefix="husky_coll_").setParent(
            self.C.getFrame("husky_base_XYPhi_joint")
        ).setRelativePosition([0, 0.0, 0.16])

        # attatch both arms to the husky
        self.C.addFile(robot_path, namePrefix="a1_").setParent(
        self.C.getFrame("husky_coll_right_arm_bulkhead_joint")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion([1, 0, 0, 0])
        
        self.C.addFile(robot_path, namePrefix="a2_").setParent(
        self.C.getFrame("husky_coll_left_arm_bulkhead_joint")
            ).setRelativePosition([0, 0, 0]).setRelativeQuaternion([1, 0, 0, 0])
        
        # self.C.getFrame("a1_coll1").setContact(0)
        # self.C.getFrame("a2_coll1").setContact(0)

        # print(self.C.getFrameNames())
        self.C.view()

        return

    
    def pick_and_place_rod(self, rod_id):

        goal_center, goal_quat = self.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, 'world')

        self.C.getFrame(target_name).setPosition(goal_center)
        self.C.getFrame(target_name).setQuaternion(goal_quat)
        
        komo = ry.KOMO(self.C, phases=3, slicesPerPhase=10, kOrder=2, enableCollisions=True)

        komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
        komo.addControlObjective([], 2, 1e0)

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e2])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

        # grab the rod in spawning position
        komo.addObjective([1.], ry.FS.positionDiff, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1]) # an equality constraint on the 3D position difference between ur_gripper_center and box
        komo.addObjective([1.], ry.FS.scalarProductXZ, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [1.0])
        komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
        komo.addModeSwitch([1.,-1], ry.SY.stable, ['ur_gripper_center', f"rod_{rod_id}"], True)

        goal_center_up = goal_center.copy()
        goal_center_up[2] += 0.10   # move 20 cm up in world z
        komo.addObjective([2.], ry.FS.position, [f"rod_{rod_id}"], ry.OT.eq, [1e1], goal_center_up)
        komo.addObjective([3.], ry.FS.scalarProductZZ, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1], [1.0])

        komo.addObjective([3.], ry.FS.positionDiff, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1])
        komo.addObjective([3.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
        komo.addObjective([3.], ry.FS.scalarProductZZ, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1], [1.0])

        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        print(ret)

        if not ret.feasible:

            komo = ry.KOMO(self.C, phases=3, slicesPerPhase=10, kOrder=2, enableCollisions=True)

            komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
            komo.addControlObjective([], 2, 1e0)

            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e2])
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

            # grab the rod in spawning position
            komo.addObjective([1.], ry.FS.positionDiff, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1]) # an equality constraint on the 3D position difference between ur_gripper_center and box
            komo.addObjective([1.], ry.FS.scalarProductXZ, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [1.0])
            komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
            komo.addModeSwitch([1.,-1], ry.SY.stable, ['ur_gripper_center', f"rod_{rod_id}"], True)

            goal_center_up = goal_center.copy()
            goal_center_up[2] += 0.10   # move 20 cm up in world z
            komo.addObjective([2.], ry.FS.position, [f"rod_{rod_id}"], ry.OT.eq, [1e1], goal_center_up)
            komo.addObjective([3.], ry.FS.scalarProductZZ, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1], [-1.0])

            komo.addObjective([3.], ry.FS.positionDiff, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1])
            komo.addObjective([3.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
            komo.addObjective([3.], ry.FS.scalarProductZZ, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1], [-1.0])

            ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
            print(ret)


            if not ret.feasible:

                komo.view(True, "IK solution")
                raise RuntimeError(f"Pick & PLace not possible for rod {rod_id}")

        # komo.view(True, "IK solution")
        q = komo.getPath()

        print('size of path:', q.shape)

        for t in range(q.shape[0]):
            if t == 10:
                self.C.attach('ur_gripper_center', f'rod_{rod_id}')

            self.C.setJointState(q[t])
            self.C.view(False, f'place waypoint {t}')
            time.sleep(.1)

        self.C.attach('world', f'rod_{rod_id}')

        return   
    
    def pick_and_place_husky(self, rod_id):

        goal_center, goal_quat = self.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, 'world')

        self.C.getFrame(target_name).setPosition(goal_center)
        self.C.getFrame(target_name).setQuaternion(goal_quat)
        
        orientations = [1.0, -1.0]
        
        for orientation in orientations:
        
            komo = ry.KOMO(self.C, phases=3, slicesPerPhase=10, kOrder=2, enableCollisions=True)

            komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
            komo.addControlObjective([], 2, 1e0)

            # enable collisions and respect JointLimits
            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e2])
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

            # grab the rod in the center
            # TODO: change constraint to allow for flexibility when deciding on grabbing position. e.g. using inequality conctraints
            komo.addObjective([1.], ry.FS.positionDiff, ['a1_ur_gripper_center', f"rod_{rod_id}"], ry.OT.sos, [1e0]) 
            komo.addObjective([1.], ry.FS.distance, ['a1_ur_gripper_center', f"rod_{rod_id}"], ry.OT.sos, [1e1], [-0.0])
        
            # Gripper fingers are parallel to the rod center axis
            # komo.addObjective([1.], ry.FS.scalarProductXZ, ['a1_ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [orientation])
            # ensure the motions stops at pickup time
            # komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
            # attach the rod to the gripper
            komo.addModeSwitch([1.,-1], ry.SY.stable, ['a1_ur_gripper_center', f"rod_{rod_id}"], True)
            
            
            goal_center_up = goal_center.copy()
            goal_center_up[2] += 0.10   # move 20 cm up in world z
            
            komo.addObjective([2.], ry.FS.position, [f"rod_{rod_id}"], ry.OT.sos, [1e1], goal_center_up)
            # komo.addObjective([1.], ry.FS.distance, [f"rod_{rod_id}", goal_center_up], ry.OT.sos, [1e1], [-0.0])
            # komo.addObjective([2.], ry.FS.scalarProductZZ, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1], [1.0])

            komo.addObjective([3.], ry.FS.positionDiff, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1])
            # komo.addObjective([1.], ry.FS.distance, [f"rod_{rod_id}", goal_center], ry.OT.sos, [1e1], [-0.0])
            # komo.addObjective([3.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
            # komo.addObjective([3.], ry.FS.scalarProductZZ, [f"rod_{rod_id}", target_name], ry.OT.eq, [1e1], [1.0])

            ret = ry.NLP_Solver(komo.nlp(), verbose=4).solve()
            
            # for i in range(1000):
            #     komo.initRandom()   # randomize trajectory initialization
                
            #     ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
            #     print(ret)

            #     if ret.feasible:
            #         break
                
            print(ret)
            
            if ret.feasible:
                break

        if not ret.feasible:

            komo.view(True, "IK solution")
            raise RuntimeError(f"Pick & PLace not possible for rod {rod_id}")

        # komo.view(True, "IK solution")
        q = komo.getPath()

        print('size of path:', q.shape)

        for t in range(q.shape[0]):
            if t == 1:
                self.C.attach('a1_ur_gripper_center', f'rod_{rod_id}')

            self.C.setJointState(q[t])
            self.C.view(False, f'place waypoint {t}')
            time.sleep(.2)

        self.C.attach('world', f'rod_{rod_id}')

    def pick_and_place_using_keyframes(self, rod_id):
        """
        This functions builds upon previously found joint constellations for the placement and pickup locations

        Args:
            q_pickup: joint configuration of the pickup location
            q_placement: joint configuration of the placement location
        """
        
        solutions_pickup = self.q_pickup(rod_id)
        solutions_place = self.q_place(rod_id)
        
        for q_pickup in solutions_pickup:
            
           for q_place in solutions_place:
        
                komo = ry.KOMO(self.C, phases=2, slicesPerPhase=10, kOrder=2, enableCollisions=True)

                komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
                komo.addControlObjective([], 2, 1e0)

                # enable collisions and respect JointLimits
                komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
                komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

                # grab the rod in the center
                # TODO: change constraint to allow for flexibility when deciding on grabbing position. e.g. using inequality conctraints
                komo.addObjective([1.], ry.FS.jointState, [], ry.OT.eq, [1e1], q_pickup)
                # ensure the motions stops at pickup time
                komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)
                # attach the rod to the gripper
                komo.addModeSwitch([1.,-1], ry.SY.stable, ['a1_ur_gripper_center', f"rod_{rod_id}"], True)
                

                komo.addObjective([2.], ry.FS.jointState, [], ry.OT.eq, [1e0], q_place)
                komo.addObjective([2.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)

                ret = ry.NLP_Solver(komo.nlp(), verbose=4).solve()
                    
                print(ret)
                
                if ret.feasible:
                    break

        if not ret.feasible:

            komo.view(True, "IK solution")
            raise RuntimeError(f"Pick & PLace not possible for rod {rod_id}")

        # komo.view(True, "IK solution")
        q = komo.getPath()

        print('size of path:', q.shape)

        for t in range(q.shape[0]):
            if t == 10:
                self.C.attach('a1_ur_gripper_center', f'rod_{rod_id}')

            self.C.setJointState(q[t])
            self.C.view(False, f'place waypoint {t}')
            time.sleep(.2)

        self.C.attach('world', f'rod_{rod_id}')
        
        
        return
    
    def q_pickup(self, rod_id):
        
        orientations = [1.0, -1.0]
        solutions = []
        
        for orientation in orientations:
            komo = ry.KOMO(self.C, phases=1, slicesPerPhase=1, kOrder=1, enableCollisions=True)

            komo.addControlObjective([], 0, 1e-1) 
            
            # enable collisions and respect JointLimits
            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e2])
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

            # grab the rod in the center
            # TODO: change constraint to allow for flexibility when deciding on grabbing position. e.g. using inequality conctraints
            komo.addObjective([1.], ry.FS.positionDiff, ['a1_ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1]) 
            # Gripper fingers are parallel to the rod center axis
            komo.addObjective([1.], ry.FS.scalarProductXZ, ['a1_ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [orientation])
            
            ret = ry.NLP_Solver(komo.nlp(), verbose=4).solve()
            
            print(ret)
            if ret.feasible:
                q = komo.getPath()
                solutions.append(q)
            
        if not solutions:
            komo.view(True, "IK solution")
            print("FAILED to find solution")
            
            
        return solutions
    
    def q_place(self, rod_id):
        
        goal_center, goal_quat = self.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, 'world')

        self.C.getFrame(target_name).setPosition(goal_center)
        self.C.getFrame(target_name).setQuaternion(goal_quat)
        
        orientations = [1.0, -1.0]
        solutions = []
        
        for orientation in orientations:
            komo = ry.KOMO(self.C, phases=1, slicesPerPhase=1, kOrder=0, enableCollisions=True)

            komo.addControlObjective([], 0, 1e-1) 
            
            # enable collisions and respect JointLimits
            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e2])
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

            # place the end effector in desired final position
            komo.addObjective([1.], ry.FS.positionDiff, ['a1_ur_gripper_center', target_name], ry.OT.eq, [1e1]) 
            # Gripper fingers are parallel to the rod center axis
            komo.addObjective([1.], ry.FS.scalarProductXZ, ['a1_ur_gripper_center', target_name], ry.OT.eq, [1e1], [orientation])

            ret = ry.NLP_Solver(komo.nlp(), verbose=4).solve()
            
            print(ret)
            if ret.feasible:
                q = komo.getPath()
                solutions.append(q)

            
        if not solutions:
            komo.view(True, "IK solution")
            print("FAILED to find solution")
            
        return solutions
    
    
    def get_keyframes(self, rod_id):
        
        goal_center, goal_quat = self.get_goal_pose(rod_id)

        target_name = f"rod_{rod_id}_target"
        if self.C.getFrame(target_name) is None:
            self.C.addFrame(target_name, 'world')

        self.C.getFrame(target_name).setPosition(goal_center)
        self.C.getFrame(target_name).setQuaternion(goal_quat)
        
        orientations = [1.0]
        
        q0 = self.C.getJointState()
        
        for orientation in orientations:
            komo = ry.KOMO(self.C, phases=3, slicesPerPhase=1, kOrder=1, enableCollisions=True)

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

            
            # move back to starting position
            komo.addObjective([3., -1], ry.FS.jointState, [], ry.OT.eq, [1e0], q0)
            
            keyframes = (self.solve_komo(komo))
            

        # for t in range(keyframes.shape[0]):
        #     if t == 1:
        #         self.C.attach('a1_ur_gripper_center', f'rod_{rod_id}')
            
        #     elif t == 2:  
        #         self.C.attach('table', f'rod_{rod_id}')

        #     self.C.setJointState(keyframes[t])
        #     self.C.view(False, f'place waypoint {t}')
        #     time.sleep(.1)
            
        return keyframes, q0
    
    def find_path(self, keyframes, q0, rod_id):
        full_path = []
        q_start = q0

        for keyframe_id, q_goal in enumerate(keyframes):
            rrt = ry.PathFinder()
            rrt.setProblem(self.C, q_start, q_goal)

            ret = rrt.solve()
            print(ret)

            path = ret.x
            full_path.append(path)

            # Replay only the path segment just planned
            for t in range(path.shape[0]):
                self.C.setJointState(path[t])
                self.C.view()
                time.sleep(.1)

            # Update attachment after reaching the keyframe
            self.C.setJointState(q_goal)

            if keyframe_id == 0:
                self.C.attach('a1_ur_gripper_center', f'rod_{rod_id}')
                print("rod attached to robot")

            elif keyframe_id == 1:
                self.C.attach('table', f'rod_{rod_id}')
                print("rod attached to table")

            q_start = q_goal

        return full_path


        
    
    # based on implementation of vhartman
    def solve_komo(self, komo, attempts = 100, mult = 3, offset = -1.5, view = False): 
        for attempt in range(attempts):
        
            if attempt > 0:
                dim = len(self.C.getJointState())
                x_init = np.random.rand(dim) * mult + offset
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)

            retval = solver.solve()
            retval = retval.dict()

            # print(retval)

            if view:
                print(retval)
                komo.view(True, "IK solution")


            if retval["feasible"]: #retval["ineq"] < 1 and retval["eq"] < 1 and 
                keyframes = komo.getPath()
                return keyframes
        
        print("FAILED to find solution")
        
        return None

    def prepare_next_grab(self, rod_id):

        print("prepare next grab")
        
        komo = ry.KOMO(self.C, phases=2, slicesPerPhase=10, kOrder=2, enableCollisions=True)

        komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
        komo.addControlObjective([], 2, 1e0)

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

        tool_pos = np.array(self.C.getFrame('ur_gripper_center').getPosition())
        tool_pos_up = tool_pos.copy()
        tool_pos_up[2] += 0.10   # move 10 cm up in world z

        komo.addObjective([1.], ry.FS.position, ['ur_gripper_center'], ry.OT.eq, [1e1], tool_pos_up)

        rod_center = np.array(self.C.getFrame(f"rod_{rod_id}").getPosition())
        rod_center_up = rod_center.copy()
        rod_center_up[2] += 0.10   # move 20 cm up in world z
        komo.addObjective([2.], ry.FS.position, ['ur_gripper_center'], ry.OT.eq, [1e1], rod_center_up)
        komo.addObjective([2.], ry.FS.scalarProductXZ, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [1.0])
        komo.addObjective([2.], ry.FS.scalarProductZZ, ['ur_gripper', 'world'], ry.OT.eq,[1e1],[-1.])

        ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
        print(ret)
        print(komo.report())

        if not ret.feasible:
            komo = ry.KOMO(self.C, phases=2, slicesPerPhase=10, kOrder=2, enableCollisions=True)

            komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
            komo.addControlObjective([], 2, 1e0)

            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
            komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

            tool_pos = np.array(self.C.getFrame('ur_gripper_center').getPosition())
            tool_pos_up = tool_pos.copy()
            tool_pos_up[2] += 0.10   # move 10 cm up in world z

            komo.addObjective([1.], ry.FS.position, ['ur_gripper_center'], ry.OT.eq, [1e1], tool_pos_up)

            rod_center = np.array(self.C.getFrame(f"rod_{rod_id}").getPosition())
            rod_center_up = rod_center.copy()
            rod_center_up[2] += 0.10   # move 20 cm up in world z
            komo.addObjective([2.], ry.FS.position, ['ur_gripper_center'], ry.OT.eq, [1e1], rod_center_up)
            komo.addObjective([2.], ry.FS.scalarProductXZ, ['ur_gripper_center', f"rod_{rod_id}"], ry.OT.eq, [1e1], [-1.0])
            komo.addObjective([2.], ry.FS.scalarProductZZ, ['ur_gripper', 'world'], ry.OT.eq,[1e1],[-1.])

            ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
            print(ret)
            print(komo.report())

            if not ret.feasible:
                komo.view(True, "IK solution")
                raise RuntimeError(f"Pick & PLace not possible for rod {rod_id}")

        # komo.view(True, "IK solution")
        q = komo.getPath()

        print('size of path:', q.shape)

        for t in range(q.shape[0]):
            self.C.setJointState(q[t])
            self.C.view(False, f'place waypoint {t}')
            time.sleep(.1)

        return   
    

    def husky_simple_move_test(self):

        target_position = np.array(self.C.getFrame("husky_coll_base_link").getPosition())
        target_position[0] += 3
        target_position[1] += 3 
        
        komo = ry.KOMO(self.C, phases=1, slicesPerPhase=10, kOrder=2, enableCollisions=True)

        komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
        komo.addControlObjective([], 2, 1e0)

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e2])
        komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

        komo.addObjective([1.], ry.FS.position, ["husky_coll_base_link"], ry.OT.eq, [1e1], target_position)
        komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e0], [], 1)

        ret = ry.NLP_Solver(komo.nlp(), verbose=4).solve()
        print(ret)

        komo.view(True, "IK solution")
        q = komo.getPath()

        print('size of path:', q.shape)

        for t in range(q.shape[0]):
            self.C.setJointState(q[t])
            self.C.view(False, f'place waypoint {t}')
            time.sleep(.1)


        return
    
    
    

if __name__ == "__main__":

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
    builder.import_husky()