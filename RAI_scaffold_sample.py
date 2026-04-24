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
        # self.C.view()
        # input("Press Enter to close...")
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
        # self.C.view()

        return
      
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
    
    def find_path(self, keyframes, q0, rod_id, show_visualization = False):
        q_start = np.asarray(q0, dtype=float).copy()

        for keyframe_id, q_goal in enumerate(keyframes):
            
            path = None
            
            for attempt in range (50):
                rrt = ry.PathFinder()
                rrt.setProblem(self.C, q_start, q_goal)

                ret = rrt.solve()
                print(f"RRT returns: ", ret)
                
                if ret.feasible:
                    path = ret.x
                    break
                
            if path is None:
                raise RuntimeError("RRT failed to find a Path")

            if show_visualization:
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

        return 
    
    # based on implementation of vhartman
    def solve_komo(self, komo, attempts = 1000, mult = 3, offset = -1.5, view = False): 
        for attempt in range(attempts):
        
            if attempt > 0:
                dim = len(self.C.getJointState())
                x_init = np.random.rand(dim) * mult + offset
                komo.initWithConstant(x_init)
                # komo.initWithPath(np.random.rand(3, 12) * 5 - 2.5)

            solver = ry.NLP_Solver(komo.nlp(), verbose=4)

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
 
    def path_cost(self, path, weights=None):
        """
        Computes path cost 
        """
        path = np.asarray(path, dtype=float)

        if path.ndim != 2:
            raise ValueError("path has invalid shape")

        if len(path) < 2:
            return 0.0

        diffs = np.diff(path, axis=0)

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (path.shape[1],):
                raise ValueError(
                    f"weights must have shape ({path.shape[1]},), got {weights.shape}"
                )
            diffs = diffs * weights

        return float(np.sum(np.linalg.norm(diffs, axis=1)))
  
    def interpolate_path(self, path, max_step = 0.02):
        """
        Interpolate the path to get a higher resolution path
        Based on Valentins implementation
        """
        path = np.asarray(path, dtype=float)
        new_path = []
        
        if len(path) == 0:
            print("Trying to interpolate empty path")
            return np.empty((0, 0), dtype=float)
        elif len(path) == 1:
            print("Interpolating between single point! Point is returned")
            return path

        # discretize path
        for i in range(len(path) - 1):
            q0 = path[i]
            q1 = path[i + 1]

            dist = np.linalg.norm(q1 - q0)
            N = max(2, int(np.ceil(dist / max_step)) + 1)
            dir = (q1 - q0) / N

            for j in range(N):
                q = q0 + dir * j
                new_path.append((q))

        # add the final state (which is not added in the interpolation before)
        new_path.append(path[-1])

        return np.asarray(new_path, dtype=float)

    def path_collision_free(self, path, verbose=False):
        # check if a new path segment is collision free
        
        Ctest = ry.Config()
        Ctest.addConfigurationCopy(self.C)
        
        path_np = np.asarray(path, dtype=float)
        if path_np.ndim != 2:
            return False

        q_start = self.C.getJointState().copy()

       
        for q in path_np:
            
            # set robot into joint configuration and test if it causes collision
            Ctest.setJointState(q)
            Ctest.computeCollisions()

            total_penetration = Ctest.getCollisionsTotalPenetration()

            if total_penetration > 1e-6:
                # print("Collision detected")
                # self.C.view()
                # time.sleep(1)
                return False
        
        # print("No Collision detected")
        return True
        
        # finally:
        #     self.C.setJointState(q_start)

    def shortcut_path(self, path, max_iter=200, max_step=0.02, min_gap=2, verbose=True):
        # shortcut if a segment results in a better (= shorter) path
        # TODO: Think about wheter just short q is acctually is the proper metric e.g. moving a joint 0.1rad is different to moving the husky 0.1 m
        # TODO: Is it even useful to have the cost. Linear path should always be cheapest
        
        path = np.asarray(path, dtype=float)
        new_path = self.interpolate_path(path, max_step=max_step)
        path = new_path
        
        if new_path.ndim != 2 or len(new_path) < 3:
            return new_path

        # setup current path as baseline
        best = new_path.copy()

        # Cut the path into three segments repeatedly and check if the interpolated path is collision free and cheaper
        for _ in range(max_iter):
            
            # The path is already a line -> line interpolation can't improve
            if len(best) < 3:
                break
            
            # randomly select two steps in the path to interpolate between
            i = np.random.randint(0, len(best))
            j = np.random.randint(0, len(best))

            # shortcut doesn't work between same point
            if i == j:
                continue
            
            # j should always be the first one
            if i > j:
                i, j = j, i
                
            # only test them if the path elements are not following each other
            if j - i < min_gap:
                continue

            q0 = best[i]
            q1 = best[j]

            if self.path_cost([best[i], best[j]]) >= self.path_cost(best[i:j+1]):
                continue
            
            candidate = best.copy()

            for k in range(j - i):
                q = q0 + (q1 - q0) / (j - i) * k
                candidate[i + k] = q
        
            
            if self.path_collision_free(candidate, verbose=False):
                best = candidate.copy()
                continue

        if verbose:
            print(f"original cost: {self.path_cost(path):.4f}")
            print(f"shortcut cost: {self.path_cost(best):.4f}")
            print(f"original points: {len(path)}")
            print(f"shortcut points: {len(best)}")

        return best

    def play_path(self, path, dt=0.01, title="path"):
        path = np.asarray(path, dtype=float)
        for t in range(path.shape[0]):
            self.C.setJointState(path[t])
            self.C.view(False, f"{title} {t}")
            time.sleep(dt)
            
    def find_path_shortcut(self, keyframes, q0, rod_id, do_shortcut=True, shortcut_iter=300, shortcut_step=0.02):
        full_path = []
        q_start = np.asarray(q0, dtype=float).copy()

        for keyframe_id, q_goal in enumerate(keyframes):
            q_goal = np.asarray(q_goal, dtype=float).copy()

            self.C.setJointState(q_start)
            
            path = None

            for attempt in range (50):
                rrt = ry.PathFinder()
                rrt.setProblem(self.C, q_start, q_goal)

                ret = rrt.solve()
                print(f"RRT returns: ", ret)
                
                if ret.feasible:
                    path = ret.x
                    break

            if path is None:
                raise RuntimeError(f"PathFinder failed for segment {keyframe_id}: ret.x is None")

            path = np.asarray(path, dtype=float)

            if path.ndim == 1:
                if path.shape[0] != len(q_start):
                    raise RuntimeError(
                        f"PathFinder failed for segment {keyframe_id}: invalid path shape {path.shape}"
                    )
                path = path.reshape(1, -1)

            if path.ndim != 2 or path.shape[0] == 0:
                raise RuntimeError(
                    f"PathFinder failed for segment {keyframe_id}: invalid path shape {path.shape}"
                )

            print(f"Segment {keyframe_id}: raw path points = {len(path)}, cost = {self.path_cost(path):.4f}")

            if do_shortcut and len(path) >= 3:
                path = self.shortcut_path(
                    path,
                    max_iter=shortcut_iter,
                    max_step=shortcut_step,
                    min_gap=2,
                    verbose=True
                )
                print(f"Segment {keyframe_id}: shortcut path points = {len(path)}, cost = {self.path_cost(path):.4f}")

            full_path.append(path)
            
            self.path_collision_free(path)
            
            # if self.path_collision_free(path):
            #     print("the path has been checked again and is collision free")
            # else:
            #     print("The path contains collisions but apparently it doesn't give a f")

            # replay the final segment path
            self.play_path(path, dt=0.005, title=f"segment {keyframe_id}")

            # snap exactly to goal before switching mode
            self.C.setJointState(q_goal)

            if keyframe_id == 0:
                self.C.attach('a1_ur_gripper_center', f'rod_{rod_id}')
                print("rod attached to robot")

            elif keyframe_id == 1:
                self.C.attach('table', f'rod_{rod_id}')
                print("rod attached to table")

            q_start = q_goal.copy()

        return full_path
    
    def super_simple_collision(self):
        
        self.C.computeCollisions()

        total_penetration = self.C.getCollisionsTotalPenetration()
        print(total_penetration)

        if total_penetration > 0:
            # print("collision")
            return False

if __name__ == "__main__":

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
    builder.import_husky()
