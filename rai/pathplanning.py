import time
import numpy as np
import robotic as ry

from experiment_metrics import CounterMetrics, Timer


class PathPlanner:
    def __init__(self, C, metrics=None):
        self.C = C
        self.metrics = metrics or CounterMetrics()

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

    def path_collision_free(self, path, Ctest, verbose=False):
        # check if a new path segment is collision free
        
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
        # TODO: Is it even useful to have the cost. Linear path should always be cheapest (only if cost function is evaluating all the same !, but might still be desirable)
        
        Ctest = ry.Config()
        Ctest.addConfigurationCopy(self.C)
        
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

            for k in range(j - i + 1):
                alpha = k / (j - i)
                candidate[i + k] = q0 + alpha * (q1 - q0)
        
            new_segment = candidate[i:j+1]
            
            if self.path_collision_free(new_segment, Ctest, verbose=False):
                best = candidate.copy()
                continue

        if verbose:
            print(f"original cost: {self.path_cost(path):.4f}")
            print(f"shortcut cost: {self.path_cost(best):.4f}")
            print(f"original points: {len(path)}")
            print(f"shortcut points: {len(best)}")

        return best
    
    def rrt(self, q_start, q_goal, attempts=50, active_joint_names=None):
        q_start = np.asarray(q_start, dtype=float).copy()
        q_goal = np.asarray(q_goal, dtype=float).copy()

        if active_joint_names is not None:
            return self.rrt_selected_joints(
                q_start=q_start,
                q_goal=q_goal,
                active_joint_names=active_joint_names,
                attempts=attempts,
            )
        
        self.metrics.inc("rrt_calls")

        with Timer(self.metrics, "rrt_time_s"):
            for attempt in range(attempts):
                rrt = ry.PathFinder()
                rrt.setProblem(self.C, q_start, q_goal)

                ret = rrt.solve()
                print(f"RRT returns: ", ret)
                
                if ret.feasible:
                    self.metrics.inc("rrt_successes")
                    return ret.x
                
        
        self.metrics.inc("rrt_failures")
        print("RRT failed to find a path")
        return None

    def rrt_selected_joints(
        self,
        q_start,
        q_goal,
        active_joint_names,
        attempts=50,
        frozen_tolerance=1e-6,
    ):
        if not hasattr(self.C, "selectJoints"):
            raise RuntimeError(
                "RRT support locking requires Config.selectJoints(...)"
            )

        all_joint_names = list(self.C.getJointNames())
        joint_index = {
            joint_name: i
            for i, joint_name in enumerate(all_joint_names)
        }

        def joint_frame_names(joint_names):
            frame_names = []

            for joint_name in joint_names:
                frame_name = joint_name.split(":", maxsplit=1)[0]

                if frame_name not in frame_names:
                    frame_names.append(frame_name)

            return frame_names

        active_joint_names = [
            joint_name
            for joint_name in active_joint_names
            if joint_name in joint_index
        ]

        active_indices = [
            joint_index[joint_name]
            for joint_name in active_joint_names
        ]

        frozen_indices = [
            i
            for i in range(len(all_joint_names))
            if i not in set(active_indices)
        ]

        frozen_delta = (
            np.linalg.norm(q_goal[frozen_indices] - q_start[frozen_indices])
            if frozen_indices
            else 0.0
        )

        if frozen_delta > frozen_tolerance:
            print(
                "Rejecting RRT segment because locked support joints "
                f"move by {frozen_delta:.6g}"
            )
            return None

        q_saved = self.C.getJointState().copy()
        all_joint_frame_names = joint_frame_names(all_joint_names)
        active_joint_frame_names = joint_frame_names(active_joint_names)

        try:
            self.C.setJointState(q_start)
            self.C.selectJoints(active_joint_frame_names)

            selected_joint_names = list(self.C.getJointNames())
            selected_indices = [
                joint_index[joint_name]
                for joint_name in selected_joint_names
            ]

            active_start = self.C.getJointState().copy()

            self.C.selectJoints(all_joint_frame_names)
            self.C.setJointState(q_goal)
            self.C.selectJoints(active_joint_frame_names)

            active_goal = self.C.getJointState().copy()

            self.C.selectJoints(all_joint_frame_names)
            self.C.setJointState(q_start)
            self.C.selectJoints(active_joint_frame_names)

            self.metrics.inc("rrt_calls")

            with Timer(self.metrics, "rrt_time_s"):
                for attempt in range(attempts):
                    rrt = ry.PathFinder()
                    rrt.setProblem(self.C, active_start, active_goal)

                    ret = rrt.solve()
                    print(f"RRT returns: ", ret)

                    if ret.feasible:
                        self.metrics.inc("rrt_successes")
                        active_path = np.asarray(ret.x, dtype=float)
                        full_path = np.tile(
                            q_start,
                            (len(active_path), 1),
                        )
                        full_path[:, selected_indices] = active_path
                        return full_path

            self.metrics.inc("rrt_failures")
            print("RRT failed to find a path")
            return None

        finally:
            self.C.selectJoints(all_joint_frame_names)
            self.C.setJointState(q_saved)
        
    def plan_segment(
        self,
        q_start,
        q_goal,
        do_shortcut=True,
        shortcut_iter=300,
        shortcut_step=0.02,
        rrt_attempts=50,
        active_joint_names=None,
    ):
        path = self.rrt(
            q_start=q_start,
            q_goal=q_goal,
            attempts=rrt_attempts,
            active_joint_names=active_joint_names,
        )

        if path is None:
            return None

        self.metrics.inc("planned_segments")
        self.metrics.add("path_points", len(path))
        self.metrics.add("path_cost", self.path_cost(path))

        if do_shortcut and len(path) >= 3:
            path = self.shortcut_path(
                path,
                max_iter=shortcut_iter,
                max_step=shortcut_step,
                min_gap=2,
                verbose=True,
            )
            self.metrics.add("shortcut_path_points", len(path))
            self.metrics.add("shortcut_path_cost", self.path_cost(path))


        return path        
    

    def play_path(self, path, dt=0.01, title="path"):
        path = np.asarray(path, dtype=float)
        for t in range(path.shape[0]):
            self.C.setJointState(path[t])
            self.C.view(False, f"{title} {t}")
            time.sleep(dt)
            
            
            
