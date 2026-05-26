from truss import Truss
import numpy as np
from DataClasses import RodPathRecord, AttachmentEvent
from .scene import RaiScene
from .rods import RodManager
from .keyframes import KeyframePlanner
from .pathplanning import PathPlanner
from .replay import PlanReplayer
from .viser_replay import ViserPlanReplayer
import time

class RaiTrussBuilder:

    def __init__(self, truss, radius=0.0015, scale = 0.00351):
        self.truss = truss
        self.radius = radius
        self.scale = scale 
        
        self.scene = RaiScene()
        self.C = self.scene.C
        
        self.rods = RodManager(self.C, truss, radius=radius, scale=scale)
        
        self.keyframes = KeyframePlanner(self.C, self.rods)
        self.paths = PathPlanner(self.C)
        self.replayer = PlanReplayer(self.C, self.rods)
        self.viser_replayer = ViserPlanReplayer(self.C, self.rods)

    def import_husky(self):
        self.scene.import_husky()
        
    def import_support_husky(self, name="h2", base_q=(3.0, -3.0, 0.0)):
        self.scene.import_support_husky(
            name=name,
            base_q=base_q,
        )
        
    def import_floating_grippers_debug(self):
        self.scene.import_floating_grippers_debug()
        
    def import_robots(self):
        
        debug = True
        if debug:
            self.import_floating_grippers_debug()
            
        else: 
            self.import_husky()

            self.import_support_husky(
                name="h2",
            )
        
    def display_recorded_plan_viser(self, *args, **kwargs):
        return self.viser_replayer.display_recorded_plan_viser(*args, **kwargs)
    
    def reset_scene_with_rods(self, placed_rods):
        """
        Rebuilds scene with not yet removed rods in final pos
        """

        self.scene.clear()
        self.import_robots()

        for rod_id in placed_rods:
            self.rods.create_rod(
                rod_id,
                pos=[-3, -1, 1.0],
                ori=[0.5, 0.0, 0.5, 0.70710678],
            )

            self.rods.set_to_goal_pose(rod_id, view=False)

            self.C.attach("table", f"rod_{rod_id}")
            
        self.C.view()
    
    
    def _attach_and_record(
        self,
        record,
        rod_id,
        segment_id,
        parent,
        child,
    ):
        self.C.attach(parent, child)

        record.events.append(
            AttachmentEvent(
                rod_id=rod_id,
                segment_id=segment_id,
                parent=parent,
                child=child,
                action="attach",
            )
        )

        print(f"[event] segment={segment_id}: {child} -> {parent}")
        
        
    def move_support_to_rod_and_attach(
            self,
            record,
            rod_id,
            support_gripper="h2_a1_ur_gripper_center",
            main_gripper="a1_ur_gripper_center",
            use_rrt=False,
            shortcut_step=0.02,
        ):


        support_keyframes, support_q0 = self.keyframes.get_support_keyframes(
            rod_id,
            support_gripper=support_gripper,
            main_gripper=main_gripper,
            freeze_main=False,
            keep_rod_at_target=True,
        )

        q_start = np.asarray(support_q0, dtype=float).copy()
        q_goal = np.asarray(support_keyframes[-1], dtype=float).copy()

        if use_rrt:
            path = self.paths.plan_segment(
                q_start=q_start,
                q_goal=q_goal,
                do_shortcut=True,
                shortcut_iter=300,
                shortcut_step=shortcut_step,
                rrt_attempts=50,
            )

            if path is None:
                raise RuntimeError(f"Support robot failed to reach rod {rod_id}")
        else:
            path = self.paths.interpolate_path(
                np.asarray([q_start, q_goal], dtype=float),
                max_step=shortcut_step,
            )

        record.segments.append(path.copy())
        support_segment_id = len(record.segments) - 1

        self.C.setJointState(q_goal)

        self._attach_and_record(
            record=record,
            rod_id=rod_id,
            segment_id=support_segment_id,
            parent=support_gripper,
            child=f"rod_{rod_id}",
        )

        print(f"Support robot now holds rod {rod_id}")
        
    def show_keyframes(self, keyframes, title="keyframe", dt=1.0):
        """
        Visualize a list/array of keyframes in the RAI viewer.
        """
        if keyframes is None:
            print("No keyframes to show")
            return

        for i, q in enumerate(keyframes):
            print(f"Showing {title} {i}/{len(keyframes) - 1}")
            self.C.setJointState(q)
            self.C.view(False, f"{title} {i}")
            time.sleep(dt)
            
            
    def try_remove_and_commit_rod(
        self,
        current_state,
        new_state,
        rod_id,
        q_start=None,
        supported=None,
        support_q=None,
        candidate_is_supported=False,
        old_support_gripper=None,
        new_support_assignments=None,
        use_rrt=False,
        do_shortcut=False,
    ):
        """
        Backward-search motion test.

        current_state:
            rods currently installed before candidate removal

        new_state:
            rods remaining after candidate rod is removed

        supported:
            dict support_gripper -> rod_id
        """
        if supported is None:
            supported = {}

        if support_q is None:
            support_q = {}

        if new_support_assignments is None:
            new_support_assignments = {}

        # 1. Build scene with candidate rod still installed
        self.reset_scene_with_rods(current_state)

        if q_start is not None:
            self.C.setJointState(q_start)

        # 2. Restore branch-local support attachments
        for support_gripper, supported_rod in supported.items():
            print("Support robot is actually used")
            rod_frame = f"rod_{supported_rod}"
            if rod_frame in self.C.getFrameNames():
                self.C.attach(support_gripper, rod_frame)

        record = RodPathRecord(rod_id=rod_id)


        # 4. Compute removal keyframes
        keyframes, q0, new_supported, phase_info = self.keyframes.get_remove_keyframes_dual(
            rod_id=rod_id,
            supported=supported,
            support_q=support_q,
            candidate_is_supported=candidate_is_supported,
            old_support_gripper=old_support_gripper,
            new_support_assignments=new_support_assignments,
        )

        if keyframes is None:
            return None

        # 5. Convert keyframes into path segments
        q_current = self.C.getJointState().copy()
        for i, q_goal in enumerate(keyframes):
            if use_rrt:
                path = self.paths.plan_segment(
                    q_start=q_current,
                    q_goal=q_goal,
                    do_shortcut=do_shortcut,
                )
            else:
                path = np.asarray([q_current, q_goal])

            if path is None:
                return None

            record.segments.append(path)

            self.C.setJointState(q_goal)
            q_current = q_goal.copy()

        # ------------------------------------------------------------
        # Events
        # ------------------------------------------------------------

        main_grasp_segment_id = phase_info["main_grasp_segment"]
        pickup_segment_id = phase_info["pickup_segment"]

        # Main robot grasps the candidate rod while it is still part of the scaffold.
        self._attach_and_record(
            record=record,
            rod_id=rod_id,
            segment_id=main_grasp_segment_id,
            parent="a1_ur_gripper_center",
            child=f"rod_{rod_id}",
        )

        # If this candidate rod was previously supported, the old support releases it
        # after the main robot has taken over.
        if candidate_is_supported and old_support_gripper is not None:
            old_release_segment_id = phase_info.get("old_support_away_segment")

            # If the old support gripper is reused for a newly affected rod,
            # there is no safe-away phase. It releases the candidate when it starts
            # moving to the new support target.
            if old_release_segment_id is None and old_support_gripper in phase_info["new_support_segments"]:
                old_release_segment_id = phase_info["new_support_segments"][old_support_gripper] - 1

            if old_release_segment_id is None:
                old_release_segment_id = main_grasp_segment_id

            record.events.append(
                AttachmentEvent(
                    rod_id=rod_id,
                    segment_id=old_release_segment_id,
                    parent=old_support_gripper,
                    child=f"rod_{rod_id}",
                    action="detach",
                )
            )

            print(
                f"[event] segment={old_release_segment_id}: "
                f"{old_support_gripper} releases rod_{rod_id}"
            )

        # Newly affected rods get support before the candidate is detached/removed.
        for support_gripper, support_rod_id in new_support_assignments.items():
            support_segment_id = phase_info["new_support_segments"][support_gripper]

            self._attach_and_record(
                record=record,
                rod_id=support_rod_id,
                segment_id=support_segment_id,
                parent=support_gripper,
                child=f"rod_{support_rod_id}",
            )

        # Candidate rod detaches from the scaffold only immediately before the
        # pickup/removal segment. This prevents the early detach in viser.
        detach_candidate_segment_id = max(0, pickup_segment_id - 1)

        record.events.append(
            AttachmentEvent(
                rod_id=rod_id,
                segment_id=detach_candidate_segment_id,
                parent="table",
                child=f"rod_{rod_id}",
                action="detach",
            )
        )

        print(
            f"[event] segment={detach_candidate_segment_id}: "
            f"detach rod_{rod_id} from table"
        )       


        # ------------------------------------------------------------
        # Updated support joint states
        # ------------------------------------------------------------

        new_support_q = dict(support_q)

        # If candidate was supported, that support was released.
        if candidate_is_supported and old_support_gripper is not None:
            new_support_q.pop(old_support_gripper, None)

        # Store the exact joint configuration at which each new support robot
        # took over its affected rod.
        for support_gripper in new_support_assignments:
            support_segment_id = phase_info["new_support_segments"][support_gripper]
            q_support = np.asarray(record.segments[support_segment_id][-1], dtype=float).copy()
            new_support_q[support_gripper] = q_support
            
        return {
            "record": record,
            "q_final": q_current,
            "supported": new_supported,
            "support_q": new_support_q,
        }
               
   

if __name__ == "__main__":

    truss = Truss.from_json("JSON/long_beam_test.json")

    # build_entire_truss_in_rai(radius, node_positions, rods, C)
    builder = RaiTrussBuilder(truss, radius=0.0015)
   # builder.import_robots()
