# Function Summary

This file lists the Python functions and methods in the project and briefly describes what each one does.

## `backward_search.py`

- `AssemblyPlanner.__init__(truss, builder=None)`: Stores the truss, optional RAI builder, motion records, and support state for the search.
- `AssemblyPlanner.build_graph(active_rods)`: Builds an adjacency graph and active-node set from the currently active rods.
- `AssemblyPlanner.is_valid_state(active_rods)`: Checks whether every connected component of the active rods is connected to at least one grounded node.
- `AssemblyPlanner.is_motion_feasible(support_rods, rod_id)`: Uses the builder to test whether a rod can be placed given the current support rods.
- `AssemblyPlanner.heuristic(rod_id)`: Scores a rod by the average height of its two endpoints.
- `AssemblyPlanner.backward_search()`: Runs a priority-based backward removal search and records feasible removal motions.
- `AssemblyPlanner.is_removal_feasible(node, rod_id)`: Tests whether a candidate rod can be removed from a search node, including optional support handling.

## `DataClasses.py`

- `AssemblyPlan.assembly_sequence`: Property that returns the reverse of the removal sequence.
- `AssemblyPlan.add(rod_id, record)`: Adds a removed rod and its motion record to the plan.
- `AssemblyPlan.records_in_assembly_order()`: Returns recorded rod paths in forward assembly order.
- `AssemblyPlan.sequence()`: Convenience wrapper that returns `assembly_sequence`.
- `AssemblyPlan.reverse_removal_plan_to_assembly(removal_plan)`: Converts a recorded removal plan into a forward assembly plan by reversing segments and rewriting attachment events.
- `SearchNode.unused_helpers(helper_grippers)`: Returns helper grippers that are not currently supporting rods.
- `SearchNode.first_unused_helper(helper_grippers)`: Returns the first available helper gripper, if any.
- `SearchNode.has_unused_helper(helper_grippers)`: Checks whether at least one helper gripper is available.

## `truss.py`

- `Truss.__init__(nodes, elements, grounded_nodes)`: Stores node positions, rod elements, and grounded nodes.
- `Truss.from_json(path)`: Loads a truss definition from a JSON file.

## `rai/utils.py`

- `quaternion_from_z_to_vector(direction)`: Computes a quaternion that rotates the local z-axis onto a target direction vector.

## `rai/builder.py`

- `RaiTrussBuilder.__init__(truss, radius=0.0015, scale=0.00351)`: Creates the RAI scene, rod manager, keyframe planner, path planner, and replay helpers.
- `RaiTrussBuilder.import_husky()`: Imports the main Husky robot into the scene.
- `RaiTrussBuilder.import_support_husky(name="h2", base_q=(3.0, -3.0, 0.0))`: Imports a support Husky robot with a chosen name and base pose.
- `RaiTrussBuilder.import_floating_grippers_debug()`: Imports simplified floating grippers for debugging.
- `RaiTrussBuilder.import_robots()`: Imports the active robot setup; currently uses floating debug grippers.
- `RaiTrussBuilder.replay_recorded_plan(*args, **kwargs)`: Delegates recorded-plan replay to `PlanReplayer`.
- `RaiTrussBuilder.display_recorded_plan_viser(*args, **kwargs)`: Delegates browser-based plan display to `ViserPlanReplayer`.
- `RaiTrussBuilder.reset_scene_with_rods(placed_rods)`: Clears and rebuilds the scene with selected rods fixed at their goal poses.
- `RaiTrussBuilder._attach_and_record(record, rod_id, segment_id, parent, child)`: Attaches one frame to another and records the attachment event.
- `RaiTrussBuilder.move_support_to_rod_and_attach(...)`: Plans a support gripper motion to a rod and records the support attachment.
- `RaiTrussBuilder.show_keyframes(keyframes, title="keyframe", dt=1.0)`: Steps through keyframes in the RAI viewer.
- `RaiTrussBuilder.try_remove_and_commit_rod(...)`: Tries to remove a rod, creates path segments, updates attachments, and returns the resulting removal record.
- `RaiTrussBuilder._detach_and_record(record, rod_id, segment_id, child, new_parent="world")`: Detaches a child frame by reattaching it to a new parent and records the detach event.
- `RaiTrussBuilder.try_plan_and_commit_rod(...)`: Tries to plan and commit a rod placement motion, including optional support and replay.

## `rai/keyframes.py`

- `KeyframePlanner.__init__(C, rod_manager)`: Stores the RAI config and rod manager used for keyframe planning.
- `KeyframePlanner.solve_komo(komo, attempts=1000, mult=3, offset=-1.5, view=False, view_accepted=False)`: Solves a KOMO problem, retrying with random initial states until a feasible path is found.
- `KeyframePlanner.get_remove_keyframes_dual(...)`: Builds and solves a dual-arm KOMO problem for removing a rod to a pickup pose.
- `KeyframePlanner.get_remove_keyframes_with_support(...)`: Builds and solves a KOMO problem where a support robot stabilizes another rod during removal.
- `KeyframePlanner.choose_support_rod_after_removal(removed_rod_id)`: Picks a remaining rod frame to use as a temporary support target.
- `KeyframePlanner.get_keyframes(rod_id)`: Generates single-arm grasp and placement keyframes for a rod.
- `KeyframePlanner.get_keyframes_dual(...)`: Generates dual-arm grasp, carry, placement, and return keyframes for a rod.
- `KeyframePlanner.get_support_keyframes(...)`: Finds a keyframe where a support gripper grasps a rod at one of several candidate fractions.

## `rai/pathplanning.py`

- `PathPlanner.__init__(C)`: Stores the RAI config used for path planning.
- `PathPlanner.path_cost(path, weights=None)`: Computes the summed joint-space length of a path, optionally with per-joint weights.
- `PathPlanner.interpolate_path(path, max_step=0.02)`: Densifies a path by linearly interpolating between waypoints.
- `PathPlanner.path_collision_free(path, Ctest, verbose=False)`: Checks whether each waypoint in a path is collision-free in a test config.
- `PathPlanner.shortcut_path(path, max_iter=200, max_step=0.02, min_gap=2, verbose=True)`: Attempts random path shortcuts and keeps collision-free improvements.
- `PathPlanner.rrt(q_start, q_goal, attempts=50)`: Uses RAI's `PathFinder` to search for an RRT path between two joint states.
- `PathPlanner.plan_segment(...)`: Runs RRT for one segment and optionally shortcuts the resulting path.
- `PathPlanner.play_path(path, dt=0.01, title="path")`: Plays a path in the RAI viewer.

## `rai/replay.py`

- `PlanReplayer.__init__(C, rod_manager)`: Stores the RAI config and rod manager used for replay.
- `PlanReplayer.replay_recorded_plan(...)`: Replays recorded rod path segments in the RAI viewer and applies recorded attach/detach events.

## `rai/rods.py`

- `RodManager.__init__(C, truss, radius=0.0015, scale=0.00351)`: Stores scene, truss, rod radius, and coordinate scale.
- `RodManager.get_rod_endpoints(rod_id)`: Returns the scaled endpoint positions for a rod.
- `RodManager.get_goal_pose(rod_id)`: Computes the rod's final center position and orientation from its truss endpoints.
- `RodManager.get_rod_length(rod_id)`: Computes the displayed rod length, including the same shortening used when creating rods.
- `RodManager.create_rod(rod_id, pos=[-0.4, -0.05, 0.2], ori=[0.7070, 1, 0, 0.7070])`: Adds a cylindrical rod frame to the RAI config.
- `RodManager.create_target_frame(rod_id)`: Creates or updates a world-frame target pose for a rod.
- `RodManager.create_dual_arm_grasp_frames(...)`: Creates two fixed grasp frames on a rod for dual-arm manipulation.
- `RodManager.set_to_end_position(rod_id)`: Moves a rod to its final truss pose and opens the RAI viewer.
- `RodManager.set_to_goal_pose(rod_id, view=False)`: Moves a rod to its computed goal pose, optionally showing the viewer.
- `RodManager.create_sliding_support_grasp_frame(rod_id)`: Creates a support grasp frame with a translational joint along the rod.
- `RodManager.create_support_grasp_frame_at_fraction(rod_id, fraction)`: Creates a fixed support grasp frame at a fraction along the rod length.

## `rai/scene.py`

- `RaiScene.__init__()`: Creates a new RAI config with a world frame.
- `RaiScene.clear()`: Clears the config and recreates the world frame.
- `RaiScene.import_husky()`: Adds the table, Husky base, and two UR5 arms to the scene.
- `RaiScene.import_support_husky(name="h2", base_q=(3.0, -3.0, 0.0), arm="right")`: Adds a named single-arm support Husky to the scene.
- `RaiScene.import_floating_grippers_debug()`: Adds table and floating Robotiq grippers for simplified debugging.
- `add_floating_robotiq(prefix, ball_name, pos, color)`: Nested helper inside `import_floating_grippers_debug` that creates one floating Robotiq gripper.

## `rai/viser_replay.py`

- `ViserPlanReplayer.__init__(C, rod_manager)`: Stores the RAI config and rod manager used for Viser replay.
- `ViserPlanReplayer._build_display_config(...)`: Builds a display config containing all rods referenced by a recorded plan.
- `ViserPlanReplayer._precompute_viser_steps(recorder, C_base, replay_mode="removal")`: Simulates replay once and stores frame poses, visibility, rod IDs, and segment IDs for each step.
- `ViserPlanReplayer._viser_set_step(i, steps, handles, mode_label=None)`: Applies one precomputed replay step to the Viser scene handles and status label.
- `ViserPlanReplayer.display_recorded_plan_viser(...)`: Starts a Viser server and exposes GUI controls for stepping through or playing a recorded plan.
- `clamped_step()`: Nested helper inside `display_recorded_plan_viser` that clamps the GUI slider step to a valid range.
- `_(_)`: Nested Viser callback names used for Stop, Prev, Next, and slider-update events.
