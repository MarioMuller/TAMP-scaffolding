from time import perf_counter

from truss import Truss
from backward_search import AssemblyPlanner
from rai.builder import RaiTrussBuilder
from DataClasses import AssemblyPlan

def create_rai_builder(truss):
    builder = RaiTrussBuilder(
        truss=truss,
        radius=0.005,
        scale=0.0011,
    )

    builder.import_robots()
    return builder


def validate_structural_plan_with_rai(
    truss,
    structural_steps,
):
    """
    Validate one complete structural removal plan in RAI.

    Returns:
        {
            "success": bool,
            "builder": RaiTrussBuilder,
            "records": list[RodPathRecord],
            "failed_index": int | None,
            "failed_step": StructuralRemovalStep | None,
        }
    """

    # Every full validation starts from a clean robot configuration.
    builder = create_rai_builder(truss)

    q_current = builder.C.getJointState().copy()
    supported = {}
    support_q = {}
    records = []

    for step_index, step in enumerate(structural_steps):
        expected_supports_before = dict(
            step.supports_before
        )

        if supported != expected_supports_before:
            raise RuntimeError(
                "Structural/RAI support-state mismatch before "
                f"step {step_index}, rod {step.rod_id}.\n"
                f"RAI state:       {supported}\n"
                f"Structural state:{expected_supports_before}"
            )

        continuing_supports = {
            gripper: supported_rod
            for gripper, supported_rod in supported.items()
            if supported_rod != step.rod_id
        }

        releasable_supports = {
            gripper: supported_rod
            for gripper, supported_rod in supported.items()
            if supported_rod == step.rod_id
        }

        candidate_is_supported = bool(
            releasable_supports
        )

        old_support_gripper = next(
            iter(releasable_supports),
            None,
        )

        print(
            "\nRAI validation "
            f"{step_index + 1}/{len(structural_steps)}: "
            f"remove rod {step.rod_id}"
        )

        motion_result = builder.try_remove_and_commit_rod(
            current_state=step.rods_before,
            new_state=step.rods_after,
            rod_id=step.rod_id,
            q_start=q_current,
            supported=supported,
            support_q=support_q,
            candidate_is_supported=candidate_is_supported,
            old_support_gripper=old_support_gripper,
            continuing_supports=continuing_supports,
            releasable_supports=releasable_supports,
            new_support_assignments=dict(
                step.added_supports
            ),
            use_rrt=False,
            do_shortcut=False,
        )

        if motion_result is None:
            print(
                f"RAI failed at removal step {step_index}: "
                f"rod {step.rod_id}"
            )

            return {
                "success": False,
                "builder": builder,
                "records": records,
                "failed_index": step_index,
                "failed_step": step,
            }

        q_current = motion_result["q_final"].copy()
        supported = dict(
            motion_result["supported"]
        )
        support_q = dict(
            motion_result["support_q"]
        )
        records.append(
            motion_result["record"]
        )

        expected_supports_after = dict(
            step.supports_after
        )

        if supported != expected_supports_after:
            raise RuntimeError(
                "Structural/RAI support-state mismatch after "
                f"step {step_index}, rod {step.rod_id}.\n"
                f"RAI state:       {supported}\n"
                f"Structural state:{expected_supports_after}"
            )

    return {
        "success": True,
        "builder": builder,
        "records": records,
        "failed_index": None,
        "failed_step": None,
    }
    
def main():
    truss = Truss.from_json(
        "JSON/own_examples/260804_RobArchDemo_ini.json"
    )

    support_grippers = (
        "h1_a1_ur_gripper_center",
        "h2_a1_ur_gripper_center",
    )
    
    filter = False  # Set to True to filter rods for testing.
        
    if filter:
    
        # Filter to chose a subset of rods to include in the search. This is useful for testing
        selected_rods = {
            13, 2, 1 #15, 2, 1, # 0, 5, 45, 33, 55, 53, 47, 50, 49, 46 #, 34, 35, 38, 51, 39, 52, 54 # single cube
        }

        unknown_rods = selected_rods - set(truss.elements)
        if unknown_rods:
            raise ValueError(
                f"Selected rods do not exist: {sorted(unknown_rods)}"
            )

        # Keep only the selected rods.
        truss.elements = {
            rod_id: endpoints
            for rod_id, endpoints in truss.elements.items()
            if rod_id in selected_rods
        }

        # Keep grounding only for selected rods.
        truss.grounded_rods &= selected_rods

        # Keep couplers only when both connected rods are selected.
        truss.couplers = {
            (rod_1, rod_2)
            for rod_1, rod_2 in truss.couplers
            if rod_1 in selected_rods and rod_2 in selected_rods
        }

        print("Included rods:", sorted(truss.elements))

    forbidden_transitions = set()
    max_structural_replans = 100

    accepted_sequence = None
    accepted_records = None

    total_start_time = perf_counter()

    for planning_round in range(max_structural_replans):
        print(
            "\n"
            + "=" * 70
            + f"\nStructural planning round {planning_round + 1}"
            + f"\nForbidden RAI transitions: "
              f"{len(forbidden_transitions)}"
            + "\n"
            + "=" * 70
        )

        # Run the backward search with rigidity check only, without RAI validation. 
        # The RAI validation will be done after a complete structural plan is found.
        searcher = AssemblyPlanner(
            truss=truss,
            builder=None,
            max_supports=2,
            support_grippers=support_grippers,
            forbidden_transitions=forbidden_transitions,
        )

        removal_sequence = searcher.backward_search(
            capture_key="v",
        )

        if removal_sequence is None:
            partial_sequence = (
                searcher.final_node.sequence
                if searcher.final_node is not None
                else []
            )

            raise RuntimeError(
                "No structurally feasible complete sequence remains.\n"
                f"Stop reason: {searcher.search_stop_reason}\n"
                f"Partial sequence: {partial_sequence}\n"
                f"Forbidden RAI transitions: "
                f"{len(forbidden_transitions)}"
            )

        structural_steps = list(
            searcher.final_node.structural_steps
        )

        print(
            "\nComplete structural sequence found:"
        )
        print(removal_sequence)

        print(
            "\nStarting RAI validation of the complete sequence..."
        )

        validation = validate_structural_plan_with_rai(
            truss=truss,
            structural_steps=structural_steps,
        )

        if validation["success"]:
            print(
                "\nComplete sequence passed RAI validation."
            )

            accepted_sequence = list(removal_sequence)
            accepted_records = list(
                validation["records"]
            )
            accepted_searcher = searcher
            break

        failed_step = validation["failed_step"]

        failed_transition = (
            AssemblyPlanner
            .structural_transition_key_from_step(
                failed_step
            )
        )

        if failed_transition in forbidden_transitions:
            raise RuntimeError(
                "The same forbidden transition was returned "
                "again. Check the transition-key handling."
            )

        forbidden_transitions.add(
            failed_transition
        )

        print(
            "\nRAI rejected the transition:"
        )
        print(
            f"  rod: {failed_step.rod_id}"
        )
        print(
            f"  rods before: "
            f"{sorted(failed_step.rods_before)}"
        )
        print(
            f"  supports before: "
            f"{failed_step.supports_before}"
        )
        print(
            "Restarting the structural backward search "
            "with this transition forbidden."
        )

    else:
        raise RuntimeError(
            f"No RAI-valid sequence found after "
            f"{max_structural_replans} structural plans."
        )

    elapsed = perf_counter() - total_start_time

    print(
        f"\nLazy structural/RAI search took "
        f"{elapsed:.2f} seconds."
    )
    print("Removal:", accepted_sequence)
    print(
        "Assembly:",
        list(reversed(accepted_sequence)),
    )
    print(
        "Forbidden RAI transitions:",
        len(forbidden_transitions),
    )

    removal_plan = AssemblyPlan(
        removal_sequence=accepted_sequence,
        records=accepted_records,
    )

    assembly_plan = (
        AssemblyPlan.reverse_removal_plan_to_assembly(
            removal_plan
        )
    )

    replay_builder = create_rai_builder(truss)

    replay_builder.display_recorded_plan_viser(
        assembly_plan,
        port=8080,
        pause_time=0.03,
        rod_pos=[-3.0, -1.0, 1.0],
        rod_ori=[
            0.5,
            0.0,
            0.5,
            0.70710678,
        ],
        replay_mode="assembly",
    )


if __name__ == "__main__":
    main()