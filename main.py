import copy
from time import perf_counter

import numpy as np

from backward_search import AssemblyPlanner
from DataClasses import AssemblyPlan
from rai.builder import RaiTrussBuilder
from truss import Truss


def create_rai_builder(truss):
    builder = RaiTrussBuilder(
        truss=truss,
        radius=0.005,
        scale=0.0011,
    )
    builder.import_robots()
    return builder


def make_rai_cache_key(
    step,
    q_current,
    supported,
    support_q,
):
    """Create a key for safely reusing a successful RAI transition.

    A cached motion is reusable only when the structural transition, physical
    support assignment, and robot starting configuration are the same.
    """
    return (
        tuple(sorted(step.rods_before)),
        int(step.rod_id),
        tuple(sorted(step.rods_after)),
        tuple(sorted(supported.items())),
        tuple(sorted(step.supports_after.items())),
        tuple(sorted(step.added_supports.items())),
        tuple(
            np.round(
                np.asarray(q_current, dtype=float),
                decimals=8,
            )
        ),
        tuple(
            sorted(
                (
                    gripper,
                    tuple(
                        np.round(
                            np.asarray(q, dtype=float),
                            decimals=8,
                        )
                    ),
                )
                for gripper, q in support_q.items()
            )
        ),
    )


def validate_structural_plan_with_rai(
    builder,
    q_initial,
    structural_steps,
    rai_cache,
):
    """Validate one complete structural removal plan sequentially in RAI.

    Successful transitions are cached. If a later transition fails, a new
    structural plan may reuse any cached prefix reached from the same physical
    robot and support state.
    """
    q_current = np.asarray(
        q_initial,
        dtype=float,
    ).copy()

    supported = {}
    support_q = {}
    records = []

    for step_index, step in enumerate(structural_steps):
        expected_supports_before = dict(step.supports_before)

        if supported != expected_supports_before:
            raise RuntimeError(
                "Structural/RAI support-state mismatch before "
                f"step {step_index}, rod {step.rod_id}.\n"
                f"RAI state:        {supported}\n"
                f"Structural state: {expected_supports_before}"
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

        candidate_is_supported = bool(releasable_supports)
        old_support_gripper = next(
            iter(releasable_supports),
            None,
        )

        print(
            "\nRAI validation "
            f"{step_index + 1}/{len(structural_steps)}: "
            f"remove rod {step.rod_id}"
        )

        cache_key = make_rai_cache_key(
            step=step,
            q_current=q_current,
            supported=supported,
            support_q=support_q,
        )

        cached_result = rai_cache.get(cache_key)

        if cached_result is not None:
            print(
                "Reusing cached RAI solution for "
                f"step {step_index}, rod {step.rod_id}"
            )
            motion_result = copy.deepcopy(cached_result)

        else:
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
                new_support_assignments=dict(step.added_supports),
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

            # Only successful results are cached. A RAI failure is represented
            # by the forbidden structural transition in the outer loop.
            rai_cache[cache_key] = copy.deepcopy(motion_result)

        q_current = np.asarray(
            motion_result["q_final"],
            dtype=float,
        ).copy()

        supported = dict(motion_result["supported"])

        support_q = {
            gripper: np.asarray(q, dtype=float).copy()
            for gripper, q in motion_result["support_q"].items()
        }

        records.append(copy.deepcopy(motion_result["record"]))

        expected_supports_after = dict(step.supports_after)

        if supported != expected_supports_after:
            raise RuntimeError(
                "Structural/RAI support-state mismatch after "
                f"step {step_index}, rod {step.rod_id}.\n"
                f"RAI state:        {supported}\n"
                f"Structural state: {expected_supports_after}"
            )

    return {
        "success": True,
        "builder": builder,
        "records": records,
        "failed_index": None,
        "failed_step": None,
    }


def filter_truss(truss, selected_rods):
    """Restrict a truss to a selected rod subset for testing."""
    selected_rods = set(selected_rods)

    unknown_rods = selected_rods - set(truss.elements)
    if unknown_rods:
        raise ValueError(
            f"Selected rods do not exist: {sorted(unknown_rods)}"
        )

    truss.elements = {
        rod_id: endpoints
        for rod_id, endpoints in truss.elements.items()
        if rod_id in selected_rods
    }

    truss.grounded_rods &= selected_rods

    truss.couplers = {
        (rod_1, rod_2)
        for rod_1, rod_2 in truss.couplers
        if rod_1 in selected_rods and rod_2 in selected_rods
    }

    print("Included rods:", sorted(truss.elements))


def main():
    truss = Truss.from_json(
        "JSON/own_examples/260804_RobArchDemo_ini.json"
    )

    support_grippers = (
        "h1_a1_ur_gripper_center",
        "h2_a1_ur_gripper_center",
    )

    filter_enabled = False  # Set to True to restrict the truss to a subset of rods.

    if filter_enabled:
        selected_rods = {
            # Example:
            4, 14, #2, 1,
        }
        filter_truss(truss, selected_rods)

    # The builder must be created after optional filtering so its RAI scene and
    # the structural truss contain exactly the same rods.
    rai_builder = create_rai_builder(truss)
    rai_initial_q = rai_builder.C.getJointState().copy()

    # These persist across complete structural replanning rounds.
    rai_cache = {}
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
            + "\nStructural search mode: rigidity only"
            + f"\nForbidden RAI transitions: "
            f"{len(forbidden_transitions)}"
            + f"\nCached RAI transitions: {len(rai_cache)}"
            + "\n"
            + "=" * 70
        )

        # Complete rigidity-only search. RAI is deliberately disabled here.
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
                f"Deepest structural sequence: {partial_sequence}\n"
                f"Forbidden RAI transitions: "
                f"{len(forbidden_transitions)}"
            )

        structural_steps = list(
            searcher.final_node.structural_steps
        )

        print("\nComplete rigidity-feasible sequence found:")
        print(removal_sequence)
        print("\nStarting sequential RAI validation...")

        validation = validate_structural_plan_with_rai(
            builder=rai_builder,
            q_initial=rai_initial_q,
            structural_steps=structural_steps,
            rai_cache=rai_cache,
        )

        if validation["success"]:
            accepted_sequence = list(removal_sequence)
            accepted_records = list(validation["records"])

            print("\nComplete sequence passed RAI validation.")
            break

        failed_step = validation["failed_step"]

        failed_transition = (
            AssemblyPlanner.structural_transition_key_from_step(
                failed_step
            )
        )

        if failed_transition in forbidden_transitions:
            raise RuntimeError(
                "The structural search returned an already forbidden "
                "RAI transition. Check forbidden-transition handling."
            )

        forbidden_transitions.add(failed_transition)

        print("\nRAI rejected transition:")
        print(f"  step: {validation['failed_index']}")
        print(f"  rod: {failed_step.rod_id}")
        print(
            "  rods before: "
            f"{sorted(failed_step.rods_before)}"
        )
        print(
            "  supports before: "
            f"{dict(failed_step.supports_before)}"
        )
        print(
            "Restarting the complete structural search with this "
            "transition blocked."
        )

    else:
        raise RuntimeError(
            "No complete RAI-valid sequence found after "
            f"{max_structural_replans} structural replans."
        )

    elapsed = perf_counter() - total_start_time

    print(
        f"\nLazy rigidity/RAI search took {elapsed:.2f} seconds."
    )
    print("Removal:", accepted_sequence)
    print("Assembly:", list(reversed(accepted_sequence)))
    print("Forbidden RAI transitions:", len(forbidden_transitions))
    print("Cached RAI transitions:", len(rai_cache))

    removal_plan = AssemblyPlan(
        removal_sequence=accepted_sequence,
        records=accepted_records,
    )

    assembly_plan = AssemblyPlan.reverse_removal_plan_to_assembly(
        removal_plan
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
