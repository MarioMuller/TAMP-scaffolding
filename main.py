from time import perf_counter

from truss import Truss
from backward_search import AssemblyPlanner
from rai.builder import RaiTrussBuilder
from DataClasses import AssemblyPlan


def main():
    # First test with the small scaffold.
    truss = Truss.from_json(
        # "JSON/own_examples/260804_FoC_demo.json"
        "JSON/own_examples/260804_RobArchDemo_ini.json"
    )
    
    filter = True
    
    if filter:
    
        # Filter to chose a subset of rods to include in the search. This is useful for testing
        selected_rods = {
            0, 5, 13, 15, 7, 6, #2, 1, # 45, 33, 55, 53, 47, 50, 49, 46 #, 34, 35, 38, 51, 39, 52, 54 # single cube
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

    # RAI scene used during backward search.
    builder = RaiTrussBuilder(
        truss=truss,
    )
    builder.import_robots(debug = False)

    searcher = AssemblyPlanner(
        truss=truss,
        builder=builder,
        max_supports=2,
        rigidity_cache_size=500,
    )

    start_time = perf_counter()

    removal_sequence = searcher.backward_search(
        capture_key="v",
        max_runtime=1800.0,
        max_expansions_without_progress=20_000,
    )

    elapsed = perf_counter() - start_time

    print(f"Backward search took {elapsed:.2f} seconds.")
    print("Stop reason:", searcher.search_stop_reason)
    print("Rigidity cache:", searcher.rigidity.cache_info())

    if removal_sequence is None:
        partial_sequence = (
            searcher.final_node.sequence
            if searcher.final_node is not None
            else []
        )

        raise RuntimeError(
            f"No complete sequence found. "
            f"Stop reason: {searcher.search_stop_reason}. "
            f"Partial removal sequence: {partial_sequence}"
        )

    assembly_sequence = list(reversed(removal_sequence))

    print("Removal:", removal_sequence)
    print("Assembly:", assembly_sequence)

    # Convert recorded removal motions into an assembly plan.
    removal_plan = AssemblyPlan(
        removal_sequence=searcher.final_node.sequence,
        records=searcher.final_node.records,
    )

    assembly_plan = AssemblyPlan.reverse_removal_plan_to_assembly(
        removal_plan
    )

    # Use a fresh builder for replay.
    replay_builder = RaiTrussBuilder(
        truss=truss,
        radius=0.005,
        scale=0.0011,
    )
    replay_builder.import_robots()

    replay_builder.display_recorded_plan_viser(
        assembly_plan,
        port=8080,
        pause_time=0.03,
        rod_pos=[-3.0, -1.0, 1.0],
        rod_ori=[0.5, 0.0, 0.5, 0.70710678],
        replay_mode="assembly",
    )


if __name__ == "__main__":
    main()