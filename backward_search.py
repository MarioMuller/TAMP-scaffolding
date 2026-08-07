from truss import Truss
from collections import defaultdict, deque
import heapq
from DataClasses import SearchNode, StructuralRemovalStep
from rigidityCheck.truss_rigidity import TrussRigidityChecker

# for debugging
import select
import sys
import termios
import tty
from contextlib import nullcontext

class TerminalHotkey:
    """Read individual terminal keys without stopping the search."""

    def __init__(self, key="v"):
        self.key = key.lower()
        self.fd = None
        self.previous_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Keyboard capture requires a real terminal. "
                "Run the program in an integrated or normal terminal."
            )

        self.fd = sys.stdin.fileno()
        self.previous_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        return self

    def consume_presses(self):
        presses = 0

        while select.select([sys.stdin], [], [], 0.0)[0]:
            pressed_key = sys.stdin.read(1).lower()

            if pressed_key == self.key:
                presses += 1

        return presses

    def __exit__(self, exc_type, exc_value, traceback):
        if self.previous_settings is not None:
            termios.tcsetattr(
                self.fd,
                termios.TCSADRAIN,
                self.previous_settings,
            )

class AssemblyPlanner:
    def __init__(self, truss, builder=None, max_supports=2):
        self.truss = truss
        self.builder = builder
        self.max_supports = max_supports
        self.rigidity = TrussRigidityChecker(truss)

        self.virtual_supports = tuple(
            f"support_{index + 1}"
            for index in range(max_supports)
        )
        
        # debug
        self.debug_capture_active = False
        self.debug_capture_steps = []
        
    def process_debug_hotkey(self, hotkey):
        if hotkey is None:
            return

        for _ in range(hotkey.consume_presses()):
            if not self.debug_capture_active:
                self.debug_capture_steps.clear()
                self.debug_capture_active = True

                print(
                    "\nDebug capture started. "
                    "Press V again to display the tested configurations."
                )

            else:
                self.debug_capture_active = False

                print(
                    f"\nDebug capture stopped: "
                    f"{len(self.debug_capture_steps)} configurations recorded."
                )

                if not self.debug_capture_steps:
                    print("Nothing was tested during the capture interval.")
                    continue

                # Import lazily to avoid coupling the normal search to plotting.
                from rigidityCheck.structural_replay import (
                    display_structural_assembly,
                )

                display_structural_assembly(
                    truss=self.truss,
                    removal_steps=list(self.debug_capture_steps),
                    scale=0.0011,
                    label_rods=True,
                    video_path=None,
                    seconds_per_step=0.5,
                    fps=30,
                )

                print("Search continuing. Press V to start another capture.")

    @property
    def helper_grippers(self):
        if self.builder is None:
            return self.virtual_supports

        return tuple(self.builder.support_grippers)

    # create graph structure
    def build_graph(self, active_rods):
        graph = defaultdict(set)
        active_nodes = set()

        for eid in active_rods:
            n1, n2 = self.truss.elements[eid]
            graph[n1].add(n2)
            graph[n2].add(n1)
            active_nodes.add(n1)
            active_nodes.add(n2)

        return graph, active_nodes

    # check that rods are not flying
    def is_valid_state(self, active_rods, supported_rods=None):
        return self.rigidity.is_rigid(active_rods, supported_rods=supported_rods)

    # use height as heuristicc
    def heuristic(self, rod_id):
        n1, n2 = self.truss.elements[rod_id]
        return 0.5 * (self.truss.nodes[n1][2] + self.truss.nodes[n2][2])
    
    def is_supported_candidate(self, node, rod_id):
        return rod_id in node.supported.values()

    # TODO: Combine this with heuristic
    def removal_priority(self, node, rod_id):
        support_count = len(
            set(node.supported.values())
        )

        grounded_rank = (
            1 if rod_id in self.truss.grounded_rods else 0
        )

        connection_count = self.active_connection_count(
            node,
            rod_id,
        )

        supported_rank = (
            0 if self.is_supported_candidate(node, rod_id) else 1
        )

        return (
            len(node.state),      # 1. Prefer deeper search states
            support_count,        # 2. Prefer states requiring fewer supports
            grounded_rank,        # 3. Remove non-grounded rods first
            connection_count,     # 4. Remove rods with fewer connections
            supported_rank,
            -self.heuristic(rod_id),
            rod_id,
        )

    def choose_placeholder_support_targets(
        self,
        node,
        removed_rod,
        new_state,
        max_targets=2,
        probability_two=0.5,
    ):
        return self.rigidity.choose_support_targets(
            active_rods=new_state,
            already_supported=node.supported.values(),
            max_targets=max_targets,
            key=self.heuristic,
        )
    

    # greedy backward search
    def backward_search(self, capture_key=None):
        
        """
        Run the backward search
        """
        hotkey_context = (
            TerminalHotkey(capture_key)
            if capture_key is not None
            else nullcontext(None)
        )
        
        # initial state: all rods in final position
        initial_state = frozenset(self.truss.elements.keys())

        open_list = []
        counter = 0

        # initialize search

        initial_node = SearchNode(
            state=initial_state,
            sequence=[],
            q=None,
            supported={},
            support_q={},
            records=[],
            structural_steps=[],
        )
        
        def configuration_key(state, supported_rods):
            return (
                state,
                frozenset(supported_rods),
            )

        visited = {
            configuration_key(
                initial_node.state,
                initial_node.supported.values(),
            )
        }
        
        attempted_transitions = set()

        # Add all possible first removals to the open list
        for candidate_rod in initial_state:
            priority = self.removal_priority(initial_node, candidate_rod)
            heapq.heappush(open_list, (priority, counter, initial_node, candidate_rod))
            counter += 1

        with hotkey_context as hotkey:
            print(
                f"Search running. Press {capture_key.upper()} to start/stop capture."
                if capture_key
                else "Search running."
            )

            # loop until solution found or open list exhausted
            while open_list:
                priority, _, node, candidate_rod = heapq.heappop(
                    open_list
                )

                parent_key = configuration_key(
                    node.state,
                    node.supported.values(),
                )

                transition_key = (
                    parent_key,
                    candidate_rod,
                )

                if transition_key in attempted_transitions:
                    continue

                attempted_transitions.add(transition_key)

                new_state = frozenset(
                    node.state - {candidate_rod}
                )

                state_key = configuration_key(
                    new_state,
                    (
                        rod_id
                        for rod_id in node.supported.values()
                        if rod_id != candidate_rod
                    ),
                )

                if state_key in visited:
                    continue

                feasible, result = self.is_removal_feasible(
                    node,
                    candidate_rod,
                )
                
                # Record both successful and unsuccessful tested configurations.
                if self.debug_capture_active and result is not None:
                    structural_step = result.get("structural_step")

                    if structural_step is not None:
                        self.debug_capture_steps.append(
                            structural_step
                        )

                # If V was pressed while the rigidity calculation was running,
                # process it now.
                self.process_debug_hotkey(hotkey)

                if not feasible:
                    continue

                motion_record = result["motion_record"]

                new_node = SearchNode(
                    state=new_state,
                    sequence=node.sequence + [candidate_rod],
                    q=result["q_final"],
                    supported=result["supported"],
                    support_q=result["support_q"],
                    records=(
                        node.records
                        + ([motion_record] if motion_record is not None else [])
                    ),
                    structural_steps=(
                        node.structural_steps
                        + [result["structural_step"]]
                    ),
                )
                
                state_key = configuration_key(
                    new_node.state,
                    new_node.supported.values(),
                )

                if state_key in visited:
                    continue

                visited.add(state_key)

                if len(new_state) == 0:
                    self.final_node = new_node
                    return new_node.sequence

                # # debug stopping condition
                # if len(new_node.sequence) >= 10:
                #     self.final_node = new_node
                #     return new_node.sequence

                for next_rod in new_state:
                    priority = self.removal_priority(new_node, next_rod)
                    heapq.heappush(
                        open_list,
                        (priority, counter, new_node, next_rod)
                    )
                    counter += 1

        return None
    
    def active_connection_count(self, node, rod_id):
        """Number of rods currently coupled to rod_id."""
        return sum(
            1
            for rod_1, rod_2 in self.truss.couplers
            if (
                rod_1 == rod_id and rod_2 in node.state
            )
            or (
                rod_2 == rod_id and rod_1 in node.state
            )
        )

    def is_removal_feasible(self, node, candidate_rod):
        current_state = node.state
        new_state = frozenset(current_state - {candidate_rod})

        current_supports = dict(node.supported)

        continuing_supports = {
            support: rod_id
            for support, rod_id in current_supports.items()
            if rod_id != candidate_rod
        }

        releasable_supports = {
            support: rod_id
            for support, rod_id in current_supports.items()
            if rod_id == candidate_rod
        }

        candidate_is_supported = bool(releasable_supports)
        old_support_gripper = next(iter(releasable_supports), None)

        # A support holding the removed candidate becomes available again.
        free_supports = [
            support
            for support in self.helper_grippers
            if support not in continuing_supports
        ]

        continuing_supported_rods = set(
            continuing_supports.values()
        )

        # First test whether existing continuing supports are enough.
        result_without_new_support = self.rigidity.check(
            new_state,
            supported_rods=continuing_supported_rods,
        )
        
        def make_structural_step(
            rigidity_result,
            supported_after,
            added_supports,
        ):
            return StructuralRemovalStep(
                rod_id=candidate_rod,
                rods_before=frozenset(current_state),
                rods_after=frozenset(new_state),
                supports_before=dict(node.supported),
                supports_after=dict(supported_after),
                added_supports=dict(added_supports),
                released_supports=dict(releasable_supports),
                rank_after=rigidity_result.rank,
                dof_after=rigidity_result.dof,
            )

        if result_without_new_support.is_rigid:
            affected_rods = []

        elif not free_supports:
            print(
                f"Rod {candidate_rod} cannot be removed: "
                "the remaining scaffold is not rigid and no support is free."
            )
            
            structural_step = make_structural_step(
                rigidity_result=result_without_new_support,
                supported_after=continuing_supports,
                added_supports={},
            )
            
            return False, None

        else:
            affected_rods = self.rigidity.choose_support_targets(
                active_rods=new_state,
                already_supported=continuing_supported_rods,
                max_targets=len(free_supports),
                key=self.heuristic,
            )

        new_support_assignments = {
            support: rod_id
            for support, rod_id in zip(
                free_supports,
                affected_rods,
            )
        }

        next_supported = dict(continuing_supports)
        next_supported.update(new_support_assignments)

        rigidity_result = self.rigidity.check(
            new_state,
            supported_rods=next_supported.values(),
        )

        structural_step = make_structural_step(
            rigidity_result=rigidity_result,
            supported_after=next_supported,
            added_supports=new_support_assignments,
        )

        if not rigidity_result.is_rigid:
            print(
                f"Removing rod {candidate_rod} is structurally infeasible: "
                f"rank {rigidity_result.rank}/{rigidity_result.dof}, "
                f"supports {sorted(next_supported.values())}."
            )
            return False, {
                "structural_step": structural_step,
            }

        print(
            f"Remove rod {candidate_rod}: "
            f"remaining supports = {next_supported or 'none'}"
        )

        # ---------------------------------------------------------
        # Optional motion-planning validation
        # ---------------------------------------------------------

        if self.builder is None:
            q_final = None
            support_q = {}
            motion_record = None

        else:
            motion_result = self.builder.try_remove_and_commit_rod(
                current_state=current_state,
                new_state=new_state,
                rod_id=candidate_rod,
                q_start=node.q,
                supported=node.supported,
                support_q=node.support_q,
                candidate_is_supported=candidate_is_supported,
                old_support_gripper=old_support_gripper,
                continuing_supports=continuing_supports,
                releasable_supports=releasable_supports,
                new_support_assignments=new_support_assignments,
                use_rrt=False,
                do_shortcut=False,
            )

            if motion_result is None:
                return False, None

            q_final = motion_result["q_final"]
            next_supported = motion_result["supported"]
            support_q = motion_result["support_q"]
            motion_record = motion_result["record"]

        structural_step = StructuralRemovalStep(
            rod_id=candidate_rod,
            rods_before=frozenset(current_state),
            rods_after=frozenset(new_state),
            supports_before=dict(node.supported),
            supports_after=dict(next_supported),
            added_supports=dict(new_support_assignments),
            released_supports=dict(releasable_supports),
            rank_after=rigidity_result.rank,
            dof_after=rigidity_result.dof,
        )

        return True, {
            "q_final": q_final,
            "supported": dict(next_supported),
            "support_q": support_q,
            "motion_record": motion_record,
            "structural_step": structural_step,
        }





if __name__ == "__main__":
    # truss = Truss.from_json("JSON/long_beam_test.json")
    truss = Truss.from_json("JSON/scaffold_test.json")
    searcher = AssemblyPlanner(truss)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
    print("Assembly:", assembly_sequence)


