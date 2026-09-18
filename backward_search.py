import hashlib

from shapely import node
from truss import Truss
from collections import deque
import heapq
import time
import numpy as np
from dataclasses import dataclass

from DataClasses import SearchNode, StructuralRemovalStep
from rigidityCheck.Datastructures import ElementStatus
from rigidityCheck.truss_rigidity import TrussRigidityChecker

# for debugging
import select
import sys
import termios
import tty
from contextlib import nullcontext, redirect_stdout


@dataclass(frozen=True)
class RemovalSupportContext:
    """Support state implied by trying to remove one candidate rod."""

    current_state: frozenset[int]
    new_state: frozenset[int]
    continuing_supports: dict[str, int]
    releasable_supports: dict[str, int]
    free_supports: list[str]
    continuing_supported_rods: set[int]
    candidate_is_supported: bool
    old_support_gripper: str | None


@dataclass(frozen=True)
class SupportEvaluation:
    """Structural support outcome for one candidate removal."""

    feasible: bool
    rigidity_result: object
    supports_after: dict[str, int]
    added_supports: dict[str, int]

    @property
    def support_count(self):
        return len(self.supports_after)

    @property
    def new_support_count(self):
        return len(self.added_supports)


class TerminalHotkey:
    """Read individual terminal keys without stopping the search."""

    def __init__(self, key="v"):
        self.key = key.lower()
        self.fd = None
        self.previous_settings = None
        self.enabled = True

    def __enter__(self):
        if not sys.stdin.isatty():
            self.enabled = False
            print(
                "Keyboard capture disabled: stdin is not a real terminal. "
                "Run from an integrated or normal terminal to use the hotkey."
            )
            return self

        self.fd = sys.stdin.fileno()
        self.previous_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

        return self

    def consume_presses(self):
        presses = 0

        if not self.enabled:
            return presses

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
    STRATEGY_NAMES = (
        "default",
        "baseline",
        "highest_first",
        "fast_reduce_support",
        "reduced_overall_supports",
    )
    SUPPORT_TARGET_ORDERS = (
        "lowest_first",
        "least_connected",
        "random",
        "furthest_from_removed",
    )

    def __init__(
        self,
        truss,
        builder=None,
        max_supports=2,
        rigidity_cache_size=2000,
        support_grippers=None,
        forbidden_transitions=None,
        strategy_name="default",
        random_seed=0,
        shuffle_ties=False,
        support_target_order="lowest_first",
    ):
        self.truss = truss
        self.builder = builder
        self.max_supports = max_supports
        if strategy_name not in self.STRATEGY_NAMES:
            raise ValueError(
                "strategy_name must be one of: "
                f"{', '.join(self.STRATEGY_NAMES)}."
            )
        self.strategy_name = strategy_name
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(random_seed)
        self.baseline_candidate_orders = {}
        self.support_evaluations = {}
        self._rod_physical_distance_cache = {}
        self.shuffle_ties = shuffle_ties
        if support_target_order not in self.SUPPORT_TARGET_ORDERS:
            raise ValueError(
                "support_target_order must be 'lowest_first', "
                "'least_connected', 'random', or "
                "'furthest_from_removed'."
            )
        self.support_target_order = support_target_order

        self.rigidity = TrussRigidityChecker(
            truss,
            max_cache_entries=rigidity_cache_size,
        )

        self.final_node = None
        self.search_stop_reason = None
        self.search_expansions = 0
        self.search_attempted_transitions = 0
        self.search_enqueued_candidates = 0
        self.search_backtracks = 0
        self._support_grippers_override = (
            tuple(support_grippers)
            if support_grippers is not None
            else None
        )

        self.forbidden_transitions = set(
            forbidden_transitions or []
        )

        self.rod_neighbors = {
            rod_id: set()
            for rod_id in self.truss.elements
        }

        for rod_1, rod_2 in self.truss.couplers:
            if (
                rod_1 in self.rod_neighbors
                and rod_2 in self.rod_neighbors
            ):
                self.rod_neighbors[rod_1].add(rod_2)
                self.rod_neighbors[rod_2].add(rod_1)

        self.virtual_supports = tuple(
            f"support_{index + 1}"
            for index in range(max_supports)
        )

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
        if self._support_grippers_override is not None:
            return self._support_grippers_override

        if self.builder is not None:
            return tuple(self.builder.support_grippers)

        return self.virtual_supports
    
    @staticmethod
    def structural_state_key(state, supported):
        """
        Structural state including which physical support robot holds which rod.
        """
        state = frozenset(state)

        active_supports = frozenset(
            (gripper, rod_id)
            for gripper, rod_id in supported.items()
            if rod_id in state
        )

        return (
            state,
            active_supports,
        )
        
    def search_state_key(self, node):
        """
        Complete search state.

        When RAI is enabled, two nodes with the same rods and supports
        may still be different because the robots have different joint
        configurations.
        """
        structural_key = self.structural_state_key(
            node.state,
            node.supported,
        )

        if node.q is None:
            q_key = None

        else:
            q_key = tuple(
                np.round(
                    np.asarray(
                        node.q,
                        dtype=float,
                    ),
                    decimals=3,
                )
            )

        return (
            *structural_key,
            q_key,
        )


    @classmethod
    def structural_transition_key(
        cls,
        state,
        supported,
        candidate_rod,
    ):
        return (
            cls.structural_state_key(
                state,
                supported,
            ),
            int(candidate_rod),
        )


    @classmethod
    def structural_transition_key_from_step(
        cls,
        structural_step,
    ):
        return cls.structural_transition_key(
            state=structural_step.rods_before,
            supported=structural_step.supports_before,
            candidate_rod=structural_step.rod_id,
        )

    def rod_midpoint_height(self, rod_id):
        """Return the rod midpoint height used by height-based heuristics."""
        n1, n2 = self.truss.elements[rod_id]
        return 0.5 * (self.truss.nodes[n1][2] + self.truss.nodes[n2][2])
    
    def is_supported_candidate(self, node, rod_id):
        return rod_id in node.supported.values()

    def priority_tie_breaker(self, rod_id):
        if self.shuffle_ties:
            return float(self.rng.random())

        return rod_id

    def distance_from_ground(self, active_rods):
        """Return coupler-graph distances from active grounded rods."""
        active = set(active_rods)
        distances = {}
        queue = deque()

        for rod_id in set(self.truss.grounded_rods) & active:
            distances[rod_id] = 0
            queue.append(rod_id)

        while queue:
            rod_id = queue.popleft()

            for neighbour in self.rod_neighbors[rod_id] & active:
                if neighbour not in distances:
                    distances[neighbour] = distances[rod_id] + 1
                    queue.append(neighbour)

        return distances

    def support_context_after_removal(
        self,
        node,
        candidate_rod,
    ) -> RemovalSupportContext:
        """Split existing supports into continuing, releasable, and free."""
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

        free_supports = [
            support
            for support in self.helper_grippers
            if support not in continuing_supports
        ]

        return RemovalSupportContext(
            current_state=current_state,
            new_state=new_state,
            continuing_supports=continuing_supports,
            releasable_supports=releasable_supports,
            free_supports=free_supports,
            continuing_supported_rods=set(
                continuing_supports.values()
            ),
            candidate_is_supported=bool(releasable_supports),
            old_support_gripper=next(iter(releasable_supports), None),
        )
    
    def evaluate_supports_after_removal(
        self,
        node,
        candidate_rod,
        support_context=None,
        initial_result=None,
    ) -> SupportEvaluation:
        """Evaluate and cache the support outcome of one removal."""
        cache_key = self.structural_transition_key(
            state=node.state,
            supported=node.supported,
            candidate_rod=candidate_rod,
        )

        if cache_key in self.support_evaluations:
            return self.support_evaluations[cache_key]
        if support_context is None:
            support_context = self.support_context_after_removal(
                node,
                candidate_rod,
            )

        rigidity_result = initial_result

        if rigidity_result is None:
            rigidity_result = self.rigidity.check(
                support_context.new_state,
                supported_rods=support_context.continuing_supported_rods,
            )

        affected_rods = []

        if (
            not rigidity_result.is_rigid
            and support_context.free_supports
        ):
            affected_rods, rigidity_result = (
                self.rigidity.choose_support_targets(
                    active_rods=support_context.new_state,
                    already_supported=(
                        support_context.continuing_supported_rods
                    ),
                    max_targets=len(support_context.free_supports),
                    contextual_key=lambda rod_id, _supported_rods: (
                        self.support_target_priority(
                            rod_id,
                            removed_rod=candidate_rod,
                            active_rods=support_context.new_state,
                        )
                    ),
                    initial_result=rigidity_result,
                    return_result=True,
                )
            )

        added_supports = {
            support: rod_id
            for support, rod_id in zip(
                support_context.free_supports,
                affected_rods,
            )
        }

        supports_after = dict(support_context.continuing_supports)
        supports_after.update(added_supports)

        result = SupportEvaluation(
            feasible=rigidity_result.is_rigid,
            rigidity_result=rigidity_result,
            supports_after=supports_after,
            added_supports=added_supports,
        )

        self.support_evaluations[cache_key] = result
        return result
        
    def random_candidate_order(self, node, candidate_rods):
        """Return a fixed random candidate ranking for this structural state."""
        state_key = self.structural_state_key(
            node.state,
            node.supported,
        )

        if state_key not in self.baseline_candidate_orders:
            candidates = sorted(candidate_rods)
            payload = repr(
                (
                    self.random_seed,
                    tuple(sorted(node.state)),
                    tuple(
                        sorted(
                            (gripper, rod_id)
                            for gripper, rod_id in node.supported.items()
                            if rod_id in node.state
                        )
                    ),
                    tuple(candidates),
                )
            ).encode("utf-8")
            state_seed = int.from_bytes(
                hashlib.blake2b(
                    payload,
                    digest_size=8,
                ).digest(),
                "little",
            )
            state_rng = np.random.default_rng(state_seed)
            state_rng.shuffle(candidates)
            self.baseline_candidate_orders[state_key] = {
                rod_id: rank
                for rank, rod_id in enumerate(candidates)
            }

        return self.baseline_candidate_orders[state_key]

    def removal_priority(
        self,
        node,
        rod_id,
        ground_distances=None,
        actual_support_result=None,
        minimum_new_support_count=0,
        tie_breaker=None,
    ):
        """Return a priority tuple; lower values are preferred."""

        if tie_breaker is None:
            tie_breaker = self.priority_tie_breaker(rod_id)

        # if ground_distances is None:
        #     ground_distances = self.distance_from_ground(node.state)

        # distance = ground_distances.get(
        #     rod_id,
        #     len(node.state) + 1,
        # )

        connection_count = len(
            self.rod_neighbors[rod_id] & node.state
        )

        supported_rank = (
            0 if self.is_supported_candidate(node, rod_id) else 1
        )

        if self.strategy_name == "baseline":
            return (
                len(node.state),
                tie_breaker,
            )

        if self.strategy_name == "default":
            return (
                len(node.state),
                supported_rank,
                connection_count,
                -self.rod_midpoint_height(rod_id),
                tie_breaker,
            )

        if self.strategy_name == "highest_first":
            return (
                len(node.state),
                -self.rod_midpoint_height(rod_id),
                tie_breaker,
            )

        if self.strategy_name == "reduced_overall_supports":
            if actual_support_result is None:
                raise ValueError(
                    "reduced_overall_supports requires an evaluated "
                    "support outcome."
                )

            # reduce the total number of supports used in the entire assembly sequence
            # secondary tie-breakers: maximize the number of rods removed, minimize the number of current supports
            return (
                node.support_additions_so_far
                + actual_support_result.new_support_count,
                len(node.state),
                actual_support_result.support_count,
                tie_breaker,
            )

        if self.strategy_name == "fast_reduce_support":
            if actual_support_result is None:
                # This candidate has not had its full support search yet. Build
                # a cheap lower-bound priority so promising candidates can be
                # evaluated lazily instead of evaluating every possible removal.
                
                # Existing supports on rods other than the candidate continue
                # after removal. A support on the candidate itself is excluded,
                # because removing that rod releases the support.
                continuing_support_count = sum(
                    supported_rod != rod_id
                    for supported_rod in node.supported.values()
                )

                return (
                    # Prefer states that have already removed more rods.
                    len(node.state),
                    # Estimated total supports after removal: assuming unavoidable continuing supports plus the known lower bound on new ones.
                    continuing_support_count + minimum_new_support_count,
                    # Among equal totals, prefer fewer newly placed supports.
                    minimum_new_support_count,
                    # Then prefer rods connected to fewer remaining rods.
                    connection_count,
                    # Then prefer physically higher rods.
                    -self.rod_midpoint_height(rod_id),
                    # Final deterministic or seeded-random tie breaker.
                    tie_breaker,
                )

            # The expensive support evaluation has now been performed. Replace
            # the lower bounds above with the actual number of supports that
            # remain assigned and the actual number of newly assigned supports.
            return (
                len(node.state),
                actual_support_result.support_count,
                actual_support_result.new_support_count,
                connection_count,
                -self.rod_midpoint_height(rod_id),
                tie_breaker,
            )


        raise ValueError(
            f"Unknown search strategy: {self.strategy_name}"
        )
        

    def removal_candidates_with_priorities(self, node):
        """Return candidate removals with initial heap priorities."""

        candidates = list(node.state)
       
        random_order = self.random_candidate_order(
            node,
            candidates,
        )

        ranked_candidates = []

        for rod_id in candidates:
            tie_breaker = (
                random_order[rod_id]
                if random_order is not None
                else rod_id
            )
            actual_support_result = None

            if self.strategy_name == "reduced_overall_supports":
                actual_support_result = (
                    self.evaluate_supports_after_removal(
                        node,
                        rod_id,
                    )
                )

                if not actual_support_result.feasible:
                    continue

            priority = self.removal_priority(
                node,
                rod_id,
                actual_support_result=actual_support_result,
                tie_breaker=tie_breaker,
            )

            ranked_candidates.append(
                (priority, rod_id)
            )

        return ranked_candidates

    def rod_physical_distance(self, rod_1, rod_2):
        """Return and cache the Euclidean distance between rod centers."""
        cache_key = tuple(sorted((rod_1, rod_2)))
        cached_distance = self._rod_physical_distance_cache.get(cache_key)

        if cached_distance is not None:
            return cached_distance

        node_1_start, node_1_end = self.truss.elements[rod_1]
        node_2_start, node_2_end = self.truss.elements[rod_2]
        center_1 = 0.5 * (
            np.asarray(self.truss.nodes[node_1_start], dtype=float)
            + np.asarray(self.truss.nodes[node_1_end], dtype=float)
        )
        center_2 = 0.5 * (
            np.asarray(self.truss.nodes[node_2_start], dtype=float)
            + np.asarray(self.truss.nodes[node_2_end], dtype=float)
        )
        distance = float(np.linalg.norm(center_1 - center_2))
        self._rod_physical_distance_cache[cache_key] = distance
        return distance

    def support_target_priority(
        self,
        rod_id,
        removed_rod=None,
        active_rods=None,
    ):
        """Order structurally valid support targets by the configured rule."""
        # TrussRigidityChecker sorts target keys with reverse=True.
        if self.support_target_order == "random":
            return (float(self.rng.random()),)

        if self.support_target_order == "least_connected":
            active_rods = set(active_rods or ())
            connection_count = len(
                self.rod_neighbors[rod_id] & active_rods
            )
            return (
                -connection_count,
                self.priority_tie_breaker(rod_id),
            )

        if self.support_target_order == "furthest_from_removed":
            distance = (
                self.rod_physical_distance(rod_id, removed_rod)
                if removed_rod is not None
                else float("-inf")
            )
            return (distance, self.priority_tie_breaker(rod_id))

        return (-self.rod_midpoint_height(rod_id),)

    # greedy backward search
    def backward_search(
        self,
        capture_key=None,
        max_runtime=1800.0,
        max_expansions_without_progress=20000,
        initial_supported=None,
        initial_support_q=None, #joint configuration of the support grippers
    ):
        
        """
        Run the backward search
        """
        if max_runtime is not None and max_runtime <= 0:
            raise ValueError("max_runtime must be positive or None.")
        if (
            max_expansions_without_progress is not None
            and max_expansions_without_progress <= 0
        ):
            raise ValueError(
                "max_expansions_without_progress must be positive or None."
            )

        # backward search visualization
        hotkey_context = (
            TerminalHotkey(capture_key)
            if capture_key is not None
            else nullcontext(None)
        )
        
        # initial state: all rods in final position
        initial_state = frozenset(self.truss.elements.keys())

        initial_supported = dict(initial_supported or {})
        initial_support_q = dict(initial_support_q or {})

        unknown_support_rods = (
            set(initial_supported.values()) - set(initial_state)
        )
        
        if unknown_support_rods:
            raise ValueError(
                "Initial supports reference rods that are not active: "
                f"{sorted(unknown_support_rods)}"
            )

        unknown_supports = (
            set(initial_supported) - set(self.helper_grippers)
        )
        if unknown_supports:
            raise ValueError(
                "Initial supports reference unknown support grippers: "
                f"{sorted(unknown_supports)}"
            )

        if len(initial_supported) > len(self.helper_grippers):
            raise ValueError(
                "Initial support assignments exceed available supports."
            )
        
        unsupported_final_result = self.rigidity.check(
            initial_state,
            supported_rods=set(),
        )

        self.final_structure_is_rigid_without_supports = (
            unsupported_final_result.is_rigid
        )

        if initial_supported:
            initial_result = self.rigidity.check(
                initial_state,
                supported_rods=set(initial_supported.values()),
            )
        else:
            initial_result = unsupported_final_result

        if not initial_result.is_rigid:
            print(
                "\nWarning: The initial truss configuration is not rigid with the inherited support assignments."
            )
        elif not unsupported_final_result.is_rigid:
            if initial_supported:
                support_word = (
                    "support"
                    if len(initial_supported) == 1
                    else "supports"
                )
                print(
                    "\nPrefix scaffold starts with "
                    f"{len(initial_supported)} inherited {support_word}; "
                    "it is not rigid without them."
                )
            else:
                print(
                    "\nWarning: The final truss configuration is not rigid "
                    "without supports. Required supports will remain in the "
                    "final visualization frame."
                )
            # input(
            #     "Press Enter to continue the search anyway, "
            #     "or Ctrl+C to abort..."
            # )

        open_list = []
        counter = 0

        # initialize search

        initial_node = SearchNode(
            state=initial_state,
            sequence=[],
            q=(
                self.builder.C.getJointState().copy()
                if self.builder is not None
                else None
            ),
            supported=initial_supported,
            support_q=initial_support_q,
            records=[],
            structural_steps=[],
        )

        self.final_node = initial_node
        self.search_stop_reason = None
        self.search_expansions = 0
        self.search_attempted_transitions = 0
        self.search_enqueued_candidates = 0
        self.search_backtracks = 0

        start_time = time.monotonic()
        best_node = initial_node
        best_remaining = len(initial_state)
        last_progress_expansion = 0

        visited = {
            self.search_state_key(initial_node)
        }
        best_state_support_additions = {
            self.search_state_key(initial_node): 0
        }
        
        attempted_transitions = set()

        # helper function to enqueue candidate removals for a given search node
        def enqueue_removals(node):
            nonlocal counter

            for priority, candidate_rod in (
                self.removal_candidates_with_priorities(node)
            ):
                self.search_enqueued_candidates += 1
                heapq.heappush(
                    open_list,
                    (
                        priority,
                        counter,
                        node,
                        candidate_rod,
                        None,
                        None,
                    ),
                )
                counter += 1

        enqueue_removals(initial_node)

        with hotkey_context as hotkey:
            capture_enabled = (
                capture_key
                and hotkey is not None
                and getattr(hotkey, "enabled", True)
            )
            print(
                f"Search running. Press {capture_key.upper()} to start/stop capture."
                if capture_enabled
                else "Search running."
            )
            self.process_debug_hotkey(hotkey)

            # loop until solution found or open list exhausted
            while open_list:
                self.process_debug_hotkey(hotkey)

                # abort if limits are reached
                elapsed = time.monotonic() - start_time
                if max_runtime is not None and elapsed >= max_runtime:
                    self.search_stop_reason = "runtime_limit"
                    self.final_node = best_node
                    print("Search runtime limit reached.")
                    print(f"Deepest state: {best_remaining} rods remaining.")
                    return None

                if (
                    max_expansions_without_progress is not None
                    and self.search_expansions - last_progress_expansion
                    >= max_expansions_without_progress
                ):
                    self.search_stop_reason = "stagnation_limit"
                    self.final_node = best_node
                    print("Search stopped because no deeper state was found.")
                    print(f"Deepest state: {best_remaining} rods remaining.")
                    return None

                (
                    priority,
                    candidate_counter,
                    node,
                    candidate_rod,
                    support_evaluation,
                    initial_rigidity_result,
                ) = heapq.heappop(open_list)

                if self.strategy_name == "reduced_overall_supports":
                    node_state_key = self.search_state_key(node)

                    if (
                        node.support_additions_so_far
                        != best_state_support_additions.get(node_state_key)
                    ):
                        continue

                # Reject candidate rods that are no longer present in the current state.
                if candidate_rod not in node.state:
                    continue

                # Structural transition identity. This is used only for transitions
                # explicitly forbidden independent of robot configuration.
                structural_transition_key = (
                    self.structural_transition_key(
                        state=node.state,
                        supported=node.supported,
                        candidate_rod=candidate_rod,
                    )
                )

                if (
                    structural_transition_key
                    in self.forbidden_transitions
                ):
                    continue

                if (
                    self.strategy_name == "fast_reduce_support"
                    and support_evaluation is None
                ):
                    # Tighten the optimistic priority in two stages: first
                    # test continuing supports, then search for new targets.
                    support_context = self.support_context_after_removal(
                        node,
                        candidate_rod,
                    )
                    support_evaluation = self.support_evaluations.get(
                        structural_transition_key
                    )

                    if (
                        support_evaluation is None
                        and initial_rigidity_result is None
                    ):
                        initial_rigidity_result = self.rigidity.check(
                            support_context.new_state,
                            supported_rods=(
                                support_context.continuing_supported_rods
                            ),
                        )

                        if (
                            not initial_rigidity_result.is_rigid
                            and support_context.free_supports
                        ):
                            refined_priority = self.removal_priority(
                                node,
                                candidate_rod,
                                minimum_new_support_count=1,
                                tie_breaker=priority[-1],
                            )
                            heapq.heappush(
                                open_list,
                                (
                                    refined_priority,
                                    candidate_counter,
                                    node,
                                    candidate_rod,
                                    None,
                                    initial_rigidity_result,
                                ),
                            )
                            continue

                    if support_evaluation is None:
                        support_evaluation = (
                            self.evaluate_supports_after_removal(
                                node,
                                candidate_rod,
                                support_context=support_context,
                                initial_result=initial_rigidity_result,
                            )
                        )

                    if not support_evaluation.feasible:
                        continue

                    refined_priority = self.removal_priority(
                        node,
                        candidate_rod,
                        actual_support_result=support_evaluation,
                        tie_breaker=priority[-1],
                    )
                    heapq.heappush(
                        open_list,
                        (
                            refined_priority,
                            candidate_counter,
                            node,
                            candidate_rod,
                            support_evaluation,
                            None,
                        ),
                    )
                    continue

                # The same structural transition may behave differently when reached
                # with a different robot configuration.
                physical_transition_key = (
                    self.search_state_key(node),
                    int(candidate_rod),
                )

                if physical_transition_key in attempted_transitions:
                    continue

                attempted_transitions.add(
                    physical_transition_key
                )
                self.search_attempted_transitions += 1

                new_state = frozenset(
                    node.state - {candidate_rod}
                )

                if len(new_state) != len(node.state) - 1:
                    raise RuntimeError(
                        f"Removal of rod {candidate_rod} did not reduce "
                        "the state by exactly one rod."
                    )

                self.search_expansions += 1

                feasible, result = self.is_removal_feasible(
                    node,
                    candidate_rod,
                    support_evaluation=support_evaluation,
                )

                # If the hotkey was pressed while the feasibility calculation was
                # running, apply it before deciding whether this result is captured.
                self.process_debug_hotkey(hotkey)
                
                # Record both successful and unsuccessful tested configurations.
                if self.debug_capture_active and result is not None:
                    structural_step = result.get("structural_step")

                    if structural_step is not None:
                        self.debug_capture_steps.append(
                            structural_step
                        )

                if not feasible:
                    self.search_backtracks += 1
                    continue

                motion_record = result["motion_record"]

                # create a new search node for the resulting state after removing the candidate rod
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
                    support_additions_so_far=(
                        node.support_additions_so_far
                        + len(result["structural_step"].added_supports)
                    ),
                )
                
                state_key = self.search_state_key(
                    new_node
                )

                if self.strategy_name == "reduced_overall_supports":
                    previous_cost = best_state_support_additions.get(
                        state_key
                    )

                    if (
                        previous_cost is not None
                        and previous_cost
                        <= new_node.support_additions_so_far
                    ):
                        continue

                    best_state_support_additions[state_key] = (
                        new_node.support_additions_so_far
                    )
                else:
                    if state_key in visited:
                        continue

                    visited.add(state_key)

                if len(new_state) < best_remaining:
                    best_node = new_node
                    best_remaining = len(new_state)
                    last_progress_expansion = self.search_expansions
                    self.final_node = best_node
                    # print(
                    #     f"New deepest state: {best_remaining} rods remaining "
                    #     f"after {self.search_expansions} attempted transitions."
                    # )

                if len(new_state) == 0:
                    self.final_node = new_node
                    self.search_stop_reason = "complete"
                    return new_node.sequence

                # # debug stopping condition
                # if len(new_node.sequence) >= 10:
                #     self.final_node = new_node
                #     return new_node.sequence

                enqueue_removals(new_node)

        self.search_stop_reason = "open_list_exhausted"
        self.final_node = best_node
        print("Search exhausted all queued configurations.")
        print(f"Deepest state: {best_remaining} rods remaining.")
        return None
    
    def is_removal_feasible(
        self,
        node,
        candidate_rod,
        support_evaluation=None,
    ):
        support_context = self.support_context_after_removal(
            node,
            candidate_rod,
        )

        def make_structural_step(
            rigidity_result,
            supported_after,
            added_supports,
        ):
            return StructuralRemovalStep(
                rod_id=candidate_rod,
                rods_before=frozenset(support_context.current_state),
                rods_after=frozenset(support_context.new_state),
                supports_before=dict(node.supported),
                supports_after=dict(supported_after),
                added_supports=dict(added_supports),
                released_supports=dict(support_context.releasable_supports),
                rank_after=rigidity_result.rank,
                dof_after=rigidity_result.dof,
            )

        if support_evaluation is None:
            support_evaluation = self.evaluate_supports_after_removal(
                node,
                candidate_rod,
                support_context=support_context,
            )
        rigidity_result = support_evaluation.rigidity_result
        new_support_assignments = dict(
            support_evaluation.added_supports
        )
        next_supported = dict(support_evaluation.supports_after)

        structural_step = make_structural_step(
            rigidity_result=rigidity_result,
            supported_after=next_supported,
            added_supports=new_support_assignments,
        )

        if not support_evaluation.feasible:
            # print(
            #     f"Removing rod {candidate_rod} is structurally infeasible: "
            #     f"rank {rigidity_result.rank}/{rigidity_result.dof}, "
            #     f"supports {sorted(next_supported.values())}."
            # )
            return False, {
                "structural_step": structural_step,
            }

        # print(
        #     f"Remove rod {candidate_rod}: "
        #     f"remaining supports = {next_supported or 'none'}"
        # )

        # ---------------------------------------------------------
        # Optional motion-planning validation
        # ---------------------------------------------------------

        if self.builder is None:
            q_final = None
            support_q = {}
            motion_record = None

        else:
            motion_result = self.builder.try_remove_and_commit_rod(
                current_state=support_context.current_state,
                new_state=support_context.new_state,
                rod_id=candidate_rod,
                q_start=node.q,
                supported=node.supported,
                support_q=node.support_q,
                candidate_is_supported=(
                    support_context.candidate_is_supported
                ),
                old_support_gripper=support_context.old_support_gripper,
                continuing_supports=support_context.continuing_supports,
                releasable_supports=support_context.releasable_supports,
                new_support_assignments=new_support_assignments,
                use_rrt=False,
                do_shortcut=False,
            )

            if motion_result is None:
                return False, {
                    "structural_step": structural_step,
                }

            q_final = motion_result["q_final"]
            next_supported = motion_result["supported"]
            support_q = motion_result["support_q"]
            motion_record = motion_result["record"]

        structural_step = StructuralRemovalStep(
            rod_id=candidate_rod,
            rods_before=frozenset(support_context.current_state),
            rods_after=frozenset(support_context.new_state),
            supports_before=dict(node.supported),
            supports_after=dict(next_supported),
            added_supports=dict(new_support_assignments),
            released_supports=dict(support_context.releasable_supports),
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
