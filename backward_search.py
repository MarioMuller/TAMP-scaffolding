import hashlib

from shapely import node
from truss import Truss
from collections import deque
from itertools import combinations
import heapq
import io
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
        optimal_objective="support_moves",
    ):
        self.truss = truss
        self.builder = builder
        self.max_supports = max_supports
        self.strategy_name = strategy_name
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(random_seed)
        self.baseline_candidate_orders = {}
        self.support_evaluations = {}
        self.shuffle_ties = shuffle_ties
        if optimal_objective not in {
            "peak",
            "support_steps",
            "support_moves",
        }:
            raise ValueError(
                "optimal_objective must be 'support_moves', "
                "'support_steps', or 'peak'."
            )
        self.optimal_objective = optimal_objective

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
        self.optimal_support_peak = None
        self.optimal_support_steps = None
        self.optimal_support_moves = None
        self.best_support_peak = None
        self.best_support_steps = None
        self.best_support_moves = None
        self.optimality_proven = False

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

    def heuristic(self, rod_id):
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
                    key=lambda rod_id: self.support_target_priority(
                        rod_id,
                        removed_rod=candidate_rod,
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
        initial_rigidity_result=None,
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
                # -distance,
                -self.heuristic(rod_id),
                tie_breaker,
            )

        if self.strategy_name == "highest_first":
            return (
                len(node.state),
                -self.heuristic(rod_id),
                tie_breaker,
            )

        if self.strategy_name == "lowest_first":
            return (
                len(node.state),
                self.heuristic(rod_id),
                tie_breaker,
            )

        if self.strategy_name == "rankbased":
            if initial_rigidity_result is None:
                raise ValueError(
                    "rankbased requires the post-removal rigidity result."
                )

            return (
                len(node.state),
                -initial_rigidity_result.rank,
                tie_breaker,
            )

        if self.strategy_name in {
            "full_reduce_support",
            "full_reduce_support_no_depth",
            "reduced_overall_supports",
        }:
            if actual_support_result is None:
                raise ValueError(
                    f"{self.strategy_name} requires an evaluated support "
                    "outcome."
                )

            depth_priority = (
                ()
                if self.strategy_name == "full_reduce_support_no_depth"
                else (len(node.state),)
            )

            if self.strategy_name == "reduced_overall_supports":
                return (
                    node.support_additions_so_far
                    + actual_support_result.new_support_count,
                    len(node.state),
                    actual_support_result.support_count,
                    actual_support_result.new_support_count,
                    tie_breaker,
                )

            return (
                *depth_priority,
                actual_support_result.support_count,
                actual_support_result.new_support_count,
                tie_breaker,
            )

        if self.strategy_name == "fast_reduce_support":
            if actual_support_result is None:
                # Continuing supports are unavoidable. Additional supports
                # start at an optimistic lower bound until evaluated.
                continuing_support_count = sum(
                    supported_rod != rod_id
                    for supported_rod in node.supported.values()
                )

                return (
                    len(node.state),
                    continuing_support_count
                    + minimum_new_support_count,
                    minimum_new_support_count,
                    supported_rank,
                    connection_count,
                    -self.heuristic(rod_id),
                    tie_breaker,
                )

            return (
                len(node.state),
                actual_support_result.support_count,
                actual_support_result.new_support_count,
                supported_rank,
                connection_count,
                -self.heuristic(rod_id),
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
            initial_rigidity_result = None

            if self.strategy_name == "rankbased":
                support_context = self.support_context_after_removal(
                    node,
                    rod_id,
                )
                initial_rigidity_result = self.rigidity.check(
                    support_context.new_state,
                    supported_rods=(
                        support_context.continuing_supported_rods
                    ),
                )

            if self.strategy_name in {
                "full_reduce_support",
                "full_reduce_support_no_depth",
                "reduced_overall_supports",
            }:
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
                initial_rigidity_result=initial_rigidity_result,
                tie_breaker=tie_breaker,
            )

            ranked_candidates.append(
                (priority, rod_id)
            )

        return ranked_candidates

    def support_target_priority(self, rod_id, removed_rod=None):
        """Prefer support targets near the removed rod, then higher rods."""
        local_rank = 0

        # if removed_rod is not None:
        #     direct_neighbours = self.rod_neighbors.get(
        #         removed_rod,
        #         set(),
        #     )

        #     if rod_id in direct_neighbours:
        #         local_rank = 2
        #     elif any(
        #         rod_id in self.rod_neighbors.get(neighbour, set())
        #         for neighbour in direct_neighbours
        #     ):
        #         local_rank = 1

        return (
            # local_rank,
            -self.heuristic(rod_id),
        )

    def minimum_support_outcome(
        self,
        active_rods,
        support_limit,
        outcome_cache,
        deadline=None,
    ):
        """Find one minimum-cardinality support set for an active state."""
        active_rods = frozenset(active_rods)
        cache_key = (active_rods, support_limit)

        if cache_key in outcome_cache:
            return outcome_cache[cache_key]

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError

        initial_result = self.rigidity.check(
            active_rods,
            supported_rods=(),
        )

        if initial_result.is_rigid:
            outcome_cache[cache_key] = (frozenset(), initial_result)
            return outcome_cache[cache_key]

        frontier = {frozenset(): initial_result}
        evaluated = {frozenset()}

        for _ in range(min(support_limit, len(active_rods))):
            next_frontier = {}

            for supported_rods, current_result in frontier.items():
                candidates = [
                    rod_id
                    for rod_id in sorted(active_rods)
                    if (
                        rod_id not in supported_rods
                        and current_result.statuses[rod_id]
                        != ElementStatus.fixed
                    )
                ]

                for rod_id in candidates:
                    next_supported_rods = frozenset(
                        set(supported_rods) | {rod_id}
                    )

                    if next_supported_rods in evaluated:
                        continue

                    evaluated.add(next_supported_rods)

                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError

                    rigidity_result = self.rigidity.check(
                        active_rods,
                        supported_rods=next_supported_rods,
                    )

                    if rigidity_result.is_rigid:
                        outcome_cache[cache_key] = (
                            next_supported_rods,
                            rigidity_result,
                        )
                        return outcome_cache[cache_key]

                    next_frontier[next_supported_rods] = rigidity_result

            frontier = next_frontier

            if not frontier:
                break

        outcome_cache[cache_key] = None
        return None

    def support_mapping_for_rods(self, current_supports, supported_rods):
        """Preserve existing holds when assigning structural support rods."""
        supported_rods = set(supported_rods)
        next_supports = {
            support: rod_id
            for support, rod_id in current_supports.items()
            if rod_id in supported_rods
        }
        assigned_rods = set(next_supports.values())
        free_supports = [
            support
            for support in self.helper_grippers
            if support not in next_supports
        ]

        for support, rod_id in zip(
            free_supports,
            sorted(supported_rods - assigned_rods),
        ):
            next_supports[support] = rod_id

        return next_supports

    def optimal_candidate_order(self, active_rods):
        ordering_node = SearchNode(
            state=frozenset(active_rods),
            supported={},
        )
        random_order = self.random_candidate_order(
            ordering_node,
            active_rods,
        )
        return sorted(
            active_rods,
            key=lambda rod_id: random_order[rod_id],
        )

    def optimal_successor_node(
        self,
        node,
        candidate_rod,
        supported_rods,
        rigidity_result,
    ):
        new_state = frozenset(node.state - {candidate_rod})
        next_supported = self.support_mapping_for_rods(
            node.supported,
            supported_rods,
        )
        added_supports = {
            support: rod_id
            for support, rod_id in next_supported.items()
            if node.supported.get(support) != rod_id
        }
        released_supports = {
            support: rod_id
            for support, rod_id in node.supported.items()
            if next_supported.get(support) != rod_id
        }
        structural_step = StructuralRemovalStep(
            rod_id=candidate_rod,
            rods_before=frozenset(node.state),
            rods_after=new_state,
            supports_before=dict(node.supported),
            supports_after=dict(next_supported),
            added_supports=dict(added_supports),
            released_supports=dict(released_supports),
            rank_after=rigidity_result.rank,
            dof_after=rigidity_result.dof,
        )

        return SearchNode(
            state=new_state,
            sequence=node.sequence + [candidate_rod],
            q=None,
            supported=next_supported,
            support_q={},
            records=[],
            structural_steps=node.structural_steps + [structural_step],
            support_additions_so_far=(
                node.support_additions_so_far + len(added_supports)
            ),
        )

    def minimum_peak_plan(
        self,
        initial_node,
        outcome_cache,
        randomize=True,
        deadline=None,
    ):
        """Prove the minimum simultaneous support count and return one plan."""
        initial_peak = len(initial_node.supported)

        for support_limit in range(
            initial_peak,
            len(self.helper_grippers) + 1,
        ):
            failed_states = set()
            print(f"Testing support peak <= {support_limit}.")

            def find_plan(state):
                state = frozenset(state)

                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError

                if not state:
                    return []

                if state in failed_states:
                    return None

                candidate_rods = (
                    self.optimal_candidate_order(state)
                    if randomize
                    else sorted(state)
                )

                for candidate_rod in candidate_rods:
                    self.search_expansions += 1
                    self.search_attempted_transitions += 1
                    new_state = frozenset(state - {candidate_rod})
                    outcome = self.minimum_support_outcome(
                        new_state,
                        support_limit=support_limit,
                        outcome_cache=outcome_cache,
                        deadline=deadline,
                    )

                    if outcome is None:
                        self.search_backtracks += 1
                        continue

                    self.search_enqueued_candidates += 1
                    supported_rods, rigidity_result = outcome
                    suffix = find_plan(new_state)

                    if suffix is not None:
                        return [
                            (
                                candidate_rod,
                                supported_rods,
                                rigidity_result,
                            )
                        ] + suffix

                failed_states.add(state)
                return None

            plan = find_plan(initial_node.state)

            if plan is not None:
                return support_limit, plan

        return None, None

    def build_node_from_optimal_plan(self, initial_node, plan):
        node = initial_node

        for candidate_rod, supported_rods, rigidity_result in plan:
            node = self.optimal_successor_node(
                node,
                candidate_rod,
                supported_rods,
                rigidity_result,
            )

        return node

    def rigid_support_outcomes(
        self,
        active_rods,
        outcome_cache,
        deadline=None,
    ):
        """Return every rigid support-rod set within the support limit."""
        active_rods = frozenset(active_rods)

        if active_rods in outcome_cache:
            return outcome_cache[active_rods]

        outcomes = []
        maximum_supports = min(
            len(self.helper_grippers),
            len(active_rods),
        )

        for support_count in range(maximum_supports + 1):
            for supported_tuple in combinations(
                sorted(active_rods),
                support_count,
            ):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError

                supported_rods = frozenset(supported_tuple)
                rigidity_result = self.rigidity.check(
                    active_rods,
                    supported_rods=supported_rods,
                )

                if rigidity_result.is_rigid:
                    outcomes.append((supported_rods, rigidity_result))

        outcome_cache[active_rods] = tuple(outcomes)
        return outcome_cache[active_rods]

    def support_move_outcomes(
        self,
        active_rods,
        continuing_supported_rods,
        outcome_cache,
        deadline=None,
    ):
        """Return useful rigid outcomes for placement-count optimization."""
        continuing_supported_rods = frozenset(
            continuing_supported_rods
        ) & frozenset(active_rods)
        outcomes = self.rigid_support_outcomes(
            active_rods,
            outcome_cache=outcome_cache,
            deadline=deadline,
        )
        rigid_sets = {
            supported_rods
            for supported_rods, _ in outcomes
        }
        useful_outcomes = []

        for supported_rods, rigidity_result in outcomes:
            newly_supported = (
                supported_rods - continuing_supported_rods
            )

            # A newly placed support that is not needed yet can always be
            # deferred. Deferring has the same move cost and fewer occupied
            # support-steps, so such an outcome is dominated.
            if any(
                supported_rods - {rod_id} in rigid_sets
                for rod_id in newly_supported
            ):
                continue

            useful_outcomes.append((supported_rods, rigidity_result))

        useful_outcomes.sort(
            key=lambda item: (
                len(item[0] - continuing_supported_rods),
                len(item[0]),
                tuple(sorted(item[0])),
            )
        )
        return useful_outcomes

    def support_move_incumbent_plan(
        self,
        initial_node,
        outcome_cache,
        deadline=None,
    ):
        """Find a complete low-move plan to bound the exact search."""
        failed_states = set()

        def find_plan(state, supported_rods):
            state = frozenset(state)
            supported_rods = frozenset(supported_rods) & state
            state_key = (state, supported_rods)

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError
            if not state:
                return []
            if state_key in failed_states:
                return None

            transitions = []

            for candidate_rod in self.optimal_candidate_order(state):
                new_state = frozenset(state - {candidate_rod})
                continuing_supported_rods = (
                    supported_rods - {candidate_rod}
                )
                outcomes = self.support_move_outcomes(
                    new_state,
                    continuing_supported_rods,
                    outcome_cache=outcome_cache,
                    deadline=deadline,
                )

                for outcome in outcomes:
                    next_supported_rods, _ = outcome
                    transitions.append(
                        (
                            (
                                len(
                                    next_supported_rods
                                    - continuing_supported_rods
                                ),
                                len(next_supported_rods),
                                self.heuristic(candidate_rod),
                            ),
                            candidate_rod,
                            outcome,
                        )
                    )

            transitions.sort(key=lambda item: item[0])

            for _, candidate_rod, outcome in transitions:
                self.search_expansions += 1
                self.search_attempted_transitions += 1
                next_supported_rods, rigidity_result = outcome
                suffix = find_plan(
                    state - {candidate_rod},
                    next_supported_rods,
                )

                if suffix is not None:
                    self.search_enqueued_candidates += 1
                    return [
                        (
                            candidate_rod,
                            next_supported_rods,
                            rigidity_result,
                        )
                    ] + suffix

                self.search_backtracks += 1

            failed_states.add(state_key)
            return None

        return find_plan(
            initial_node.state,
            frozenset(initial_node.supported.values()),
        )

    def support_move_greedy_incumbent(
        self,
        initial_node,
        deadline=None,
    ):
        """Seed exact optimization with the fast reduced-support strategy."""
        remaining_runtime = (
            max(0.0, deadline - time.monotonic())
            if deadline is not None
            else 30.0
        )

        if remaining_runtime <= 0:
            return None

        incumbent_deadline = time.monotonic() + min(
            30.0,
            remaining_runtime,
        )
        best_node = None
        best_cost = None
        trials = [
            ("fast_reduce_support", self.random_seed),
            ("full_reduce_support", self.random_seed),
            ("full_reduce_support", self.random_seed + 10_000),
            ("full_reduce_support", self.random_seed + 20_000),
        ]

        for strategy_name, seed in trials:
            trial_runtime = incumbent_deadline - time.monotonic()

            if trial_runtime <= 0:
                break

            greedy_searcher = AssemblyPlanner(
                truss=self.truss,
                builder=None,
                max_supports=len(self.helper_grippers),
                support_grippers=self.helper_grippers,
                forbidden_transitions=None,
                strategy_name=strategy_name,
                random_seed=seed,
                shuffle_ties=self.shuffle_ties,
                optimal_objective="support_moves",
            )
            # Share the checker so incumbent checks remain cached for proof.
            greedy_searcher.rigidity = self.rigidity
            with redirect_stdout(io.StringIO()):
                sequence = greedy_searcher.backward_search(
                    capture_key=None,
                    max_runtime=trial_runtime,
                    max_expansions_without_progress=None,
                    initial_supported=initial_node.supported,
                    initial_support_q=initial_node.support_q,
                )
            self.search_expansions += greedy_searcher.search_expansions
            self.search_attempted_transitions += (
                greedy_searcher.search_attempted_transitions
            )
            self.search_enqueued_candidates += (
                greedy_searcher.search_enqueued_candidates
            )
            self.search_backtracks += greedy_searcher.search_backtracks

            if sequence is None:
                continue

            node = greedy_searcher.final_node
            cost = (
                sum(
                    len(step.added_supports)
                    for step in node.structural_steps
                ),
                sum(
                    len(step.supports_after)
                    for step in node.structural_steps
                ),
                max(
                    [len(initial_node.supported)]
                    + [
                        len(step.supports_after)
                        for step in node.structural_steps
                    ]
                ),
            )

            if best_cost is None or cost < best_cost:
                best_node = node
                best_cost = cost

            if best_cost[0] == 0:
                break

        return best_node

    def optimal_support_moves_search(
        self,
        initial_node,
        outcome_cache,
        deadline,
        max_expansions_without_progress,
        best_goal_node,
        best_goal_cost,
    ):
        """Minimize placements/moves, then occupancy, then peak supports."""
        initial_supported_rods = frozenset(
            initial_node.supported.values()
        )
        initial_cost = (0, 0, len(initial_node.supported))
        initial_key = (
            frozenset(initial_node.state),
            initial_supported_rods,
        )
        best_costs = {initial_key: initial_cost}
        open_list = []
        counter = 0
        heapq.heappush(
            open_list,
            (
                initial_cost,
                len(initial_node.state),
                counter,
                initial_node,
            ),
        )
        counter += 1
        best_remaining = len(initial_node.state)
        last_progress_expansion = self.search_expansions

        def finish_with_goal(proven, stop_reason="runtime_limit"):
            self.final_node = best_goal_node
            self.best_support_moves = best_goal_cost[0]
            self.best_support_steps = best_goal_cost[1]
            self.best_support_peak = best_goal_cost[2]
            self.optimality_proven = proven

            if proven:
                self.optimal_support_moves = best_goal_cost[0]
                self.optimal_support_steps = best_goal_cost[1]
                self.optimal_support_peak = best_goal_cost[2]
                self.search_stop_reason = "complete"
                print(
                    "Optimal support cost: "
                    f"moves={best_goal_cost[0]}, "
                    f"support_steps={best_goal_cost[1]}, "
                    f"peak={best_goal_cost[2]}."
                )
            else:
                self.search_stop_reason = stop_reason
                print(
                    "Support-move optimum not proven before the search "
                    "limit. Returning the best complete path found: "
                    f"moves={best_goal_cost[0]}, "
                    f"support_steps={best_goal_cost[1]}, "
                    f"peak={best_goal_cost[2]}."
                )

            return best_goal_node.sequence

        while open_list:
            if open_list[0][0] >= best_goal_cost:
                return finish_with_goal(proven=True)
            if deadline is not None and time.monotonic() >= deadline:
                return finish_with_goal(proven=False)
            if (
                max_expansions_without_progress is not None
                and self.search_expansions - last_progress_expansion
                >= max_expansions_without_progress
            ):
                return finish_with_goal(
                    proven=False,
                    stop_reason="stagnation_limit",
                )

            cost, _, _, node = heapq.heappop(open_list)
            supported_rods = frozenset(node.supported.values())
            state_key = (frozenset(node.state), supported_rods)

            if cost != best_costs.get(state_key):
                continue
            if not node.state:
                best_goal_node = node
                best_goal_cost = cost
                return finish_with_goal(proven=True)

            for candidate_rod in self.optimal_candidate_order(node.state):
                self.search_expansions += 1
                self.search_attempted_transitions += 1
                new_state = frozenset(node.state - {candidate_rod})
                continuing_supported_rods = (
                    supported_rods - {candidate_rod}
                )

                try:
                    outcomes = self.support_move_outcomes(
                        new_state,
                        continuing_supported_rods,
                        outcome_cache=outcome_cache,
                        deadline=deadline,
                    )
                except TimeoutError:
                    return finish_with_goal(proven=False)

                if not outcomes:
                    self.search_backtracks += 1
                    continue

                for next_supported_rods, rigidity_result in outcomes:
                    new_node = self.optimal_successor_node(
                        node,
                        candidate_rod,
                        next_supported_rods,
                        rigidity_result,
                    )
                    transition = new_node.structural_steps[-1]
                    new_cost = (
                        cost[0] + len(transition.added_supports),
                        cost[1] + len(transition.supports_after),
                        max(cost[2], len(transition.supports_after)),
                    )

                    if new_cost >= best_goal_cost:
                        continue

                    new_key = (
                        frozenset(new_node.state),
                        frozenset(next_supported_rods),
                    )
                    previous_cost = best_costs.get(new_key)

                    if previous_cost is not None and previous_cost <= new_cost:
                        continue

                    best_costs[new_key] = new_cost
                    self.search_enqueued_candidates += 1
                    heapq.heappush(
                        open_list,
                        (
                            new_cost,
                            len(new_node.state),
                            counter,
                            new_node,
                        ),
                    )
                    counter += 1

                    if len(new_node.state) < best_remaining:
                        best_remaining = len(new_node.state)
                        last_progress_expansion = self.search_expansions
                        self.final_node = new_node

        return finish_with_goal(proven=True)

    def optimal_support_steps_search(
        self,
        initial_node,
        outcome_cache,
        deadline,
        max_expansions_without_progress,
        best_goal_node,
        best_goal_cost,
    ):
        """Minimize total support occupancy, with peak as a tie-breaker."""
        initial_peak = len(initial_node.supported)
        initial_cost = (0, initial_peak)
        best_costs = {frozenset(initial_node.state): initial_cost}
        best_pending_costs = {}
        open_list = []
        counter = 0
        heapq.heappush(
            open_list,
            (
                initial_cost,
                len(initial_node.state),
                counter,
                "node",
                initial_node,
            ),
        )
        counter += 1
        best_node = initial_node
        best_remaining = len(initial_node.state)
        last_progress_expansion = 0

        def finish_with_goal(proven, stop_reason="runtime_limit"):
            self.final_node = best_goal_node
            self.best_support_steps = best_goal_cost[0]
            self.best_support_peak = best_goal_cost[1]
            self.optimality_proven = proven

            if proven:
                self.optimal_support_steps = best_goal_cost[0]
                self.optimal_support_peak = best_goal_cost[1]
                self.search_stop_reason = "complete"
                print(
                    "Optimal support cost: "
                    f"support_steps={best_goal_cost[0]}, "
                    f"peak={best_goal_cost[1]}."
                )
            else:
                self.search_stop_reason = stop_reason
                print(
                    "Support-step optimum not proven before the runtime "
                    "limit. Returning the best complete path found: "
                    f"support_steps={best_goal_cost[0]}, "
                    f"peak={best_goal_cost[1]}."
                )

            return best_goal_node.sequence

        def enqueue_node(
            parent_node,
            candidate_rod,
            outcome,
            new_cost,
        ):
            nonlocal counter
            nonlocal best_node
            nonlocal best_remaining
            nonlocal last_progress_expansion

            supported_rods, rigidity_result = outcome
            new_node = self.optimal_successor_node(
                parent_node,
                candidate_rod,
                supported_rods,
                rigidity_result,
            )
            new_state = frozenset(new_node.state)
            previous_cost = best_costs.get(new_state)

            if previous_cost is not None and previous_cost <= new_cost:
                return

            if new_cost >= best_goal_cost:
                return

            best_costs[new_state] = new_cost
            self.search_enqueued_candidates += 1
            heapq.heappush(
                open_list,
                (
                    new_cost,
                    len(new_state),
                    counter,
                    "node",
                    new_node,
                ),
            )
            counter += 1

            if len(new_state) < best_remaining:
                best_node = new_node
                best_remaining = len(new_state)
                last_progress_expansion = self.search_expansions
                self.final_node = best_node

        def enqueue_pending(
            parent_node,
            parent_cost,
            candidate_rod,
            support_limit,
        ):
            nonlocal counter
            new_state = frozenset(parent_node.state - {candidate_rod})
            pending_cost = (
                parent_cost[0] + support_limit,
                max(parent_cost[1], support_limit),
            )
            pending_key = (new_state, support_limit)
            previous_cost = best_pending_costs.get(pending_key)

            if previous_cost is not None and previous_cost <= pending_cost:
                return

            state_cost = best_costs.get(new_state)
            if state_cost is not None and state_cost <= pending_cost:
                return
            if pending_cost >= best_goal_cost:
                return

            best_pending_costs[pending_key] = pending_cost
            self.search_enqueued_candidates += 1
            heapq.heappush(
                open_list,
                (
                    pending_cost,
                    len(new_state),
                    counter,
                    "pending",
                    (
                        parent_node,
                        parent_cost,
                        candidate_rod,
                        support_limit,
                    ),
                ),
            )
            counter += 1

        while open_list:
            if open_list[0][0] >= best_goal_cost:
                return finish_with_goal(proven=True)

            if deadline is not None and time.monotonic() >= deadline:
                return finish_with_goal(proven=False)

            if (
                max_expansions_without_progress is not None
                and self.search_expansions - last_progress_expansion
                >= max_expansions_without_progress
            ):
                return finish_with_goal(
                    proven=False,
                    stop_reason="stagnation_limit",
                )

            cost, _, _, entry_type, payload = heapq.heappop(open_list)

            if entry_type == "pending":
                (
                    parent_node,
                    parent_cost,
                    candidate_rod,
                    support_limit,
                ) = payload
                new_state = frozenset(
                    parent_node.state - {candidate_rod}
                )
                pending_key = (new_state, support_limit)

                if cost != best_pending_costs.get(pending_key):
                    continue
                if cost >= best_goal_cost:
                    continue

                state_cost = best_costs.get(new_state)
                if state_cost is not None and state_cost <= cost:
                    continue

                try:
                    outcome = self.minimum_support_outcome(
                        new_state,
                        support_limit=support_limit,
                        outcome_cache=outcome_cache,
                        deadline=deadline,
                    )
                except TimeoutError:
                    return finish_with_goal(proven=False)

                if outcome is None:
                    if support_limit < len(self.helper_grippers):
                        enqueue_pending(
                            parent_node,
                            parent_cost,
                            candidate_rod,
                            support_limit + 1,
                        )
                    else:
                        self.search_backtracks += 1
                    continue

                supported_rods, _ = outcome
                actual_cost = (
                    parent_cost[0] + len(supported_rods),
                    max(parent_cost[1], len(supported_rods)),
                )
                enqueue_node(
                    parent_node,
                    candidate_rod,
                    outcome,
                    actual_cost,
                )
                continue

            node = payload
            state_key = frozenset(node.state)

            if cost != best_costs.get(state_key):
                continue

            if not node.state:
                best_goal_node = node
                best_goal_cost = cost
                return finish_with_goal(proven=True)

            for candidate_rod in self.optimal_candidate_order(node.state):
                self.search_expansions += 1
                self.search_attempted_transitions += 1
                new_state = frozenset(node.state - {candidate_rod})

                try:
                    outcome = self.minimum_support_outcome(
                        new_state,
                        support_limit=0,
                        outcome_cache=outcome_cache,
                        deadline=deadline,
                    )
                except TimeoutError:
                    return finish_with_goal(proven=False)

                if outcome is None:
                    if self.helper_grippers:
                        enqueue_pending(
                            node,
                            cost,
                            candidate_rod,
                            support_limit=1,
                        )
                    else:
                        self.search_backtracks += 1
                    continue

                enqueue_node(
                    node,
                    candidate_rod,
                    outcome,
                    cost,
                )

        return finish_with_goal(proven=True)

    def optimal_support_search(
        self,
        initial_node,
        max_runtime,
        max_expansions_without_progress,
    ):
        """Run the selected exact structural support objective."""
        if self.builder is not None or self.forbidden_transitions:
            raise ValueError(
                "optimal_supports requires rigidity-only search without "
                "forbidden transitions."
            )

        deadline = (
            time.monotonic() + max_runtime
            if max_runtime is not None
            else None
        )
        outcome_cache = {}

        if self.optimal_objective == "support_moves":
            best_goal_node = self.support_move_greedy_incumbent(
                initial_node,
                deadline=deadline,
            )

            if best_goal_node is None:
                try:
                    incumbent_plan = self.support_move_incumbent_plan(
                        initial_node,
                        outcome_cache=outcome_cache,
                        deadline=deadline,
                    )
                except TimeoutError:
                    self.search_stop_reason = "runtime_limit"
                    self.final_node = initial_node
                    print(
                        "Search runtime limit reached before a complete "
                        "support-move incumbent was found."
                    )
                    return None

                if incumbent_plan is None:
                    self.search_stop_reason = "open_list_exhausted"
                    self.final_node = initial_node
                    print(
                        "No complete sequence exists with available supports."
                    )
                    return None

                best_goal_node = self.build_node_from_optimal_plan(
                    initial_node,
                    incumbent_plan,
                )
            best_goal_cost = (
                sum(
                    len(step.added_supports)
                    for step in best_goal_node.structural_steps
                ),
                sum(
                    len(step.supports_after)
                    for step in best_goal_node.structural_steps
                ),
                max(
                    [len(initial_node.supported)]
                    + [
                        len(step.supports_after)
                        for step in best_goal_node.structural_steps
                    ]
                ),
            )
            self.best_support_moves = best_goal_cost[0]
            self.best_support_steps = best_goal_cost[1]
            self.best_support_peak = best_goal_cost[2]
            print(
                "Optimal support-move search running with incumbent: "
                f"moves={best_goal_cost[0]}, "
                f"support_steps={best_goal_cost[1]}, "
                f"peak={best_goal_cost[2]}."
            )
            return self.optimal_support_moves_search(
                initial_node,
                outcome_cache=outcome_cache,
                deadline=deadline,
                max_expansions_without_progress=(
                    max_expansions_without_progress
                ),
                best_goal_node=best_goal_node,
                best_goal_cost=best_goal_cost,
            )

        if self.optimal_objective == "support_steps":
            try:
                _, incumbent_plan = self.minimum_peak_plan(
                    initial_node,
                    outcome_cache=outcome_cache,
                    randomize=False,
                    deadline=deadline,
                )
            except TimeoutError:
                self.search_stop_reason = "runtime_limit"
                self.final_node = initial_node
                print(
                    "Search runtime limit reached before a complete "
                    "support-step incumbent was found."
                )
                return None

            if incumbent_plan is None:
                self.search_stop_reason = "open_list_exhausted"
                self.final_node = initial_node
                print("No complete sequence exists with available supports.")
                return None

            best_goal_node = self.build_node_from_optimal_plan(
                initial_node,
                incumbent_plan,
            )
            best_goal_cost = (
                sum(
                    len(step.supports_after)
                    for step in best_goal_node.structural_steps
                ),
                max(
                    [len(initial_node.supported)]
                    + [
                        len(step.supports_after)
                        for step in best_goal_node.structural_steps
                    ]
                ),
            )
            self.best_support_steps = best_goal_cost[0]
            self.best_support_peak = best_goal_cost[1]
            print(
                "Optimal support-step search running with incumbent: "
                f"support_steps={best_goal_cost[0]}, "
                f"peak={best_goal_cost[1]}."
            )
            return self.optimal_support_steps_search(
                initial_node,
                outcome_cache=outcome_cache,
                deadline=deadline,
                max_expansions_without_progress=(
                    max_expansions_without_progress
                ),
                best_goal_node=best_goal_node,
                best_goal_cost=best_goal_cost,
            )

        try:
            minimum_peak, plan = self.minimum_peak_plan(
                initial_node,
                outcome_cache=outcome_cache,
                deadline=deadline,
            )
        except TimeoutError:
            self.search_stop_reason = "runtime_limit"
            self.final_node = initial_node
            print("Search runtime limit reached while proving support peak.")
            return None

        if plan is None:
            self.search_stop_reason = "open_list_exhausted"
            self.final_node = initial_node
            print("No complete sequence exists with the available supports.")
            return None

        final_node = self.build_node_from_optimal_plan(initial_node, plan)
        self.final_node = final_node
        self.optimal_support_peak = minimum_peak
        self.best_support_peak = minimum_peak
        self.search_stop_reason = "complete"
        support_steps = sum(
            len(step.supports_after)
            for step in final_node.structural_steps
        )
        self.best_support_steps = support_steps
        self.optimality_proven = True
        print(
            f"Minimum support peak proven: {minimum_peak}. "
            f"Returned path uses {support_steps} support-steps."
        )
        return final_node.sequence
    

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
        self.optimal_support_peak = None
        self.optimal_support_steps = None
        self.optimal_support_moves = None
        self.best_support_peak = None
        self.best_support_steps = None
        self.best_support_moves = None
        self.optimality_proven = False

        if self.strategy_name == "optimal_supports":
            return self.optimal_support_search(
                initial_node,
                max_runtime=max_runtime,
                max_expansions_without_progress=(
                    max_expansions_without_progress
                ),
            )

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
