from truss import Truss
from collections import defaultdict, deque
import heapq
from DataClasses import SearchNode
import random

class AssemblyPlanner:
    def __init__(self, truss, builder=None):
        self.truss = truss
        self.builder = builder
        self.motion_records = {}
        self.currently_supported_rod = None

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
    def is_valid_state(self, active_rods):
        graph, active_nodes = self.build_graph(active_rods)
        visited = set()

        for start in active_nodes:
            if start in visited:
                continue

            q = deque([start])
            has_ground = False

            while q:
                node = q.popleft()
                if node in visited:
                    continue

                visited.add(node)
                if node in self.truss.grounded_nodes:
                    has_ground = True

                for neighbour in graph[node]:
                    if neighbour not in visited:
                        q.append(neighbour)

            if not has_ground:
                return False

        return True

    # use height as heuristicc
    def heuristic(self, rod_id):
        n1, n2 = self.truss.elements[rod_id]
        return 0.5 * (self.truss.nodes[n1][2] + self.truss.nodes[n2][2])
    
    def is_supported_candidate(self, node, rod_id):
        return rod_id in node.supported.values()

    # TODO: Combine this with heuristic
    def removal_priority(self, node, rod_id):
        """
        Supported rods should be tried before unsupported rods.
        Within each group, use the normal height heuristic.
        """
        supported_rank = 0 if self.is_supported_candidate(node, rod_id) else 1
        return (
            len(node.state),
            supported_rank,
            -self.heuristic(rod_id),
        )

    def choose_placeholder_support_targets(
        self,
        node,
        removed_rod,
        new_state,
        max_targets=2,
        probability_two=1,
    ):
        already_supported = set(node.supported.values())

        candidates = [
            r for r in new_state
            if r != removed_rod
        ]

        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=self.heuristic,
            reverse=True,
        )

        if len(ranked) >= 2 and random.random() < probability_two:
            print("two rods need support, selecting two placeholder targets")
            n_targets = 2
        else:
            print("only one rod needs support, selecting one placeholder target")
            n_targets = 1

        return ranked[:min(n_targets, max_targets)]

    # greedy backward search
    def backward_search(self):
        initial_state = frozenset(self.truss.elements.keys())

        open_list = []
        counter = 0
        visited = set()

        # initialize search
        # counter use first in first out if same priority
        # save current state and rod to try
        # for rod_id in initial_state:
        #     # needs to be negative because heapq uses smallest!
        #     priority = (len(initial_state), -self.heuristic(rod_id))
        #     heapq.heappush(open_list, (priority, counter, initial_state, rod_id, []))
        #     counter += 1

        initial_node = SearchNode(
            state=initial_state,
            sequence=[],
            q=None,
            supported={},
            support_q={},
            records=[],
        )

        for rod_id in initial_state:
            priority = self.removal_priority(initial_node, rod_id)
            heapq.heappush(open_list, (priority, counter, initial_node, rod_id))
            counter += 1

        # TODO: Make a clean check to avoid checking already checked configurations
      
        while open_list:
            priority, counter, node, rod_id = heapq.heappop(open_list)

            current_state = node.state
            new_state = frozenset(node.state - {rod_id})

            feasible, result = self.is_removal_feasible(node, rod_id)

            if not feasible:
                continue

            new_node = SearchNode(
                state=new_state,
                sequence=node.sequence + [rod_id],
                q=result["q_final"],
                supported=result["supported"],
                support_q=result["support_q"],
                records=node.records + [result["record"]],
            )

            if len(new_state) == 0:
                self.final_node = new_node
                return new_node.sequence

            # debug stopping condition
            if len(new_node.sequence) == 3:
                self.final_node = new_node
                return new_node.sequence

            for next_rod in new_state:
                priority = self.removal_priority(new_node, next_rod)
                heapq.heappush(
                    open_list,
                    (priority, counter, new_node, next_rod)
                )
                counter += 1

        return None

    def is_removal_feasible(self, node, rod_id):
        """
        Test whether rod_id can be removed from the current scaffold state.

        node.state:
            rods currently installed before removing rod_id

        new_state:
            rods remaining after removing rod_id

        node.supported:
            branch-local support state, e.g.
            {
                "h2_a1_ur_gripper_center": 6
            }
        """
        
        candidate_rod = rod_id

        current_supports = dict(node.supported or {})
        # Expected format:
        # current_supports = {
        #     "h1_ur_gripper_center": 7,
        #     "h2_ur_gripper_center": 8,
        # }

        continuing_supports = {}
        releasable_supports = {}

        for support_gripper, supported_rod in current_supports.items():
            if supported_rod == candidate_rod:
                # This support robot is holding the rod that the main robot will remove.
                # Once the main robot grasps the candidate, this support robot can release.
                releasable_supports[support_gripper] = supported_rod
            else:
                # This support robot is holding another rod.
                # It must not move.
                continuing_supports[support_gripper] = supported_rod

        current_state = node.state
        new_state = frozenset(current_state - {rod_id})

        # Real support-dependency detection is not implemented yet.
        # Therefore, invalid states are still rejected for now.
        if not self.is_valid_state(new_state):
            print(
                f"Removing rod {rod_id} would make the scaffold invalid. "
                "Real support selection is not implemented yet."
            )
            return False, None

        HELPER_GRIPPERS = self.builder.support_grippers

        candidate_is_supported = self.is_supported_candidate(node, rod_id)
        old_support_gripper = self.old_support_gripper_for_candidate(node, rod_id)

        # Temporary placeholder for rods that become unstable after removing rod_id.
        # TODO: Replace with actual structure check
        affected_rods = self.choose_placeholder_support_targets(
            node=node,
            removed_rod=rod_id,
            new_state=new_state,
            max_targets=2,
            probability_two=1.0,
        )
        
        print(
            f"After removing rod {rod_id}, placeholder requests support for rods: "
            f"{affected_rods if affected_rods else 'none'}"
        )

        # Support state after the main robot has grasped the candidate.
        # Continuing supports stay active and must NOT be moved.
        # Releasable supports are only those holding the candidate rod itself.
        supported_after_candidate_grasp = dict(continuing_supports)

        # Only grippers that are not continuing supports may be assigned to new support.
        # This includes:
        #   - completely unused support robots
        #   - support robots that were holding the candidate rod and can release after grasp
        free_support_grippers = [
            gripper
            for gripper in HELPER_GRIPPERS
            if gripper not in continuing_supports
        ]

        new_support_assignments = {}

        for affected_rod in affected_rods:
            # Already supported by a continuing support robot.
            if affected_rod in supported_after_candidate_grasp.values():
                continue

            if not free_support_grippers:
                print(
                    f"Would like to support rod {affected_rod}, "
                    "but no helper gripper is available."
                )
                return False, None

            free_gripper = free_support_grippers.pop(0)

            new_support_assignments[free_gripper] = affected_rod
            supported_after_candidate_grasp[free_gripper] = affected_rod      

        if new_support_assignments:
            supported_rods = list(new_support_assignments.values())

            print(
                f"Before removing rod {rod_id}, support will be added for rods: "
                f"{supported_rods}"
            )

            for support_gripper, support_rod in new_support_assignments.items():
                print(
                    f"  {support_gripper} supports rod {support_rod}"
                )
        else:
            print(f"Before removing rod {rod_id}, no new support is added.")

        result = self.builder.try_remove_and_commit_rod(
            current_state=current_state,
            new_state=new_state,
            rod_id=rod_id,
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

        if result is None:
            print(f"Removal infeasible for rod {rod_id}")
            return False, None

        print(f"Removal feasible for rod {rod_id}")
        return True, result
    
    def old_support_gripper_for_candidate(self, node, rod_id):
        """
        Return the support gripper currently holding rod_id, if any.
        """
        for gripper, supported_rod in node.supported.items():
            if supported_rod == rod_id:
                return gripper

        return None





if __name__ == "__main__":
    # truss = Truss.from_json("JSON/long_beam_test.json")
    truss = Truss.from_json("JSON/scaffold_test.json")
    searcher = AssemblyPlanner(truss)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
    print("Assembly:", assembly_sequence)


