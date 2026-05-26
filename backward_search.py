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

    def choose_placeholder_support_target(self, node, removed_rod, new_state):
        """
        Temporary placeholder until real instability-based support selection exists.

        Randomly decide whether to support another remaining rod.
        If yes, choose the next/highest rod according to the heuristic.
        """
        
        # if random.random() > 0.0:
        #     return None

        already_supported = set(node.supported.values())

        candidates = [
            r for r in new_state
            if r != removed_rod and r not in already_supported
        ]

        if not candidates:
            return None

        return max(candidates, key=self.heuristic)

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

        # while open_list:
        #     priority, counter, state, rod_id, sequence = heapq.heappop(open_list)

        #     # remove rod
        #     new_state = frozenset(state - {rod_id})

        #     # TODO: Make a clean check to avoid checking already checked configurations
        #     # if new_state in visited:
        #     #     continue
        #     # visited.add(new_state)
        #     if new_state not in visited:
        #         visited.add(new_state)

        #     if not self.is_valid_state(new_state):
        #         continue

        #     if not self.is_motion_feasible(new_state, rod_id):
        #         continue

        #     # if it is a feasible option add rod to remove sequence
        #     new_sequence = sequence + [rod_id]

        #     #debug
        #     if len(new_sequence) == 3:
        #         return new_sequence

        #     # check if there are remaining nodes
        #     if len(new_state) == 0:
        #         return new_sequence

        #     # add all rods that could be removed to open_list
        #     for next_rod in new_state:
        #         priority = (len(new_state), -self.heuristic(next_rod))
        #         heapq.heappush(
        #             open_list,
        #             (priority, counter, new_state, next_rod, new_sequence)
        #         )
        #         counter += 1

        # return None

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
            if len(new_node.sequence) == 10:
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

        current_state = node.state
        new_state = frozenset(current_state - {rod_id})

        if self.builder is None:
            return self.is_valid_state(new_state), {
                "record": None,
                "q_final": node.q,
                "supported": node.supported,
                "support_q": node.support_q,
            }

        # Real support-dependency detection is not implemented yet.
        # Therefore, invalid states are still rejected for now.
        if not self.is_valid_state(new_state):
            print(
                f"Removing rod {rod_id} would make the scaffold invalid. "
                "Real support selection is not implemented yet."
            )
            return False, None

        HELPER_GRIPPERS = [
            "h1_ur_gripper_center",
            "h2_ur_gripper_center",
        ]

        candidate_is_supported = self.is_supported_candidate(node, rod_id)
        old_support_gripper = self.old_support_gripper_for_candidate(node, rod_id)

        # Temporary placeholder for rods that become unstable after removing rod_id.
        # Later this should be replaced by a real structural check.
        affected_rods = []
        placeholder_support_target = self.choose_placeholder_support_target(
            node=node,
            removed_rod=rod_id,
            new_state=new_state,
        )

        if placeholder_support_target is not None:
            affected_rods.append(placeholder_support_target)

        # Compute support availability after the main robot has grasped the candidate.
        # This means a support robot currently holding the candidate can be reused.
        available_supported = dict(node.supported)

        if candidate_is_supported and old_support_gripper is not None:
            available_supported.pop(old_support_gripper, None)

        new_support_assignments = {}

        for affected_rod in affected_rods:
            if affected_rod in available_supported.values():
                continue

            free_gripper = None
            for gripper in HELPER_GRIPPERS:
                if gripper not in available_supported:
                    free_gripper = gripper
                    break

            if free_gripper is None:
                print(
                    f"Would like to support rod {affected_rod}, "
                    "but no helper gripper is available."
                )
                return False, None

            new_support_assignments[free_gripper] = affected_rod
            available_supported[free_gripper] = affected_rod

        if new_support_assignments:
            print(
                f"Temporary placeholder support before removing rod {rod_id}: "
                f"{new_support_assignments}"
            )

        result = self.builder.try_remove_and_commit_rod(
            current_state=current_state,
            new_state=new_state,
            rod_id=rod_id,
            q_start=node.q,
            supported=node.supported,
            support_q=node.support_q,
            candidate_is_supported=candidate_is_supported,
            old_support_gripper=old_support_gripper,
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


