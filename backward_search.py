from truss import Truss
from collections import defaultdict, deque
import heapq
from DataClasses import SearchNode
from rigidityCheck.truss_rigidity import TrussRigidityChecker

class AssemblyPlanner:
    def __init__(self, truss, builder=None):
        self.truss = truss
        self.builder = builder
        self.motion_records = {}
        self.currently_supported_rod = None
        self.rigidity = TrussRigidityChecker(truss)

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
        probability_two=0.5,
    ):
        return self.rigidity.choose_support_targets(
            active_rods=new_state,
            already_supported=node.supported.values(),
            max_targets=max_targets,
            key=self.heuristic,
        )
    

    # greedy backward search
    def backward_search(self):
        
        """
        Run the backward search
        """
        
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
        )

        # Add all possible first removals to the open list
        for candidate_rod in initial_state:
            priority = self.removal_priority(initial_node, candidate_rod)
            heapq.heappush(open_list, (priority, counter, initial_node, candidate_rod))
            counter += 1

        # TODO: Make a clean check to avoid checking already checked configurations
        # loop until solution found or open list exhausted
        while open_list:
            priority, counter, node, candidate_rod = heapq.heappop(open_list)

            new_state = frozenset(node.state - {candidate_rod})

            feasible, result = self.is_removal_feasible(node, candidate_rod)

            if not feasible:
                continue

            new_node = SearchNode(
                state=new_state,
                sequence=node.sequence + [candidate_rod],
                q=result["q_final"],
                supported=result["supported"],
                support_q=result["support_q"],
                records=node.records + [result["record"]],
            )

            if len(new_state) == 0:
                self.final_node = new_node
                return new_node.sequence

            # debug stopping condition
            if len(new_node.sequence) >= 2:
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

    def is_removal_feasible(self, node, candidate_rod):
        """
        Test whether candidate_rod can be removed from the current scaffold state without violatung stability constraints

        node.state:
            rods currently installed before removing candidate_rod

        new_state:
            rods remaining after removing candidate_rod

        """

        # form as follows: "h2_a1_ur_gripper_center": 6
        current_supports = dict(node.supported or {})

        continuing_supports = {}
        releasable_supports = {}

        for support_gripper, supported_rod in current_supports.items():
            if supported_rod == candidate_rod:
                # This support robot is holding the rod that the main robot will remove.
                # Once the main robot grasps the candidate, this support robot can release the rod
                releasable_supports[support_gripper] = supported_rod
            else:
                # This support robot is holding another rod.
                # -> must keep doing so
                continuing_supports[support_gripper] = supported_rod

        current_state = node.state
        new_state = frozenset(current_state - {candidate_rod})

        HELPER_GRIPPERS = self.builder.support_grippers

        # Check whether the candidate rod is currently supported by a helper gripper.
        candidate_is_supported = self.is_supported_candidate(node, candidate_rod)
        
        print(f"Candidate rod {candidate_rod} is supported: {candidate_is_supported}")
        
        # If the candidate rod is currently supported, find which gripper is holding it.
        if candidate_is_supported:
            gripper_supporting_candidate = self.support_gripper_id_for_candidate(node, candidate_rod)
            
        else:
            gripper_supporting_candidate = None
            
        # Support state after the main robot has grasped the candidate.
        # Continuing supports stay active and must NOT be moved.
        # Releasable supports are only those holding the candidate rod itself.
        supported_after_candidate_grasp = dict(continuing_supports)

        if self.is_valid_state(
            new_state,
            supported_rods=supported_after_candidate_grasp.values(),
        ):
            affected_rods = []
        else:
            affected_rods = self.choose_placeholder_support_targets(
                node=node,
                removed_rod=candidate_rod,
                new_state=new_state,
                max_targets=2,
                probability_two=0.0,
            )

        supported_after_new_assignments = (
            set(supported_after_candidate_grasp.values()) | set(affected_rods)
        )
        rigidity_check = self.rigidity.check(
            new_state,
            supported_rods=supported_after_new_assignments,
        )

        if not rigidity_check.is_rigid:
            print(
                f"Removing rod {candidate_rod} would leave a non-rigid scaffold "
                f"(rank {rigidity_check.rank}/{rigidity_check.dof})."
            )
            return False, None

        print(
            f"After removing rod {candidate_rod}, rigidity requests support for rods: "
            f"{affected_rods if affected_rods else 'none'} "
            f"(rank {rigidity_check.rank}/{rigidity_check.dof})"
        )

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

        # print debug information about support assignments
        if new_support_assignments:
            supported_rods = list(new_support_assignments.values())

            print(
                f"Before removing rod {candidate_rod}, support will be added for rods: "
                f"{supported_rods}"
            )

            for support_gripper, support_rod in new_support_assignments.items():
                print(
                    f"  {support_gripper} supports rod {support_rod}"
                )
        else:
            print(f"Before removing rod {candidate_rod}, no new support is added.")

        result = self.builder.try_remove_and_commit_rod(
            current_state=current_state,
            new_state=new_state,
            rod_id=candidate_rod,
            q_start=node.q,
            supported=node.supported,
            support_q=node.support_q,
            candidate_is_supported=candidate_is_supported,
            old_support_gripper=gripper_supporting_candidate,
            continuing_supports=continuing_supports,
            releasable_supports=releasable_supports,
            new_support_assignments=new_support_assignments,
            use_rrt=False,
            do_shortcut=False,
        )

        if result is None:
            print(f"Removal infeasible for rod {candidate_rod}")
            return False, None

        print(f"Removal feasible for rod {candidate_rod}")
        return True, result
    
    def support_gripper_id_for_candidate(self, node, candidate_rod):
        """
        Return the support gripper currently holding candidate_rod, if any.
        """
        for gripper, supported_rod in node.supported.items():
            if supported_rod == candidate_rod:
                return gripper

        return None





if __name__ == "__main__":
    # truss = Truss.from_json("JSON/long_beam_test.json")
    truss = Truss.from_json("JSON/scaffold_test.json")
    searcher = AssemblyPlanner(truss)

    removal_sequence = searcher.backward_search()
    assembly_sequence = list(reversed(removal_sequence)) if removal_sequence else None
    print("Assembly:", assembly_sequence)


