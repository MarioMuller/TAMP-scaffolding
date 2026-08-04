from dataclasses import dataclass, field
import numpy as np

@dataclass
class AttachmentEvent:
    rod_id: int
    segment_id: int
    parent: str
    child: str
    action: str = "attach"  # "attach" or "detach"

@dataclass
class RodPathRecord:
    rod_id: int
    segments: list = field(default_factory=list)
    events: list = field(default_factory=list)
    
@dataclass
class StructuralRemovalStep:
    rod_id: int

    rods_before: frozenset[int]
    rods_after: frozenset[int]

    supports_before: dict[str, int]
    supports_after: dict[str, int]

    added_supports: dict[str, int]
    released_supports: dict[str, int]

    rank_after: int
    dof_after: int

@dataclass
class AssemblyPlan:
    removal_sequence: list = field(default_factory=list)
    records: list = field(default_factory=list)

    @property
    def assembly_sequence(self):
        return list(reversed(self.removal_sequence))

    def add(self, rod_id, record):
        self.removal_sequence.append(rod_id)
        self.records.append(record)

    def records_in_assembly_order(self):
        return list(reversed(self.records))

    def sequence(self):
        return self.assembly_sequence
    
    def reverse_removal_plan_to_assembly(removal_plan):
        assembly_plan = AssemblyPlan(
            removal_sequence=list(removal_plan.removal_sequence),
            records=[],
        )

        for removal_record in reversed(removal_plan.records):
            rod_id = removal_record.rod_id

            assembly_record = RodPathRecord(rod_id=rod_id)

            # Reverse segment order and reverse each segment internally
            assembly_record.segments = [
                np.asarray(path)[::-1].copy()
                for path in reversed(removal_record.segments)
            ]

            # Removal attaches the rod to the gripper after the "reach installed rod"
            # segment, then carries it to pickup. In reverse, the first segment is
            # the carry back to the scaffold, so placement happens after segment 0.
            place_segment_id = 0 if assembly_record.segments else -1

            # Was this rod supported during the removal plan?
            # In removal, this appears as a support gripper detaching from the candidate rod.
            support_release_events = [
                e for e in removal_record.events
                if e.child == f"rod_{rod_id}"
                and e.action == "detach"
                and e.parent.startswith("h")
            ]

            needs_support_handover = len(support_release_events) > 0

            if needs_support_handover:
                support_grippers = [
                    e.parent for e in support_release_events
                ]

                # In reversed assembly:
                # segment 0 = main carries rod to final pose
                # segment 1 = support robot moves in
                # after segment 1 = handover
                handover_segment_id = 1 if len(assembly_record.segments) > 1 else 0
            else:
                support_grippers = []
                handover_segment_id = place_segment_id

            # Before first segment:
            # rod starts at pickup pose and is held by the main gripper
            assembly_record.events.append(
                AttachmentEvent(
                    rod_id=rod_id,
                    segment_id=-1,
                    parent="a1_ur_gripper_center",
                    child=f"rod_{rod_id}",
                    action="attach",
                )
            )

            # If support handover is needed, support attaches before main releases
            for support_gripper in support_grippers:
                assembly_record.events.append(
                    AttachmentEvent(
                        rod_id=rod_id,
                        segment_id=handover_segment_id,
                        parent=support_gripper,
                        child=f"rod_{rod_id}",
                        action="attach",
                    )
                )

            # Main releases only after support is attached, or directly after placement
            assembly_record.events.append(
                AttachmentEvent(
                    rod_id=rod_id,
                    segment_id=handover_segment_id,
                    parent="a1_ur_gripper_center",
                    child=f"rod_{rod_id}",
                    action="detach",
                )
            )

                        # If no support handover is needed, rod becomes fixed to table/scaffold
            if not needs_support_handover:
                assembly_record.events.append(
                    AttachmentEvent(
                        rod_id=rod_id,
                        segment_id=place_segment_id,
                        parent="table",
                        child=f"rod_{rod_id}",
                        action="attach",
                    )
                )

            # ------------------------------------------------------------
            # Reverse support attachments from removal.
            #
            # In removal:
            #   while removing rod X, support robot h attaches to rod Y
            #
            # In reversed assembly:
            #   when rod X is assembled again, rod Y may no longer need that
            #   temporary support, so h should detach from rod Y.
            # ------------------------------------------------------------

            support_attach_events = [
                e for e in removal_record.events
                if e.action == "attach"
                and e.parent.startswith("h")
                and e.child.startswith("rod_")
                and e.child != f"rod_{rod_id}"
            ]

            support_release_segment_id = place_segment_id

            for e in support_attach_events:
                assembly_record.events.append(
                    AttachmentEvent(
                        rod_id=e.rod_id,
                        segment_id=support_release_segment_id,
                        parent=e.parent,
                        child=e.child,
                        action="detach",
                    )
                )

            assembly_plan.records.append(assembly_record)

        return assembly_plan

@dataclass
class SearchNode:
    state: frozenset # frozenset of remaining rod ids in the current search node
    sequence: list = field(default_factory=list) # rod ids in removal order

    # Full joint configuration at this search node
    q: np.ndarray | None = None

    # rods that are currently supported by support robots.
    # Example:
    # "h1_ur_gripper_center": 8,
    # "h2_ur_gripper_center": 12,
    supported: dict = field(default_factory=dict)

    # TODO: potentially use end effector poses instead of full joint configuration
    # joint configuration in which the support robot holds rod
    #
    # Example:
    # "h1_ur_gripper_center": np.array([...]),
    support_q: dict = field(default_factory=dict)

    records: list = field(default_factory=list)
    
    # Builder-independent structural transitions
    structural_steps: list = field(default_factory=list)
    
    def unused_helpers(self, helper_grippers):
        return [
            gripper
            for gripper in helper_grippers
            if gripper not in self.supported
        ]

    def first_unused_helper(self, helper_grippers):
        free = self.unused_helpers(helper_grippers)
        return free[0] if free else None

    def has_unused_helper(self, helper_grippers):
        return self.first_unused_helper(helper_grippers) is not None