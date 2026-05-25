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

            # Before first segment:
            # rod starts at pickup pose and is held by the gripper
            assembly_record.events.append(
                AttachmentEvent(
                    rod_id=rod_id,
                    segment_id=-1,
                    parent="a1_ur_gripper_center",
                    child=f"rod_{rod_id}",
                    action="attach",
                )
            )

            # After the placement segment:
            # rod has reached scaffold pose, so release from gripper
            assembly_record.events.append(
                AttachmentEvent(
                    rod_id=rod_id,
                    segment_id=place_segment_id,
                    parent="a1_ur_gripper_center",
                    child=f"rod_{rod_id}",
                    action="detach",
                )
            )

            # After the placement segment:
            # rod becomes fixed in scaffold/table
            assembly_record.events.append(
                AttachmentEvent(
                    rod_id=rod_id,
                    segment_id=place_segment_id,
                    parent="table",
                    child=f"rod_{rod_id}",
                    action="attach",
                )
            )

            assembly_plan.records.append(assembly_record)

        return assembly_plan

@dataclass
class SearchNode:
    state: frozenset
    sequence: list = field(default_factory=list)
    q: np.ndarray | None = None
    supported: dict = field(default_factory=dict)  # support_gripper -> rod_id
    records: list = field(default_factory=list)
    
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