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

@dataclass
class SearchNode:
    state: frozenset
    sequence: list = field(default_factory=list)
    q: np.ndarray | None = None
    supported: dict = field(default_factory=dict)  # support_gripper -> rod_id
    records: list = field(default_factory=list)