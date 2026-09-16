from __future__ import annotations

import json

from truss import Truss


def parse_rod_ids(raw_value):
    """Parse comma-separated rod ids or a JSON list of rod ids."""
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text:
        return None

    if text.startswith("["):
        values = json.loads(text)
    else:
        values = text.split(",")

    return {
        int(value)
        for value in values
        if str(value).strip()
    }


def load_filtered_truss(truss_path, included_rods=None):
    truss = Truss.from_json(truss_path)

    if included_rods is None:
        return truss

    included_rods = set(included_rods)
    unknown_rods = included_rods - set(truss.elements)
    if unknown_rods:
        raise ValueError(
            f"Included rods do not exist: {sorted(unknown_rods)}"
        )

    truss.elements = {
        rod_id: endpoints
        for rod_id, endpoints in truss.elements.items()
        if rod_id in included_rods
    }
    truss.grounded_rods &= included_rods
    truss.couplers = {
        (rod_1, rod_2)
        for rod_1, rod_2 in truss.couplers
        if rod_1 in included_rods and rod_2 in included_rods
    }

    return truss

