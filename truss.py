import json

# imports the truss structure based on the JSON
# TODO: replace this  with the actual building elements

class Truss:
    UNIT_TO_MILLIMETERS = {
        "millimeter": 1.0,
        "millimeters": 1.0,
        "millimetre": 1.0,
        "millimetres": 1.0,
        "mm": 1.0,
        "meter": 1000.0,
        "meters": 1000.0,
        "metre": 1000.0,
        "metres": 1000.0,
        "m": 1000.0,
    }

    def __init__(self, nodes, elements, grounded_rods, couplers):
        self.nodes = nodes
        self.elements = elements
        self.grounded_rods = grounded_rods
        self.couplers = couplers
        self.unit = "millimeter"

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)

        source_unit = str(data.get("unit", "millimeter")).strip().lower()
        coordinate_scale = cls.UNIT_TO_MILLIMETERS.get(source_unit)
        if coordinate_scale is None:
            raise ValueError(
                f"Unsupported truss coordinate unit: {source_unit!r}. "
                "Expected millimeters or meters."
            )

        nodes = {}
        rods = {}
        grounded_rods = set()
        couplers = set()
        
        for n in data["node_list"]:
            nid = n["node_id"]
            nodes[nid] = tuple(
                n["point"][axis] * coordinate_scale
                for axis in ("X", "Y", "Z")
            )

        for e in data["rod_list"]:
            rods[e["rod_id"]] = tuple(e["end_node_ids"])
            if e.get("grounded", 0) == 1:
                grounded_rods.add(e["rod_id"])
                
        for c in data.get("coupler_list", []):
            rod_1, rod_2 = c["rod_ids"]

            # Store every pair in a consistent order.
            couplers.add(tuple(sorted((rod_1, rod_2))))      
            
        # print(f"couplers: {couplers}")  

        return cls(nodes, rods, grounded_rods, couplers)
