import time

class PlanReplayer:
    def __init__(self, C, rod_manager):
        self.C = C
        self.rods = rod_manager
        
    def replay_recorded_plan(
        self,
        recorder,
        rod_pos=[-3, -1, 1.0],
        rod_ori=[0.5, 0.0, 0.5, 0.70710678],
        dt=0.001,
        wait_for_input = True,
    ):
        """
        Replays all recorded paths from the beginning.
        Use this with a fresh builder/config.
        """
        
        if wait_for_input:
            input("Press Enter to start replay...")

        for record in recorder.records:
            rod_id = record.rod_id

            print(f"Replaying rod {rod_id}")

            if self.C.getFrame(f"rod_{rod_id}") is None:
                self.rods.create_rod(rod_id, pos=rod_pos, ori=rod_ori)

            for segment_id, path in enumerate(record.segments):
                for q in path:
                    self.C.setJointState(q)
                    self.C.view(False, f"replay rod {rod_id}, segment {segment_id}")
                    time.sleep(dt)

                for event in record.events:
                    if event.segment_id == segment_id:
                        self.C.attach(event.parent, event.child)
                        print(f"Replay attach: {event.child} to {event.parent}")
                        