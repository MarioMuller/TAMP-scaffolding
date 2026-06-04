# rai/viser_replay.py

import time
import numpy as np
import robotic as ry


class ViserPlanReplayer:
    ROD_COLORS = {
        "free": (128, 255, 0),
        "robot": (220, 40, 40),
        "support": (255, 105, 180),
    }

    def __init__(self, C, rod_manager):
        self.C = C
        self.rods = rod_manager

    def _rod_id_from_name(self, frame_name):
        parts = frame_name.split("_")
        if len(parts) < 2 or parts[0] != "rod":
            return None

        try:
            return int(parts[1])
        except ValueError:
            return None

    def _is_primary_rod_frame(self, frame_name):
        rod_id = self._rod_id_from_name(frame_name)
        return rod_id is not None and frame_name == f"rod_{rod_id}"

    def _event_rod_id(self, event):
        return self._rod_id_from_name(event.child)

    def _rod_state_from_parent(self, parent):
        if parent in ("world", "table"):
            return "free"

        if "gripper" not in parent:
            return "free"

        if parent.startswith("h"):
            return "support"

        return "robot"

    def _apply_events_to_rod_states(self, rod_states, events):
        for event in events:
            rod_id = self._event_rod_id(event)
            if rod_id is None:
                continue

            if event.action == "attach":
                rod_states[rod_id] = self._rod_state_from_parent(event.parent)

            elif event.action == "detach":
                rod_states[rod_id] = "free"

    def _build_display_config(
            self,
            recorder,
            rod_pos=(-3, -1, 1.0),
            rod_ori=(0.5, 0.0, 0.5, 0.70710678),
            replay_mode="removal",
        ):
        """
        Builds a config containing all rods that appear in the recorded plan.
        """

        for record in recorder.records:
            rod_id = record.rod_id
            rod_name = f"rod_{rod_id}"

            if self.C.getFrame(rod_name) is None:
                self.rods.create_rod(
                    rod_id,
                    pos=rod_pos,
                    ori=rod_ori,
                )

            if replay_mode == "removal":
                # Removal starts with rod installed.
                self.rods.set_to_goal_pose(rod_id, view=False)

                if self.C.getFrame("table") is not None:
                    self.C.attach("table", rod_name)

            elif replay_mode == "assembly":
                # Assembly starts with rod at pickup/staging pose.
                self.C.getFrame(rod_name) \
                    .setPosition(rod_pos) \
                    .setQuaternion(rod_ori)

        C_display_base = ry.Config()
        C_display_base.addConfigurationCopy(self.C)

        return C_display_base

    def _precompute_viser_steps(self, recorder, C_base, replay_mode="removal"):
        C_sim = ry.Config()
        C_sim.addConfigurationCopy(C_base)

        steps = []

        records = list(recorder.records)

        if replay_mode == "removal":
            visible_rods = {record.rod_id for record in records}
        elif replay_mode == "assembly":
            visible_rods = {records[0].rod_id} if records else set()
        else:
            visible_rods = set()

        rod_states = {
            record.rod_id: "free"
            for record in records
        }

        for record_index, record in enumerate(records):
            rod_id = record.rod_id

            print(f"Replaying rod {rod_id}")

            # Events with segment_id == -1 happen before the first segment.
            pre_events = [
                event for event in record.events
                if event.segment_id == -1
            ]

            pre_events_applied = False
            for segment_id, path in enumerate(record.segments):
                for q_id, q in enumerate(path):
                    C_sim.setJointState(q)

                    # Pre-events need the first replay configuration already set.
                    # Otherwise C.attach preserves a relative transform from the
                    # default robot pose and the rod gets carried with a bad offset.
                    if not pre_events_applied and segment_id == 0 and q_id == 0:
                        for event in pre_events:
                            if event.action == "detach":
                                C_sim.attach("world", event.child)
                                print(f"Replay pre-detach: {event.child} from {event.parent}")

                        for event in pre_events:
                            if event.action == "attach":
                                C_sim.attach(event.parent, event.child)
                                print(f"Replay pre-attach: {event.child} to {event.parent}")

                        self._apply_events_to_rod_states(
                            rod_states,
                            pre_events,
                        )

                        pre_events_applied = True

                    poses = {}
                    for frame in C_sim.getFrames():
                        poses[frame.name] = (
                            np.asarray(frame.getPosition(), dtype=np.float32),
                            np.asarray(frame.getQuaternion(), dtype=np.float32),
                        )

                    steps.append({
                        "rod_id": rod_id,
                        "segment_id": segment_id,
                        "poses": poses,
                        "visible_rods": set(visible_rods),
                        "rod_states": dict(rod_states),
                    })

                segment_events = [
                    event for event in record.events
                    if event.segment_id == segment_id
                ]

                # First apply detaches
                for event in segment_events:
                    if event.action == "detach":
                        C_sim.attach("world", event.child)
                        print(f"Replay detach: {event.child} from {event.parent}")

                # Then apply attaches
                for event in segment_events:
                    if event.action == "attach":
                        C_sim.attach(event.parent, event.child)
                        print(f"Replay attach: {event.child} to {event.parent}")

                self._apply_events_to_rod_states(
                    rod_states,
                    segment_events,
                )
                        
                        
                # Visibility update after events.
                # In assembly mode, rods stay visible after they have been introduced.
                # In removal mode, you could remove visibility after pickup later,
                # but for now keeping them visible is useful for debugging.

            if replay_mode == "assembly" and record_index + 1 < len(records):
                next_rod_id = records[record_index + 1].rod_id
                visible_rods.add(next_rod_id)

                if len(steps) > 0:
                    steps[-1]["visible_rods"] = set(visible_rods)
                    steps[-1]["rod_states"] = dict(rod_states)

        return steps

    def _viser_set_step(self, i, steps, handles, mode_label=None):
        step = steps[i]
        visible_rods = step["visible_rods"]
        rod_states = step.get("rod_states", {})

        for frame_name, handle in handles.items():
            is_rod_frame = frame_name.startswith("rod_")

            if is_rod_frame:
                rod_id = self._rod_id_from_name(frame_name)

                if isinstance(handle, dict):
                    state = rod_states.get(rod_id, "free")
                    for variant_state, variant_handle in handle.items():
                        variant_handle.visible = (
                            rod_id in visible_rods
                            and variant_state == state
                        )
                else:
                    handle.visible = rod_id in visible_rods

            if frame_name not in step["poses"]:
                continue

            pos, quat = step["poses"][frame_name]

            if isinstance(handle, dict):
                for variant_handle in handle.values():
                    variant_handle.position = pos
                    variant_handle.wxyz = quat
            else:
                handle.position = pos
                handle.wxyz = quat

        if mode_label is not None:
            support_rods = sorted(
                rod_id
                for rod_id, state in rod_states.items()
                if state == "support" and rod_id in visible_rods
            )

            mode_label.content = (
                f"**Step:** {i} / {len(steps) - 1}  \n"
                f"**Rod:** {step['rod_id']}  \n"
                f"**Segment:** {step['segment_id']}  \n"
                f"**Visible rods:** {sorted(visible_rods)}  \n"
                f"**Support required:** {support_rods if support_rods else 'none'}  \n"
                "**Colors:** green = free/placed, red = attached to robot, pink = support required"
            )

    def display_recorded_plan_viser(
        self,
        recorder,
        pause_time=0.05,
        port=8080,
        rod_pos=(-3, -1, 1.0),
        rod_ori=(0.5, 0.0, 0.5, 0.70710678),
        primitives_only=False,
        replay_mode="removal",
    ):
        try:
            import viser
        except ImportError:
            raise ImportError(
                "viser is required. Install with: pip install viser"
            )

        C_display_base = self._build_display_config(
            recorder,
            rod_pos=rod_pos,
            rod_ori=rod_ori,
            replay_mode=replay_mode,
        )

        steps = self._precompute_viser_steps(
            recorder,
            C_display_base,
            replay_mode=replay_mode,
        )

        if len(steps) == 0:
            raise RuntimeError("Recorded plan is empty.")

        C_display = ry.Config()
        C_display.addConfigurationCopy(C_display_base)

        server = viser.ViserServer(port=port)
        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = False

        handles = {}

        for frame in C_display.getFrames():
            if primitives_only and frame.info().get("shape") == "mesh":
                continue

            verts = np.asarray(frame.getMeshPoints(), dtype=np.float32)
            tris = np.asarray(frame.getMeshTriangles(), dtype=np.uint32)

            if verts.ndim < 2 or tris.ndim < 2:
                continue

            info = frame.info()
            raw_color = info.get("color", [0.7, 0.7, 0.7])
            color_rgb = tuple(int(c * 255) for c in raw_color[:3])

            opacity = None
            if len(raw_color) > 3 and raw_color[3] < 1.0:
                opacity = float(raw_color[3])

            if self._is_primary_rod_frame(frame.name):
                handles[frame.name] = {}

                for state, rod_color in self.ROD_COLORS.items():
                    handles[frame.name][state] = server.scene.add_mesh_simple(
                        name=f"frames/{frame.name}/{state}",
                        vertices=verts,
                        faces=tris,
                        color=rod_color,
                        flat_shading=False,
                        opacity=opacity,
                    )
            else:
                handles[frame.name] = server.scene.add_mesh_simple(
                    name=f"frames/{frame.name}",
                    vertices=verts,
                    faces=tris,
                    color=color_rgb,
                    flat_shading=False,
                    opacity=opacity,
                )

        step_slider = server.gui.add_slider(
            label="Step",
            min=0,
            max=len(steps) - 1,
            step=1,
            initial_value=0,
        )

        play_checkbox = server.gui.add_checkbox(
            "Play",
            initial_value=False,
        )

        pause_time_field = server.gui.add_number(
            "Pause time (s)",
            initial_value=pause_time,
            min=0.0,
            step=0.001,
        )

        step_size_field = server.gui.add_number(
            "Step size",
            initial_value=1,
            min=1,
            step=1,
        )

        prev_btn = server.gui.add_button("◀ Prev")
        next_btn = server.gui.add_button("Next ▶")
        stop_btn = server.gui.add_button("Stop")
        mode_label = server.gui.add_markdown("")

        stopped = False

        def clamped_step():
            return min(int(step_slider.value), len(steps) - 1)

        @stop_btn.on_click
        def _(_):
            nonlocal stopped
            stopped = True

        @prev_btn.on_click
        def _(_):
            play_checkbox.value = False

            step = max(
                clamped_step() - int(step_size_field.value),
                0,
            )

            step_slider.value = step

            self._viser_set_step(
                step,
                steps,
                handles,
                mode_label,
            )

        @next_btn.on_click
        def _(_):
            play_checkbox.value = False

            step = min(
                clamped_step() + int(step_size_field.value),
                len(steps) - 1,
            )

            step_slider.value = step

            self._viser_set_step(
                step,
                steps,
                handles,
                mode_label,
            )

        @step_slider.on_update
        def _(event):
            if not play_checkbox.value:
                step = min(
                    int(event.target.value),
                    len(steps) - 1,
                )

                self._viser_set_step(
                    step,
                    steps,
                    handles,
                    mode_label,
                )

        self._viser_set_step(
            0,
            steps,
            handles,
            mode_label,
        )

        print(f"[viser] Open http://localhost:{port}")
        print("[viser] Press Stop in the GUI or Ctrl-C to exit.")

        try:
            while not stopped:
                if play_checkbox.value:
                    next_step = (
                        clamped_step() + int(step_size_field.value)
                    ) % len(steps)

                    step_slider.value = next_step

                    self._viser_set_step(
                        next_step,
                        steps,
                        handles,
                        mode_label,
                    )

                time.sleep(float(pause_time_field.value))

        except KeyboardInterrupt:
            pass

        finally:
            server.stop()
