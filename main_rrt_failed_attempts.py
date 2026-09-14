from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"


@dataclass
class RRTFailure:
    index: int
    rod_id: int | None
    segment_id: int | None
    attempt_id: int
    q_start: np.ndarray
    q_goal: np.ndarray
    path: np.ndarray
    synthetic_path: bool
    ret_text: str
    config_snapshot: object


@dataclass
class RRTContext:
    rod_id: int | None = None
    segment_id: int | None = None


class StopAfterCapturedFailures(Exception):
    pass


FAILURES: list[RRTFailure] = []
CONTEXT = RRTContext()


def load_module(module_name: str, source_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_package(name: str):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    return package


def load_reference_modules():
    if not SOURCES.exists():
        sys.path.insert(0, str(ROOT))
        return importlib.import_module("main")

    ensure_package("rai")
    ensure_package("rigidityCheck")

    load_module("DataClasses", SOURCES / "DataClasses(5).py")
    load_module("truss", SOURCES / "truss(5).py")

    load_module("rigidityCheck.utils", SOURCES / "utils(20260825-074507).py")
    load_module("rigidityCheck.Datastructures", SOURCES / "Datastructures(7).py")
    load_module("rigidityCheck.rigiditycheck", SOURCES / "rigiditycheck(7).py")
    load_module("rigidityCheck.truss_rigidity", SOURCES / "truss_rigidity(7).py")

    load_module("rai.utils", SOURCES / "utils(20260825-074454).py")
    for module_name, file_name in (
        ("rai.scene", "scene(5).py"),
        ("rai.rods", "rods(5).py"),
        ("rai.replay", "replay(6).py"),
        ("rai.viser_replay", "viser_replay(5).py"),
        ("rai.pathplanning", "pathplanning(6).py"),
        ("rai.keyframes", "keyframes(7).py"),
        ("rai.builder", "builder(8).py"),
    ):
        load_module(module_name, SOURCES / file_name)

    load_module("backward_search", SOURCES / "backward_search(5).py")
    return load_module("reference_main", SOURCES / "main(5).py")


def path_from_return(ret, q_start, q_goal):
    for attr in ("x", "path"):
        if not hasattr(ret, attr):
            continue

        value = getattr(ret, attr)
        if value is None:
            continue

        try:
            path = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue

        if path.ndim == 1 and path.shape == q_start.shape:
            path = path.reshape(1, -1)

        if path.ndim == 2 and path.shape[1] == q_start.shape[0]:
            return path.copy(), False

    return np.vstack([q_start, q_goal]), True


def snapshot_config(config):
    import robotic as ry

    copied = ry.Config()
    copied.addConfigurationCopy(config)
    return copied


def patch_for_rrt(main_module, args):
    from backward_search import AssemblyPlanner
    from rai.builder import RaiTrussBuilder
    from rai.pathplanning import PathPlanner
    from truss import Truss

    original_backward_search = AssemblyPlanner.backward_search

    def backward_search_without_hotkey(self, *call_args, **call_kwargs):
        call_kwargs["capture_key"] = args.capture_key
        return original_backward_search(self, *call_args, **call_kwargs)

    AssemblyPlanner.backward_search = backward_search_without_hotkey

    original_builder_init = RaiTrussBuilder.__init__

    def builder_init_with_paths(self, *init_args, **init_kwargs):
        original_builder_init(self, *init_args, **init_kwargs)
        if not hasattr(self, "paths"):
            self.paths = PathPlanner(self.C)

    RaiTrussBuilder.__init__ = builder_init_with_paths

    original_try_remove = RaiTrussBuilder.try_remove_and_commit_rod

    def try_remove_with_rrt(self, *call_args, **call_kwargs):
        previous_rod_id = CONTEXT.rod_id
        previous_segment_id = CONTEXT.segment_id

        rod_id = call_kwargs.get("rod_id")
        if rod_id is None and len(call_args) >= 3:
            rod_id = call_args[2]

        CONTEXT.rod_id = rod_id
        CONTEXT.segment_id = None

        if hasattr(self, "paths"):
            self.paths._rrt_segment_counter = 0

        call_kwargs["use_rrt"] = True
        call_kwargs["do_shortcut"] = args.shortcut

        try:
            return original_try_remove(self, *call_args, **call_kwargs)
        finally:
            CONTEXT.rod_id = previous_rod_id
            CONTEXT.segment_id = previous_segment_id

    RaiTrussBuilder.try_remove_and_commit_rod = try_remove_with_rrt

    original_plan_segment = PathPlanner.plan_segment

    def plan_segment_with_context(self, *call_args, **call_kwargs):
        segment_id = getattr(self, "_rrt_segment_counter", 0)
        self._rrt_segment_counter = segment_id + 1

        previous_segment_id = CONTEXT.segment_id
        CONTEXT.segment_id = segment_id
        try:
            return original_plan_segment(
                self,
                *call_args,
                rrt_attempts=args.rrt_attempts,
                **call_kwargs,
            )
        finally:
            CONTEXT.segment_id = previous_segment_id

    PathPlanner.plan_segment = plan_segment_with_context

    def rrt_with_failure_capture(
        self,
        q_start,
        q_goal,
        attempts=50,
        active_joint_names=None,
    ):
        import robotic as ry

        q_start = np.asarray(q_start, dtype=float).copy()
        q_goal = np.asarray(q_goal, dtype=float).copy()

        if active_joint_names is not None and hasattr(self, "rrt_selected_joints"):
            return self.rrt_selected_joints(
                q_start=q_start,
                q_goal=q_goal,
                active_joint_names=active_joint_names,
                attempts=attempts,
            )

        for attempt_id in range(attempts):
            rrt = ry.PathFinder()
            rrt.setProblem(self.C, q_start, q_goal)
            ret = rrt.solve()
            feasible = bool(getattr(ret, "feasible", False))
            print(f"RRT attempt {attempt_id + 1}/{attempts}: {ret}")

            if feasible:
                path, _synthetic = path_from_return(ret, q_start, q_goal)
                return path

            if len(FAILURES) < args.max_failures:
                path, synthetic = path_from_return(ret, q_start, q_goal)
                FAILURES.append(
                    RRTFailure(
                        index=len(FAILURES),
                        rod_id=CONTEXT.rod_id,
                        segment_id=CONTEXT.segment_id,
                        attempt_id=attempt_id,
                        q_start=q_start.copy(),
                        q_goal=q_goal.copy(),
                        path=path,
                        synthetic_path=synthetic,
                        ret_text=str(ret),
                        config_snapshot=snapshot_config(self.C),
                    )
                )

                if (
                    args.stop_after_failures is not None
                    and len(FAILURES) >= args.stop_after_failures
                ):
                    raise StopAfterCapturedFailures(
                        f"Captured {len(FAILURES)} failed RRT attempts."
                    )

        print("RRT failed to find a path for this segment.")
        return None

    PathPlanner.rrt = rrt_with_failure_capture

    if args.json is not None:
        original_from_json = Truss.from_json.__func__

        def from_json_override(cls, _path):
            return original_from_json(cls, str(args.json))

        Truss.from_json = classmethod(from_json_override)

    if not args.show_final_plan:
        def skip_final_plan_replay(self, *replay_args, **replay_kwargs):
            print(
                "Skipping final assembly replay so the failed RRT attempts "
                "can be shown instead."
            )

        RaiTrussBuilder.display_recorded_plan_viser = skip_final_plan_replay

    return main_module


def frame_color(frame):
    info = frame.info()
    raw_color = info.get("color", [0.7, 0.7, 0.7])
    return tuple(int(max(0.0, min(1.0, c)) * 255) for c in raw_color[:3])


def add_config_meshes(server, config, prefix):
    handles = {}

    for frame in config.getFrames():
        try:
            verts = np.asarray(frame.getMeshPoints(), dtype=np.float32)
            tris = np.asarray(frame.getMeshTriangles(), dtype=np.uint32)
        except Exception:
            continue

        if verts.ndim < 2 or tris.ndim < 2 or len(verts) == 0 or len(tris) == 0:
            continue

        raw_color = frame.info().get("color", [0.7, 0.7, 0.7])
        opacity = None
        if len(raw_color) > 3 and raw_color[3] < 1.0:
            opacity = float(raw_color[3])

        handle = server.scene.add_mesh_simple(
            name=f"{prefix}/{frame.name}",
            vertices=verts,
            faces=tris,
            color=frame_color(frame),
            flat_shading=False,
            opacity=opacity,
        )
        handle.visible = False
        handles[frame.name] = handle

    return handles


def config_signature(config):
    return tuple(frame.name for frame in config.getFrames())


def poses_for_path(config, path):
    sim = snapshot_config(config)
    steps = []

    for q in path:
        sim.setJointState(q)
        poses = {}
        for frame in sim.getFrames():
            poses[frame.name] = (
                np.asarray(frame.getPosition(), dtype=np.float32),
                np.asarray(frame.getQuaternion(), dtype=np.float32),
            )
        steps.append(poses)

    return steps


def display_failures(failures, port, pause_time):
    if not failures:
        print("No failed RRT attempts were recorded.")
        return

    import viser

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    signatures = [config_signature(failure.config_snapshot) for failure in failures]
    shared_scene = all(signature == signatures[0] for signature in signatures)

    handles_by_failure = []
    steps = []

    if shared_scene:
        shared_handles = add_config_meshes(
            server,
            failures[0].config_snapshot,
            "rrt_failures",
        )
        handles_by_failure = [shared_handles for _failure in failures]
    else:
        for failure in failures:
            prefix = f"rrt_failure_{failure.index:04d}"
            handles_by_failure.append(
                add_config_meshes(
                    server,
                    failure.config_snapshot,
                    prefix,
                )
            )

    for failure in failures:
        if not shared_scene and len(handles_by_failure) <= failure.index:
            handles_by_failure.append(
                {}
            )

        for pose_index, poses in enumerate(
            poses_for_path(failure.config_snapshot, failure.path)
        ):
            steps.append((failure.index, pose_index, poses))

    if not steps:
        server.stop()
        print("Failed RRT attempts were recorded, but no drawable states exist.")
        return

    step_slider = server.gui.add_slider(
        label="Failed-attempt step",
        min=0,
        max=len(steps) - 1,
        step=1,
        initial_value=0,
    )
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
    pause_field = server.gui.add_number(
        "Pause time (s)",
        initial_value=pause_time,
        min=0.0,
        step=0.001,
    )
    previous_button = server.gui.add_button("Prev")
    next_button = server.gui.add_button("Next")
    stop_button = server.gui.add_button("Stop")
    label = server.gui.add_markdown("")

    state = {"current_failure": None, "stopped": False}

    def set_step(step_index):
        failure_index, pose_index, poses = steps[step_index]
        failure = failures[failure_index]

        if state["current_failure"] != failure_index:
            if state["current_failure"] is not None:
                for handle in handles_by_failure[state["current_failure"]].values():
                    handle.visible = False

            for handle in handles_by_failure[failure_index].values():
                handle.visible = True

            state["current_failure"] = failure_index

        for frame_name, handle in handles_by_failure[failure_index].items():
            pose = poses.get(frame_name)
            if pose is None:
                handle.visible = False
                continue

            handle.visible = True
            handle.position = pose[0]
            handle.wxyz = pose[1]

        path_kind = (
            "start/goal only"
            if failure.synthetic_path
            else f"{len(failure.path)} returned states"
        )
        label.content = (
            f"**Failure:** {failure.index + 1} / {len(failures)}  \n"
            f"**Rod:** {failure.rod_id}  \n"
            f"**Segment:** {failure.segment_id}  \n"
            f"**RRT attempt:** {failure.attempt_id + 1}  \n"
            f"**State:** {pose_index + 1} / {len(failure.path)}  \n"
            f"**Path data:** {path_kind}  \n"
            f"**Return:** `{failure.ret_text}`"
        )

    @stop_button.on_click
    def _(_event):
        state["stopped"] = True

    @previous_button.on_click
    def _(_event):
        play_checkbox.value = False
        step_slider.value = max(0, int(step_slider.value) - 1)
        set_step(int(step_slider.value))

    @next_button.on_click
    def _(_event):
        play_checkbox.value = False
        step_slider.value = min(len(steps) - 1, int(step_slider.value) + 1)
        set_step(int(step_slider.value))

    @step_slider.on_update
    def _(event):
        if not play_checkbox.value:
            set_step(int(event.target.value))

    set_step(0)

    print(f"[viser] Open http://localhost:{port}", flush=True)
    print("[viser] Use the slider to inspect failed RRT attempts.", flush=True)
    print("[viser] Press Stop in the GUI or Ctrl-C to exit.", flush=True)

    try:
        while not state["stopped"]:
            if play_checkbox.value:
                next_step = (int(step_slider.value) + 1) % len(steps)
                step_slider.value = next_step
                set_step(next_step)

            time.sleep(float(pause_field.value))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the mirrored main.py with RRT enabled and visualize failed "
            "RRT attempts in Viser."
        )
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "Optional truss JSON to use instead of the hard-coded path in "
            "sources/main(5).py."
        ),
    )
    parser.add_argument(
        "--rrt-attempts",
        type=int,
        default=50,
        help="Number of RRT solve attempts per keyframe segment.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=500,
        help="Maximum failed RRT attempts to keep for visualization.",
    )
    parser.add_argument(
        "--stop-after-failures",
        type=int,
        default=None,
        help=(
            "Stop planning and open the viewer after this many failed RRT "
            "attempts have been captured."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Viser port for the failed-attempt viewer.",
    )
    parser.add_argument(
        "--pause-time",
        type=float,
        default=0.03,
        help="Playback delay in seconds.",
    )
    parser.add_argument(
        "--shortcut",
        action="store_true",
        help="Shortcut successful RRT paths before committing them.",
    )
    parser.add_argument(
        "--show-final-plan",
        action="store_true",
        help=(
            "Keep the original final assembly replay. By default it is skipped "
            "so the failure viewer opens after planning."
        ),
    )
    parser.add_argument(
        "--capture-key",
        default=None,
        help=(
            "Optional terminal key for the structural-search capture toggle. "
            "Leave unset when running inside Codex."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.rrt_attempts <= 0:
        raise ValueError("--rrt-attempts must be positive")

    if args.max_failures <= 0:
        raise ValueError("--max-failures must be positive")

    if args.stop_after_failures is not None and args.stop_after_failures <= 0:
        raise ValueError("--stop-after-failures must be positive")

    if args.json is not None:
        args.json = args.json.resolve()
        if not args.json.exists():
            raise FileNotFoundError(args.json)

    main_module = load_reference_modules()
    patch_for_rrt(main_module, args)

    run_error = None
    try:
        main_module.main()
    except StopAfterCapturedFailures as stop:
        print(stop)
    except Exception as error:
        run_error = error
        print(f"Main stopped before completion: {error}")
    finally:
        display_failures(
            FAILURES,
            port=args.port,
            pause_time=args.pause_time,
        )

    if run_error is not None:
        raise run_error


if __name__ == "__main__":
    main()
