# Agent Teacher Visualization Audit

## Current Branch Behavior Before This Change

### Are Gazebo scenery images already produced?

Sort of, but not as live Gazebo renders.

The current branch already produced teacher-side overview images through:

- `aic_gym_gz/teacher/visual_context.py`
  - `build_scene_overview_images()`
  - `_render_scene_view()`

These are:

- synthetic schematic renders
- derived from scenario geometry plus current state
- not captured from a Gazebo render camera

### From what code path?

`TeacherContextExtractor.build_planning_state()`
-> `build_scene_overview_images()`
-> attach to `TeacherPlanningState.scene_overview_images`
-> `OpenAIPlannerBackend._visual_content()`
-> Responses API `input_image`

### Fixed world frame or robot attached?

Effectively fixed global views.

The overview render is created from:

- board pose
- plug pose
- target pose
- approximate obstacle world coordinates

Views are global projections:

- `top_down_xy`
- `front_xz`
- `side_yz`

These are not robot-attached views.

### Single-view or multi-view?

Multi-view:

- 3 views total

### How are they stored / referenced / rendered / encoded?

They are:

- rendered with PIL
- encoded as PNG data URLs
- stored in memory inside `TeacherPlanningState.scene_overview_images`
- then passed to the VLM as `input_image`

They were not previously exported as standalone video artifacts.

### Only for VLM context or usable for video export too?

Before this change: only for VLM context.

After the latest change:

- the same schematic overview render path is reused for headless video export as the fallback fourth stream
- the world now also defines a real fixed overview camera sensor on `/overview_camera/image`
- the world also defines a fixed front overview camera sensor on `/overview_front_camera/image`
- the recorder prefers that live topic when available

## Wrist Camera Support

Current live wrist-camera images come from:

- `aic_gym_gz/io.py`
  - `CameraBridgeSidecar`
  - `RosCameraSubscriber`
  - `RosCameraSidecarIO`

The camera streams are:

- `left`
- `center`
- `right`

The wrist streams are still the only real image streams used in the official observation contract.

There is now also a fixed overview camera sensor defined in:

- `aic_description/world/aic.sdf`

with bridged ROS topics configured in:

- `aic_bringup/config/ros_gz_bridge_config.yaml`

This overview camera is currently used by the video recorder when the live topic is available. It is not yet part of the official env observation surface.

Teacher-side visual context now also has an additive path to prefer live overview images for VLM context:

- `top_down_xy` can use `/overview_camera/image`
- `front_xz` can use `/overview_front_camera/image`
- `side_yz` remains schematic today

This is opt-in via `prefer_live_scene_overview`.

## New Headless Video Export Path

Added:

- `aic_gym_gz/video.py`
  - `HeadlessTrajectoryVideoRecorder`
  - `record_teacher_artifact_replay()`

### Default outputs

Per run:

1. `camera_left.mp4`
2. `camera_center.mp4`
3. `camera_right.mp4`
4. `overview_top_down_xy.mp4`

Output root:

- `aic_gym_gz/artifacts/videos/<run_name>/`

Metadata:

- `metadata.json`

### Streaming behavior

The recorder uses `imageio`/`imageio_ffmpeg` streaming writes. It does not buffer the full run in RAM.

### Headless safety

No GUI is required.

The recorder now prefers `/overview_camera/image` via a ROS/Gazebo bridge sidecar. If that topic is unavailable, it falls back to the teacher-side schematic overview render. Wrist-camera streams come from observation images when available. If wrist images are unavailable, placeholder frames are written instead of failing.

## Updated Entry Points

### `demo_teacher_rollout.py`

Now exports video by default unless `--disable-video` is passed.

### `run_teacher_search.py`

Now exports video by default for a replay of the top-ranked candidate unless `--disable-video` is passed.

This is intentionally the top-candidate replay, not simultaneous recording of every search candidate.

### `run_cheatcode_gym.py`

New script. Also exports the same four videos by default.

## Validated Artifact Paths

Generated in this environment:

- `aic_gym_gz/artifacts/videos/teacher_rollout_seed123_default/`
- `aic_gym_gz/artifacts/videos/teacher_search_top1_seed123_default/`
- `aic_gym_gz/artifacts/videos/cheatcode_gym_ep0_seed123_trial_1/`
- `aic_gym_gz/artifacts/context_audit/overview_camera_probe.json`

Each contains:

- 4 mp4 files
- `metadata.json`

## Limitations

- In this validated environment, the fourth "overview" video still fell back to the schematic renderer; `metadata.json` recorded `live_overview_topic: 0` frames and `teacher_schematic_scene_overview: 41` frames.
- The dedicated probe script also reported `ready=false` and zero timestamps for `/overview_camera/image` and `/overview_front_camera/image`, so the fixed world cameras are defined in code but not producing frames in this shell.
- There is not yet a fully validated live multi-angle Gazebo scenery camera pipeline in this shell. Code now defines two fixed world cameras, but full end-to-end validation is blocked because the local `install/` tree does not include `aic_bringup`, so `ros2 launch aic_bringup ...` cannot be run here.
- `aic_gym_gz/io.py` still contains `GazeboNativeIOPlaceholder`, which explicitly says the pure Gazebo transport image path is not wired.
- Mock-environment wrist videos are placeholders or blank because the mock env has no live cameras.

## Recommendation

If offline trajectory inspection is the immediate goal, the current headless export is adequate.

If actual scene cinematics are needed later, the next step should be a real Gazebo camera capture path for one or more fixed world-frame cameras, separate from the teacher schematic renderer.
