# ur5_station_ws

ROS 2 Humble workspace for controlling one UR5 arm with a Robotiq 2F-85 gripper
and tactile sensors (DIGIT, GelSight Mini) from Python.

## Quickstart

- **Station PC already set up?** Go straight to the robot:
  **[docs/robot_startup.md](docs/robot_startup.md)**, from power-on to first motion and shutdown,
  then `src/ur5_station/notebooks/ur5_station_demo.ipynb`.
- **New PC, or new to ROS?** Start with **[docs/getting_started.md](docs/getting_started.md)**:
  prerequisites, ROS 2 Humble install, `./setup.sh`, a simulated run, and the
  one-time network / pendant / gripper setup.

In every new terminal, from the workspace folder (wherever you cloned it):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## What's in here

| Path | What |
|---|---|
| `src/ur5/ur5_description` | Robot model: UR5 + gripper, mounting pose, calibration |
| `src/ur5/ur5_moveit_config` | MoveIt planning config |
| `src/ur5/ur5_bringup` | Launch files, controllers, planning-scene walls, gripper udev rule |
| `src/ur5_station` | Python API: arm / gripper / freedrive clients, sensor recorders, live views, demo notebook |
| `src/ros2_robotiq_gripper` | Upstream Robotiq driver (submodule, pinned). Only `robotiq_driver` is built; description + controllers come from apt |
| `src/serial` | Serial library for the gripper driver (submodule, pinned) |
| `deps.repos` | Source deps imported by `setup.sh` into `src/deps/` (pymoveit2) |
| `deps-audio.repos` | Optional ReSpeaker microphone support: `./setup.sh --with-audio` |

## Real robot

```bash
ros2 launch ur5_bringup bringup_moveit_rviz.launch.py
```

then press play on the External Control program (`ur_drive.urp`) on the pendant.
Full steps: [docs/robot_startup.md](docs/robot_startup.md). One-time network, pendant
and gripper setup: [docs/getting_started.md §5](docs/getting_started.md#5-one-time-setup-for-the-real-robot).

Only one bringup may talk to the robot and gripper at a time.

## Notes

- **Joint and frame names all start with `left_`** (e.g. `left_robotiq_85_left_knuckle_joint`).
  This is the naming carried over from the original dual-arm setup; it is the arm
  this workspace drives regardless of the prefix.
- **Gripper position feedback is the real finger position.** The pinned upstream
  driver reads byte 4 of the gripper status (`gPO`).
