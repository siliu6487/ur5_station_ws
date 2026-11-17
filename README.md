# ur5_station_ws

ROS 2 Humble workspace for controlling one UR5 arm with a Robotiq 2F-85 gripper
and tactile sensors (DIGIT, GelSight Mini) from Python.

## Quickstart

Requires Ubuntu 22.04 with [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
installed. Use a fresh terminal that does **not** source any other ROS workspace.

```bash
git clone --recursive git@github.com:siliu6487/ur5_station_ws.git ~/workspace/ur5_station_ws
cd ~/workspace/ur5_station_ws
./setup.sh                      # apt + rosdep + udev rules + build; safe to re-run
```

Then, in every new terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/ur5_station_ws/install/setup.bash
```

Try everything on fake hardware first — nothing moves:

```bash
ros2 launch ur5_bringup bringup_moveit_rviz.launch.py use_fake_hardware:=true
```

and work through `src/ur5_station/notebooks/ur5_station_demo.ipynb` with `MODE = 'fake'`.

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

One-time setup that a script cannot do:

1. **Teach pendant:** install the *External Control* URCap and set its host IP to
   this PC (default `182.22.22.1`). Robot IP defaults to `182.22.22.2`.
2. **Network:** give this PC's robot-facing interface a static IP on that subnet.
3. **Gripper:** plug in its USB-RS485 adapter; `setup.sh` installed a udev rule that
   names it `/dev/robotiq_gripper`. Check with `ls -l /dev/robotiq_gripper`.
4. **Safety:** know where the e-stop is. Run every new motion on fake hardware first,
   then on the real robot at reduced speed.

```bash
ros2 launch ur5_bringup bringup_moveit_rviz.launch.py
```

then press play on the External Control program on the pendant.

Only one bringup may talk to the robot and gripper at a time.

## Notes

- **Joint and frame names all start with `left_`** (e.g. `left_robotiq_85_left_knuckle_joint`).
  This is the naming carried over from the original dual-arm setup; it is the arm
  this workspace drives regardless of the prefix.
- **Gripper position feedback is the real finger position.** The pinned upstream
  driver reads byte 4 of the gripper status (`gPO`).
