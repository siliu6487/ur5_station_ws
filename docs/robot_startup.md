# Starting the UR5 and making it move

From a powered-off robot to a first motion, then back to off. Allow about 10 minutes the first time.

This guide assumes the station PC is already set up and tested. Setting up a new PC, or new to ROS? Start with [getting_started.md](getting_started.md).

You need:
- the PC connected to the robot by Ethernet (robot `182.22.22.2`, PC `182.22.22.1`)
- the gripper's USB cable plugged into the PC
- someone standing at the teach pendant (the robot's touchscreen tablet) who can reach the **emergency stop** (red E-STOP button)

---

## 0. Before anything moves

- Find the red **emergency stop** button on the pendant. Pressing it stops the arm immediately, so press it whenever something looks wrong. Twist it to release it again.
- Clear the space around the arm. The arm has hard-coded safety walls (a back wall, a right wall and the table), but they don't know about anything you've put on the table.
- Run every new motion slowly first (`velocity=0.05`) with a hand near the emergency stop.

---

## 1. Power up the robot (pendant)

1. Press the **power button** on the teach pendant and wait for it to boot.
2. On the warning screen, choose the option to go to **initialization**.
3. Tap **ON** to power the arm.
4. Tap **START** to release the brakes. The arm clicks and may twitch slightly. This is normal.
5. Tap **OK** (bottom right).

## 2. Load the External Control program (pendant)

The PC doesn't control the arm directly. A program running on the pendant, called **External Control**, receives motion commands from the PC. Load it now, but **don't press play yet**: the PC has to be listening first (notebook §1, step 2/3).

1. On the main screen, tap **Run Program**.
2. **File → Load Program**.
3. Select **`ur_drive.urp`** (the program containing the External Control node) and tap **Open**.

## 3. Make it move

> **Attention:** every terminal must load **only this workspace** (`ur5_station_ws`). If your `~/.bashrc` sources another ROS workspace, the old copies of these packages run silently in its place. Run these lines in EVERY new terminal, from the workspace folder (wherever you cloned `ur5_station_ws`):
>
> ```bash
> cd <path-to>/ur5_station_ws
> source /opt/ros/humble/setup.bash
> source install/setup.bash
> ```
>
> Or put them in `~/.bashrc` (replacing any other workspace `source` lines), using the full path there: `source <path-to>/ur5_station_ws/install/setup.bash`.

The guided way is the demo notebook. Open it in VS Code:
`src/ur5_station/notebooks/ur5_station_demo.ipynb` (inside the workspace folder)

or start it from a sourced terminal in the workspace folder:
```bash
jupyter lab src/ur5_station/notebooks/ur5_station_demo.ipynb
```

Run it top to bottom. §1 launches the robot from the PC and has you press play on the pendant, §3 moves the arm, §4 the gripper and §5 the sensors. **Before closing it, run §8 Shutdown** (see [4.1](#41-stop-the-ros-node-before-you-close-the-notebook)).

The minimum in plain Python (`python3 first_move.py` from a sourced terminal, with the robot launched and playing as in notebook §1, steps 2/3 and 3/3):

```python
import rclpy
from threading import Thread
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from ur5_station.robot_motion_client import RobotMotionClient
from ur5_station.gripper_client import GripperClient

rclpy.init()
node = Node('first_move')
cbg = ReentrantCallbackGroup()
executor = rclpy.executors.MultiThreadedExecutor(4)
executor.add_node(node)
Thread(target=executor.spin, daemon=True).start()   # keeps callbacks running during motions

gripper = GripperClient(node, cbg)
robot = RobotMotionClient(node, cbg, velocity=0.1, acceleration=0.1, gripper=gripper)
assert robot.wait_until_ready(timeout=15.0), 'no joint states: is the bringup running?'

robot.move_to_named_pose('home')                     # a safe, known pose
z_floor = robot.get_current_eef_pose()[0][2] - 0.05  # never go more than 5 cm below here
robot.move_eef_delta(z=+0.02)                        # up 2 cm
robot.move_eef_delta(z=-0.02, z_floor_m=z_floor)     # back down
gripper.open()

executor.shutdown()
rclpy.shutdown()
```

`velocity` and `acceleration` are fractions of the arm's maximum speed (0.1 = 10%), on top of the pendant's speed slider.

## 4. Shut down

### 4.1 Stop the ROS node before you close the notebook

> **Always run the notebook's §8 Shutdown cells (`executor.shutdown()` and `rclpy.shutdown()`)
> before you close the notebook.** Then stop the kernel: **Kernel → Shut Down Kernel**.

Closing the browser tab does **not** stop the kernel. It keeps running in the background with its ROS node still alive, and:
- it keeps the **DIGIT camera open**, so the next session can't open it;
- a second node with the same name stays on the ROS network, so tools and logs show two;
- it stays connected to the robot's controllers, out of sight.

If this has already happened, see the *DIGIT can't be opened* row under [Troubleshooting](#troubleshooting).

A plain Python script does the same cleanup with the last two lines (`executor.shutdown()` / `rclpy.shutdown()`).

### 4.2 Stop the bringup (PC)

Press `Ctrl+C` in the terminal running the bringup (notebook §1, step 2/3).

### 4.3 Turn off the robot (pendant)

1. Press **■** (the square Stop symbol) to stop the External Control program.
2. Tap **File → Exit**.
3. **Hold the arm with your hand**, then tap **Shut down robot**. The arm might sag as its power cuts out, so support it until it's fully off.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/joint_states did not arrive within 15s` | The notebook and the bringup can't see each other. Both must use the **same** `ROS_DOMAIN_ID` and source **this** workspace. Compare `echo $ROS_DOMAIN_ID` and `ros2 pkg prefix ur5_bringup` in both terminals. |
| Arm doesn't move, no error | External Control isn't playing (notebook §1, step 3/3), or the terminal never printed `Robot connected to reverse interface`. `ros2 control list_controllers` should show every controller **active** except `freedrive_mode_controller` and `joint_trajectory_controller`. |
| Pendant: *cannot connect* to External Control | The PC isn't listening yet (launch first, then press play), or the host/port in the program is wrong (`182.22.22.1:50002`). On the PC, `ss -ltnp \| grep 500` should show ports 50001–50004 while the bringup runs. |
| Launch can't reach the robot / `ping 182.22.22.2` fails / `No route to host` | Check the Ethernet cable and that the pendant is powered on. If the PC uses a USB Ethernet adapter, keep it off a shared USB hub. A RealSense camera on the same hub has knocked it offline before; plug it straight into the PC. |
| Gripper errors, `Modbus exception`, or no `/dev/robotiq_gripper` | Another process holds the port (`fuser /dev/robotiq_gripper`); only one bringup at a time. If the symlink is missing, re-run `./setup.sh` or pass `gripper_com_port:=/dev/ttyUSBn` (`ls /dev/ttyUSB*`). |
| DIGIT can't be opened / device busy, or two nodes with the same name | A notebook was closed without §8 Shutdown and its kernel is still running. Find what holds the camera with `fuser -v /dev/v4l/by-id/usb-Facebook_DIGIT_*-video-index0`, then shut that kernel down in Jupyter (**Running Terminals and Kernels** panel), or `kill <pid>`. |
| Arm moves very slowly | Pendant speed slider is low. `ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once` shows it (100.0 = full). |
| Arm stops mid-motion, pendant shows **Protective Stop** | It hit something or exceeded a limit. Clear the cause, acknowledge it on the pendant, press play again. Then restart from a safe pose (`move_to_named_pose('home')`). |
| `[safety] target Z ... would breach floor ... clamping` | A downward move would have gone below `z_floor_m`, so it stopped at the floor instead. Working as intended. |
