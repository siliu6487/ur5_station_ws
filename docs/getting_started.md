# Getting started on a new PC

For someone who has never used this workspace (or ROS) before: from a fresh Ubuntu install to a simulated robot moving in RViz, then the one-time setup needed before you can drive the real UR5. Nothing in sections 1–4 moves anything physical.

If the station PC is already set up and you just want to run the robot, go to [robot_startup.md](robot_startup.md) instead.

---

## Prerequisites

| What | Why / notes |
|---|---|
| A 64-bit x86 PC with **Ubuntu 22.04 LTS** (Jammy) | ROS 2 Humble only ships packages for 22.04. Check with `lsb_release -a`. Not a VM if you will drive the real robot: the real-time connection to the arm needs a direct Ethernet port. |
| An account with **sudo** rights | `setup.sh` installs packages and udev rules. |
| Internet access and ~10 GB free disk | ROS, MoveIt and the build. |
| **ROS 2 Humble** (Desktop) at `/opt/ros/humble` | Installed in [step 1](#1-install-ros-2-humble). |
| **git**, and access to the GitHub repo | `sudo apt install git`. Cloning with `git@github.com:...` needs an [SSH key added to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh); otherwise use the `https://` URL. |
| **Jupyter** (JupyterLab, or VS Code with the *Jupyter* extension) | For the demo notebook. Installed in [step 2](#2-get-the-workspace). |
| Basic terminal use | Opening several terminals, `cd`, copy-pasting commands, `Ctrl+C` to stop a program. |

For the real robot you will also need (section 5): an Ethernet cable from the PC to the UR5 control box, the gripper's USB-RS485 adapter, and access to the teach pendant.

---

## 1. Install ROS 2 Humble

Skip this if `ls /opt/ros/humble/setup.bash` already prints the file.

These are the commands from the [official install guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html). If any of them fails, follow that page instead; it is kept up to date.

```bash
# UTF-8 locale
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS 2 apt repository
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# ROS 2 Humble
sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-humble-desktop
```

Check it:

```bash
source /opt/ros/humble/setup.bash
ros2 --help        # prints the list of ros2 commands
```

**Don't add `source ...` lines for other workspaces to your `~/.bashrc`.** A second workspace sourced in the same shell can silently replace this workspace's packages with older copies.

## 2. Get the workspace

Open a **new** terminal (one that hasn't sourced any ROS workspace), `cd` to the folder you want the workspace in, and run:

```bash
git clone --recursive git@github.com:siliu6487/ur5_station_ws.git
# or, without an SSH key:
# git clone --recursive https://github.com/siliu6487/ur5_station_ws.git

cd ur5_station_ws
./setup.sh                      # apt + rosdep + udev rules + build; safe to re-run
```

`setup.sh` asks for your sudo password and takes 10–20 minutes the first time. It:
1. installs the build tools (colcon, rosdep, vcstool),
2. fetches the source dependencies (gripper driver, serial library, pymoveit2),
3. installs the system dependencies with `rosdep`,
4. installs the gripper's udev rule (so it appears as `/dev/robotiq_gripper`) and adds you to the `dialout` and `video` groups (serial ports and cameras),
5. builds everything into `install/`.

If it printed `added <you> to dialout` or `video`, **log out and back in** (or reboot) before continuing. Group changes only apply to new logins.

Add `--with-audio` (`./setup.sh --with-audio`) only if you will use the ReSpeaker microphone.

Install Jupyter (skip if you use VS Code's Jupyter extension):

```bash
sudo apt install -y python3-pip
python3 -m pip install --user jupyterlab
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # so the `jupyter` command is found
```

Use the system Python (`python3`, 3.10), not conda or a virtualenv: ROS's Python packages are built for it.

## 3. Source the workspace and check it

In **every new terminal** you use with the robot, go to the workspace folder (the `ur5_station_ws` folder you cloned) and run these two lines first:

```bash
cd <path-to>/ur5_station_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

All commands with relative paths below (`src/...`, `install/...`) are run from this folder. If you'd rather put the `source` lines in `~/.bashrc`, use the full path there (`source <path-to>/ur5_station_ws/install/setup.bash`), since `.bashrc` doesn't run from the workspace folder.

Check that the right copies of the packages are found:

```bash
ros2 pkg prefix ur5_bringup      # must print .../ur5_station_ws/install/ur5_bringup
python3 -c "import ur5_station; print(ur5_station.__file__)"   # must contain ur5_station_ws
```

If either path points somewhere else, another workspace is sourced in this terminal: open a fresh one, or remove the extra `source` line from `~/.bashrc`.

## 4. Try it in simulation (fake hardware)

The same launch file runs with simulated hardware: the same controllers and MoveIt, but nothing physical moves. Use it to learn the API and to try every new motion before running it on the real arm.

**Terminal A:**

```bash
ros2 launch ur5_bringup bringup_moveit_rviz.launch.py use_fake_hardware:=true
```

RViz opens and shows the arm. A back wall, a right wall and a table pad appear about 8 s later; MoveIt treats them as obstacles.

**Terminal B**, check that the controllers came up:

```bash
ros2 control list_controllers
```

All should be **active** except `freedrive_mode_controller` and `joint_trajectory_controller`, which are inactive by design.

**Terminal B**, start the demo notebook:

```bash
jupyter lab src/ur5_station/notebooks/ur5_station_demo.ipynb
```

The notebook is written for the real robot. On fake hardware, run:
- §2 Setup, and §3.1–3.6: the arm moves in RViz;
- §4: the first two gripper cells (`calibrate`, `open`);
- §5.1: joint-state recording;
- §8 Shutdown.

Skip the cells that need real hardware: **§3.7 Freedrive**, the **force-limited grasp** cell in §4, and §5.2 onward unless a DIGIT sensor is plugged in.

You can also run the short `first_move.py` script from [robot_startup.md, step 3](robot_startup.md#3-make-it-move) against the simulation.

When you're done, run the notebook's **§8 Shutdown** cells, shut the kernel down (**Kernel → Shut Down Kernel**), then `Ctrl+C` in terminal A.

## 5. One-time setup for the real robot

Do this once per PC and robot. A script can't do it for you.

### 5.1 Network

The robot is at `182.22.22.2`; the PC must be `182.22.22.1` on the Ethernet port the robot is plugged into.

1. Find the interface name: `ip -br link` (e.g. `enp3s0`, or `enx...` for a USB Ethernet adapter).
2. Give it a static address (replace `enp3s0`):
   ```bash
   sudo nmcli con add type ethernet ifname enp3s0 con-name ur5 \
     ipv4.method manual ipv4.addresses 182.22.22.1/24 ipv6.method disabled
   sudo nmcli con up ur5
   ```
   Or use **Settings → Network → Wired → IPv4 → Manual**: address `182.22.22.1`, netmask `255.255.255.0`, no gateway.
3. Check: `ping -c 3 182.22.22.2` should get replies once the robot's control box is on.

If you use a USB Ethernet adapter, plug it straight into the PC, not into a shared USB hub.

### 5.2 Teach pendant

1. Install the **External Control** URCap on the pendant (from [Universal_Robots_ROS2_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver), *Installing a URCap*), and set its host IP to **`182.22.22.1`**.
2. Create (or load) the program **`ur_drive.urp`** containing an External Control node. Its host must be **`182.22.22.1`**, port **`50002`**. It is saved in the program, so you only check it once.

### 5.3 Gripper

Plug in the gripper's USB-RS485 adapter and check:

```bash
ls -l /dev/robotiq_gripper       # a symlink to /dev/ttyUSBn
```

If it's missing, your adapter has a different serial number from the one in [99-robotiq-gripper.rules](../src/ur5/ur5_bringup/udev/99-robotiq-gripper.rules). Find yours with `udevadm info -a -n /dev/ttyUSB0 | grep '{serial}'`, put it in the rule, and re-run `./setup.sh`. Until then you can pass `gripper_com_port:=/dev/ttyUSB0` to the launch command.

### 5.4 Before the first real launch

```bash
ping -c 3 182.22.22.2            # robot answers
ls -l /dev/robotiq_gripper       # gripper port exists
pgrep -af "ros2 launch"          # prints nothing: no other bringup (e.g. the simulation) is running
```

Only **one** bringup can talk to the robot and gripper at a time, so stop the simulation first.

---

## Next

Go to [robot_startup.md](robot_startup.md) (power-on, first real motion and shutdown) and the demo notebook (its §1 launches the robot). The guide's [Troubleshooting](robot_startup.md#troubleshooting) table covers the common problems, including `/joint_states did not arrive within 15s`.
