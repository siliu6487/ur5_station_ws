#!/usr/bin/env bash
# One-shot setup for ur5_station_ws on Ubuntu 22.04 + ROS 2 Humble.
# Safe to re-run: every step checks first and skips what is already done.
#
#   ./setup.sh               # robot + gripper + DIGIT/GelSight
#   ./setup.sh --with-audio  # also the ReSpeaker microphone (audio_common)
#
# Prerequisite (not done here): ROS 2 Humble installed at /opt/ros/humble
#   https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_AUDIO=0
for arg in "$@"; do
  case "$arg" in
    --with-audio) WITH_AUDIO=1 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m %s\n' "$*"; }

# ── 0. platform ────────────────────────────────────────────────────────────────
step "Checking platform"
. /etc/os-release
[ "${VERSION_ID:-}" = "22.04" ] && ok "Ubuntu 22.04" || warn "expected Ubuntu 22.04, found ${PRETTY_NAME:-unknown}"
[ -f /opt/ros/humble/setup.bash ] || { echo "ROS 2 Humble not found at /opt/ros/humble — install it first."; exit 1; }
ok "ROS 2 Humble"
if [ -n "${COLCON_PREFIX_PATH:-}${AMENT_PREFIX_PATH:-}" ] && [ "${AMENT_PREFIX_PATH:-}" != "/opt/ros/humble" ]; then
  warn "another workspace is sourced in this shell:"
  warn "  ${AMENT_PREFIX_PATH:-}"
  warn "build from a fresh terminal that sources only /opt/ros/humble, or packages may resolve to the wrong place"
fi

# ── 1. build tools ─────────────────────────────────────────────────────────────
step "Build tools (apt)"
need=()
for p in python3-vcstool python3-rosdep python3-colcon-common-extensions git; do
  dpkg -s "$p" >/dev/null 2>&1 || need+=("$p")
done
if [ ${#need[@]} -gt 0 ]; then
  sudo apt-get update && sudo apt-get install -y "${need[@]}"
fi
ok "vcstool, rosdep, colcon"

# ── 2. source dependencies ─────────────────────────────────────────────────────
step "Source dependencies"
cd "$WS"
git submodule update --init --recursive
# From ros2_robotiq_gripper we build ONLY robotiq_driver (+ its tests) from
# source. robotiq_description and robotiq_controllers come from apt
# (ros-humble-robotiq-*): ur5.urdf.xacro uses the apt macro's parameters
# (fake_sensor_commands), which the pinned source commit renamed.
for p in robotiq_description robotiq_controllers; do
  touch "src/ros2_robotiq_gripper/$p/COLCON_IGNORE"
done
ok "submodules: ros2_robotiq_gripper (driver only), serial"
mkdir -p src/deps
vcs import --skip-existing src/deps < deps.repos
[ "$WITH_AUDIO" = 1 ] && vcs import --skip-existing src/deps < deps-audio.repos
ok "deps.repos$([ "$WITH_AUDIO" = 1 ] && echo ' + deps-audio.repos')"

# ── 3. system dependencies (rosdep) ────────────────────────────────────────────
step "System dependencies (rosdep)"
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update --rosdistro humble >/dev/null
# shellcheck disable=SC1091
set +u; source /opt/ros/humble/setup.bash; set -u
rosdep install --from-paths src --ignore-src --rosdistro humble -y
ok "rosdep"

# ── 4. device access ───────────────────────────────────────────────────────────
step "Device access (udev rules, groups)"
for rule in src/ur5/ur5_bringup/udev/*.rules; do
  dst="/etc/udev/rules.d/$(basename "$rule")"
  if ! cmp -s "$rule" "$dst"; then
    sudo cp "$rule" "$dst"
    RELOAD=1
  fi
  ok "$(basename "$rule")"
done
if [ "${RELOAD:-0}" = 1 ]; then
  sudo udevadm control --reload-rules && sudo udevadm trigger
fi
for g in dialout video; do
  if id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then
    ok "in group $g"
  else
    sudo usermod -aG "$g" "$USER"
    warn "added $USER to $g — LOG OUT AND BACK IN for it to take effect"
  fi
done

# ── 5. build ───────────────────────────────────────────────────────────────────
step "Building (colcon)"
colcon build --symlink-install
ok "build"

cat <<EOF

Done. In every new terminal:

    source /opt/ros/humble/setup.bash
    source $WS/install/setup.bash

Next: docs/getting_started.md (simulation + one-time robot setup),
then docs/robot_startup.md (real robot).
EOF
