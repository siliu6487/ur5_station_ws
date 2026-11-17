cmake_minimum_required(VERSION 3.8)
project(ur5_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/print_arm_joint_state.py
  scripts/publish_planning_scene.py
  scripts/test_sim_motion.py
  scripts/test_wrist_rotate.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
