project(change_detection)
find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  sensor_msgs
  tf
)
catkin_python_setup()
catkin_package()
