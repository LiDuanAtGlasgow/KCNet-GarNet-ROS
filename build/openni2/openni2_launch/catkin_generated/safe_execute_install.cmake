execute_process(COMMAND "/home/kentuen/ros_ws/build/openni2/openni2_launch/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/kentuen/ros_ws/build/openni2/openni2_launch/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
