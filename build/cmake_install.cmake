# Install script for directory: /home/kentuen/ros_ws/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/kentuen/ros_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/kentuen/ros_ws/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/kentuen/ros_ws/install" TYPE PROGRAM FILES "/home/kentuen/ros_ws/build/catkin_generated/installspace/_setup_util.py")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/kentuen/ros_ws/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/kentuen/ros_ws/install" TYPE PROGRAM FILES "/home/kentuen/ros_ws/build/catkin_generated/installspace/env.sh")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/kentuen/ros_ws/install/setup.bash;/home/kentuen/ros_ws/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/kentuen/ros_ws/install" TYPE FILE FILES
    "/home/kentuen/ros_ws/build/catkin_generated/installspace/setup.bash"
    "/home/kentuen/ros_ws/build/catkin_generated/installspace/local_setup.bash"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/kentuen/ros_ws/install/setup.sh;/home/kentuen/ros_ws/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/kentuen/ros_ws/install" TYPE FILE FILES
    "/home/kentuen/ros_ws/build/catkin_generated/installspace/setup.sh"
    "/home/kentuen/ros_ws/build/catkin_generated/installspace/local_setup.sh"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/kentuen/ros_ws/install/setup.zsh;/home/kentuen/ros_ws/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/kentuen/ros_ws/install" TYPE FILE FILES
    "/home/kentuen/ros_ws/build/catkin_generated/installspace/setup.zsh"
    "/home/kentuen/ros_ws/build/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/kentuen/ros_ws/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/kentuen/ros_ws/install" TYPE FILE FILES "/home/kentuen/ros_ws/build/catkin_generated/installspace/.rosinstall")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/kentuen/ros_ws/build/gtest/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_common/baxter_common/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_common/baxter_description/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/robot_manipulation/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter/baxter_sdk/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_simulator/baxter_simulator/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/image_pipeline/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/moveit_python/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_common/rethink_ee_description/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_moveit_config/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_common/baxter_maintenance_msgs/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/beginner_tutorials/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/kcnet_garnet_project/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/camera_calibration/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/openni2/openni2_launch/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_common/baxter_core_msgs/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_interface/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_simulator/baxter_sim_io/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_tools/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/image_proc/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/image_publisher/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/openni2/openni2_camera/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/image_view/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/stereo_image_proc/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/depth_image_proc/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/glasgow_calibration/calibration_glasgow/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/pcl_cloud_point/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/image_pipeline-kinetic/image_rotate/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/glasgow_calibration/twodto3d/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_simulator/baxter_sim_controllers/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_simulator/baxter_gazebo/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_simulator/baxter_sim_kinematics/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_simulator/baxter_sim_hardware/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_simulator/baxter_sim_examples/cmake_install.cmake")
  include("/home/kentuen/ros_ws/build/baxter_examples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/kentuen/ros_ws/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
