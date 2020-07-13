cmake_minimum_required(VERSION 2.8)
project(object_recognition_linemod)

option(LINEMOD_VIZ_IMG "visualize images with detected objects features" ON)
option(LINEMOD_VIZ_PCD "visualize point clouds from the reference image and its match from DB" ON)

find_package(catkin REQUIRED COMPONENTS 
              object_recognition_core 
              object_recognition_renderer 
              opencv_candidate
              cv_bridge
              image_transport
              roscpp
              rospy
              tf
              tf2_eigen
              std_msgs
              sensor_msgs
              makino_commander)
catkin_package(CATKIN_DEPENDS ecto object_recognition_core object_recognition_renderer opencv_candidate
               DEPENDS OpenCV
)

#install targets for all things  python
catkin_python_setup()

# Set cmake.module path independently.
#LIST( APPEND CMAKE_MODULE_PATH /home/simon/Dev/g2o/cmake_modules )  
#SET( G2O_ROOT /usr/local/include/g2o )
#find_package(G2O REQUIRED)
#IF(G2O_FOUND)
#    include_directories(${G2O_INCLUDE_DIR})
#    message("G2O lib is found:" ${G2O_INCLUDE_DIR})
#ENDIF(G2O_FOUND)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

include_directories(include)
include_directories(SYSTEM ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS}                        
)

# Compile the module
add_subdirectory(python)
add_subdirectory(src)

# add the tests
if(CATKIN_ENABLE_TESTING)
  add_subdirectory(test)
endif()

# build the docs
ecto_sphinx(${CMAKE_CURRENT_SOURCE_DIR}/doc/source doc)

# install the conf files
install(DIRECTORY conf
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# add_dependencies(object_recognition_core makino_commander_gencpp)