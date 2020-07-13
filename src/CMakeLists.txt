# Compile the ecto 
find_package(PCL 1.9 REQUIRED)
find_package(CUDA REQUIRED)

add_compile_options(-std=c++11)

if(LINEMOD_VIZ_IMG)
  add_definitions("-DLINEMOD_VIZ_IMG=1")
endif()

set(SRC_FILES db_linemod.cpp
              linemod_detect.cpp
              linemod_train.cpp
              ModelFiller.cpp
              module.cpp
              linemod_icp.cpp
              PCL_ICP.cpp
)
set(INCLUDE_DIRS ${catkin_INCLUDE_DIRS}
                 ${OpenCV_INCLUDE_DIRS}
                 ${PCL_INCLUDE_DIRS}
                 ${CUDA_INCLUDE_DIRS}
                 
)
set(LIBRARIES ${catkin_LIBRARIES}
              ${OpenCV_LIBRARIES}
              ${CUDA_LIBRARIES} 
              ${CUDA_CUFFT_LIBRARIES} 
              ${CUDA_CUBLAS_LIBRARIES} 
              ${CUDA_cupti_LIBRARY} 
              ${CUDA_curand_LIBRARY} 
              ${CUDA_cusparse_LIBRARY} 
              ${PCL_LIBRARIES}
)

set(SRC_FILES ${SRC_FILES} linemod_pointcloud.cpp)

find_package(roscpp)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${roscpp_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${roscpp_LIBRARIES})

add_definitions("-DLINEMOD_VIZ_PCD=1")

include_directories(BEFORE ${INCLUDE_DIRS})
ectomodule(ecto_linemod DESTINATION ${PROJECT_NAME}/ecto_cells
                        INSTALL
                        ${SRC_FILES}
)

link_ecto(ecto_linemod ${LIBRARIES})
