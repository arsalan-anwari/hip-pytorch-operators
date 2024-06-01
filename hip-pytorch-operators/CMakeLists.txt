# Only header target, does not include dependencies
add_library(hip-pytorch-operators INTERFACE)
target_include_directories(hip-pytorch-operators
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/hip-pytorch-operators>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

# This target links against HIP library
add_library(hip-pytorch-operators_hip INTERFACE)
target_link_libraries(hip-pytorch-operators_hip INTERFACE hip-pytorch-operators hip::device)
