cmake_minimum_required(VERSION 3.8)
project(aic_gazebo_transport_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)
set(GZ_TRANSPORT_VER ${gz-transport_VERSION_MAJOR})

find_package(gz_msgs_vendor REQUIRED)
find_package(gz-msgs REQUIRED)
set(GZ_MSGS_VER ${gz-msgs_VERSION_MAJOR})

add_executable(aic_gz_transport_bridge
  src/aic_gz_transport_bridge.cc
)

target_compile_features(aic_gz_transport_bridge PRIVATE cxx_std_17)
target_link_libraries(aic_gz_transport_bridge PRIVATE
  gz-transport${GZ_TRANSPORT_VER}::gz-transport${GZ_TRANSPORT_VER}
  gz-msgs${GZ_MSGS_VER}::gz-msgs${GZ_MSGS_VER}
)

install(
  TARGETS aic_gz_transport_bridge
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
