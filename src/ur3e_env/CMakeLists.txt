cmake_minimum_required(VERSION 3.5)
project(ur3e_env)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch world models urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

