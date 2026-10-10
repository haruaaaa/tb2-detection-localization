cmake_minimum_required(VERSION 3.8)
project(turtlebot_description)

# Используем ament_cmake вместо catkin
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Устанавливаем файлы
install(DIRECTORY robots meshes test urdf
        DESTINATION share/${PROJECT_NAME}
)

# Тесты (опционально)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Обязательная команда для ament
ament_package()