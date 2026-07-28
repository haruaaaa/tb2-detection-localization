"""
Мини-пайплайн для квала на записанном бэге (лидар неподвижен):
    ros2 bag play <bag>  ->  clustering/cluster_node  ->  obstacle_detector/detector_node

Запуск:
    ros2 launch obstacle_detector detect_from_bag.launch.py
    ros2 launch obstacle_detector detect_from_bag.launch.py bag:=/root/ws/src/bags/<другой_бэг> loop:=false
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import os


def generate_launch_description():
    # бэг лежит локально в src/bags (см. README); в контейнере это /root/ws/src/bags
    default_bag = '/root/ws/src/bags/rosbag2_2026_06_27-18_43-mov_01'
    pkg_share = FindPackageShare('obstacle_detector').find('obstacle_detector')
    config_path = os.path.join(pkg_share, 'config', 'config.rviz')
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.rviz')

    bag = LaunchConfiguration('bag')
    loop = LaunchConfiguration('loop')
    rate = LaunchConfiguration('rate')
    use_rviz = LaunchConfiguration('rviz')
    
    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value=default_bag,
                              description='путь к rosbag2 (папка с .mcap)'),
        DeclareLaunchArgument('loop', default_value='true',
                              description='проигрывать бэг по кругу'),
        DeclareLaunchArgument('rate', default_value='1.0',
                              description='скорость проигрывания'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='поднять rviz2'),

        ExecuteProcess(
            cmd=['bash', '-c', PythonExpression([
                "'ros2 bag play ' + '", bag, "' + ' --rate ' + '", rate,
                "' + (' --loop' if '", loop, "' == 'true' else '')"
            ])],
            output='screen',
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0.2', '0', '0', '0', 'base_link', 'livox'],
            parameters=[{'use_sim_time': True}]
        ),

        Node(
            package='clustering', 
            executable='cluster_node',
            name='point_cloud_cluster', 
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),

        Node(
            package='obstacle_detector',
            executable='detector_node',
            output='screen',
            parameters=[{
                'use_sim_time': True, 
                'target_frame': 'base_link',
                'detected_robot_frame': 'turtlebot'
            }]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', config_path],
            name='rviz2', condition=IfCondition(use_rviz)),
    ])
