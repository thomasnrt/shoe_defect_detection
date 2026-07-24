from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    # Package paths
    pkg_env = get_package_share_directory('ur3e_env')
    pkg_ign = get_package_share_directory('ros_ign_gazebo')
    pkg_ur_desc = get_package_share_directory('ur_description')

    # Paths
    world_path = os.path.join(pkg_env, "world", "lab.world")
    table_sdf = os.path.join(pkg_env, "models", "table", "model.sdf")
    shoe_sdf = os.path.join(pkg_env, "models", "shoe", "model.sdf")
    ur3e_xacro = os.path.join(pkg_ur_desc, "urdf", "ur.urdf.xacro")
    ur3e_urdf = os.path.join(pkg_env, "urdf", "ur3e_robot.urdf")

    # 1) Convert Xacro to URDF at runtime
    xacro_convert = ExecuteProcess(
        cmd=['xacro', ur3e_xacro, '-o', ur3e_urdf],
        output='screen'
    )

    # 2) Launch Ignition Gazebo
    ign_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ign, 'launch', 'ign_gazebo.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )

    # 3) Spawn table first
    spawn_table = Node(
        package='ros_ign_gazebo',
        executable='create',
        arguments=[
            '-file', table_sdf,
            '-name', 'table',
            '-x', '0.7',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # 4) Spawn UR3e robot above table
    spawn_ur3e = Node(
        package='ros_ign_gazebo',
        executable='create',
        arguments=[
            '-file', ur3e_urdf,
            '-name', 'ur3e',
            '-x', '0.7',
            '-y', '0.0',
            '-z', '0.75'  # slightly above table top
        ],
        output='screen'
    )

    # 5) Spawn shoe above table
    spawn_shoe = Node(
        package='ros_ign_gazebo',
        executable='create',
        arguments=[
            '-file', shoe_sdf,
            '-name', 'shoe',
            '-x', '0.7',
            '-y', '0.0',
            '-z', '0.85'
        ],
        output='screen'
    )

    return LaunchDescription([
        xacro_convert,
        ign_gazebo,
        spawn_table,
        spawn_ur3e,
        spawn_shoe
    ])

