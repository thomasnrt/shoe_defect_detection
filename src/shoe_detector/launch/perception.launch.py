from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # 1. RealSense camera (publishes RGB + depth + camera_info)
    #    publish_tf:=false → we'll publish the chain ourselves via hand-eye calibration
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'depth_module.depth_profile': '640x480x30',
            'rgb_camera.color_profile': '640x480x30',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'enable_sync': 'true',
            'initial_reset': 'true',
            'publish_tf': 'false',   # we publish camera TF via hand-eye
        }.items(),
    )

    # 2. Hand-eye calibration: tool0 -> camera_color_optical_frame
    handeye_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='handeye_tf_publisher',
        arguments=[
            '--x', '0.07035858523181109',
            '--y', '-0.03786070717680278',
            '--z', '0.03091576198336038',
            '--qx', '-0.006237947432436568',
            '--qy', '-0.0006472618464794522',
            '--qz', '0.6941468239226309',
            '--qw', '0.7198061238292268',
            '--frame-id', 'tool0',
            '--child-frame-id', 'camera_color_optical_frame',
        ],
    )

    return LaunchDescription([
        realsense_launch,
        handeye_tf,
    ])
