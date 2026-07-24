terminal 1: ros2 run tf2_ros static_transform_publisher \
  --x -0.050036 \
  --y 0.042568 \
  --z 0.021518 \
  --qx -0.003951 \
  --qy -0.007457 \
  --qz -0.692347 \
  --qw 0.721515 \
  --frame-id tool0 \
  --child-frame-id camera_color_optical_frame

 
terminal 2: source /opt/ros/humble/setup.bash && source ~/new_ws/install/setup.bash && ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.1.102 launch_rviz:=false
 
terminal 3: ros2 launch ur_moveit_config ur_moveit.launch.py   ur_type:=ur3e   use_fake_hardware:=false   launch_rviz:=true
 
terminal 4: ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
 
terminal 5:  python3 ~/new_ws_compressed/new_ws/src/shoe_localization/localization_node.py
 
terminal 6: ros2 run rqt_image_view rqt_image_view   
 
terminal 7: # Terminal: shoe inspection
source /opt/ros/humble/setup.bash && source ~/new_ws/install/setup.bash
python3 ~/new_ws_compressed/new_ws/shoe_inspection_node.py
 
to trigger the inspection, use this : 
in another terminal: ros2 topic pub --once /start_photo_sequence std_msgs/msg/Bool "data: true"



camera collision box:# INCREASED SIZE: [Depth (X), Width (Y), Height/Thickness (Z)] in meters
        # 12cm deep, 16cm wide (to protect the USB cable!), 10cm thick
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.08, 0.13, 0.04])
        
        # POSITION OFFSET: Shifting the center of the box from the silver wrist flange
        pose = Pose()
        # Shift slightly to balance the USB cable sticking out
        pose.position.x = -0.05 
        pose.position.y = 0.0
        # Push the box outward so it covers the blue puck AND the camera
        pose.position.z = 0.03 
        pose.orientation.w = 1.0
        
        
        
better cable camera command:ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=1920x1080x30

