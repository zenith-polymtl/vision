# Most Useful ZED ROS2 Topics

## Image Topics
- `/zed/zed_node/rgb/image_rect_color` — Rectified RGB color image *(zed_camera_component.cpp:3452–3454)*
- `/zed/zed_node/left/image_rect_color` — Left camera rectified image *(zed_camera_component.cpp:3446–3448)*
- `/zed/zed_node/right/image_rect_color` — Right camera rectified image *(zed_camera_component.cpp:3449–3451)*

## Depth and 3D Data
- `/zed/zed_node/depth/depth_registered` — Registered depth map *(zed_camera_component.cpp:3479–3480)*
- `/zed/zed_node/point_cloud/cloud_registered` — Colored 3D point cloud *(zed_camera_component.cpp:3484)*
- `/zed/zed_node/disparity/disparity_image` — Disparity image *(zed_camera_component.cpp:3469)*

## Position and Tracking
- `/zed/zed_node/pose` — Camera pose in world frame *(zed_camera_component.cpp:3499)*
- `/zed/zed_node/odom` — Visual odometry *(zed_camera_component.cpp:3509)*
- `/zed/zed_node/pose_with_covariance` — Pose with covariance matrix *(zed_camera_component.cpp:3501)*

## Sensors
- `/zed/zed_node/imu/data` — Processed IMU data *(zed_camera_component.cpp:3522–3524)*
- `/zed/zed_node/imu/data_raw` — Raw IMU data *(zed_camera_component.cpp:3523–3524)*
- `/zed/zed_node/temperature/imu` — IMU temperature *(zed_camera_component.cpp:3525–3526)*

## Object Detection and AI
- `/zed/zed_node/obj_det/objects` — Detected objects *(zed_camera_component.cpp:3488)*
- `/zed/zed_node/body_trk/skeletons` — Human skeleton tracking *(zed_camera_component.cpp:3491)*




## Flow de conversion xyz relatif du pointcloud

*Camera frame is FLU*
*Optical frame is RDF*

- Camera driver -> Image
- Image -> Pixels à analyser
- Pixels -> Pointcloud -> relative xyz RDF (right-down-forward)
- NavSatFix GNSS data + camera pose URDF -> TF from cam to drone ENU
- relative cam xyz -> TF -> ENU (m)


### Info on TF

```
# Get transform from camera optical to ENU map frame  
transform = tf_buffer.lookup_transform(  
    'map',                                    # Target frame (ENU)  
    'zed_left_camera_optical_frame',         # Source frame (RDF)  
    rospy.Time()  
)
```

