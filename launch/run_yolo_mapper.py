import os
os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='slam_deep_mapper',
            executable='yolo_mapper',
            name='yolo_mapper',
            output='screen',
            parameters=[{'name_of_stero_image_topic': '/orbslam3/georeferenced_stereo_image'},
                        {'model_yolo_path': '/datadisk/data/agh_projects/dydaktyka/street_view_project/dnn/yolo11m-seg.pt'},
                        {'model_depth_path': '/home/kuba/dev/projects/ros2_jazzy_vimbax_ws/metric3d_convnext_large.onnx'},
                        {'confidence_threshold': 0.62},
                        {'publish_visualizations': True},
                        {'save_visualizations': False},
                        {'output_directory': '/datadisk/data/agh_projects/yolo_mapper_project/results/yolo_detections/'},
                        {'depth_estimation_roi_rows': [28,929]},
                        {'depth_estimation_roi_cols': [87,1280]},
                        {'camera_constant_image_left': [562.32565, 562.17107]},
                        {'camera_constant_image_right': [551.66086, 551.60775]},
                        {'max_depth_meters': 10.0}],
            emulate_tty=True
        )])
