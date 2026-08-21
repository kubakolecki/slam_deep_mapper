import os
os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='slam_deep_mapper',
            executable='dnn_mapper',
            name='dnn_mapper',
            output='screen',
            parameters=[{'name_of_stero_image_topic': '/orbslam3/georeferenced_stereo_image'},
                        {'model_yolo_path': '/datadisk/data/agh_projects/dydaktyka/street_view_project/dnn/yolo11m-seg.pt'},
                        {'model_depth_path': '/datadisk/data/depth_prediction_models/depth_pro/model_fp16.onnx'},
                        {'confidence_threshold': 0.62},
                        {'do_run_yolo_detection': False},
                        {'publish_visualizations': True},
                        {'save_visualizations': False},
                        {'save_images': True},
                        {'do_save_depth_maps': True},
                        {'path_to_save_images' : '/datadisk/data/agh_projects/20260601_depth_map_datasets/variant_depth_pro/depth_maps/2026_06_01-11_54_05'},
                        {'path_to_save_depth_maps': '/datadisk/data/agh_projects/20260601_depth_map_datasets/variant_depth_pro/depth_maps/2026_06_01-11_54_05'},
                        {'output_directory': '/datadisk/data/agh_projects/yolo_mapper_project/results/yolo_detections/'},
                        {'depth_estimation_roi_rows': [0,479]},
                        {'depth_estimation_roi_cols': [0,751]},
                        {'camera_constant_image_left': [456.71499, 456.71499]},
                        {'camera_constant_image_right': [456.71499, 456.71499]},
                        {'max_depth_meters': 10.0}],
            emulate_tty=True
        )])
