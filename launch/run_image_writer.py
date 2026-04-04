import os
os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='slam_deep_mapper',
            executable='stereo_image_writer',
            name='stereo_image_writer',
            output='screen',
            parameters=[{'name_of_stero_image_topic': '/orbslam3/georeferenced_stereo_image'},
                        {'output_directory': '/datadisk/data/agh_projects/yolo_mapper_project/results/images'}],
            emulate_tty=True
        )])
