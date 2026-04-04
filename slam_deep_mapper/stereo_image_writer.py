import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msgs
import numpy as np
import cv2

from ros_common_messages.msg import GeoreferencedStereoImage

class StereoImageWriter(Node):
    def __init__(self):
        super().__init__('stereo_image_writer')
        self.declare_parameter('name_of_stero_image_topic', '/orbslam3/georeferenced_stereo_image')
        self.declare_parameter('output_directory', '/tmp/stereo_images/')
        

        self.subscription_stereo_image = self.create_subscription(
        GeoreferencedStereoImage,
        self.get_parameter('name_of_stero_image_topic').get_parameter_value().string_value,
        self.process_stereo_image,
        10)

        self.bridge = CvBridge()
        self.output_directory = self.get_parameter('output_directory').get_parameter_value().string_value
        self.counter = 0

        path_file_poses = self.output_directory + '/poses.txt'
        self.file_poses = open(path_file_poses, 'w')
        self.file_poses.write('timestamp_sec,timestamp_nanosec,x,y,z,qw,qx,qy,qz\n')

        self.get_logger().info('Waiting for stereo images on topic: ' + self.get_parameter('name_of_stero_image_topic').get_parameter_value().string_value) 

    def process_stereo_image(self, stereo_image_message: GeoreferencedStereoImage):

            image_left = self.bridge.imgmsg_to_cv2(stereo_image_message.image_left, desired_encoding='passthrough')
            image_right = self.bridge.imgmsg_to_cv2(stereo_image_message.image_right, desired_encoding='passthrough')
            if image_left.ndim == 2: #grayscale image
                image_left_color = np.stack((image_left, image_left, image_left), axis=-1)
            else: #color image
                image_left_color = image_left

            if image_right.ndim == 2: #grayscale image
                image_right_color = np.stack((image_right, image_right, image_right), axis=-1)
            else: #color image
                image_right_color = image_right

            timestamp_sec = str(stereo_image_message.pose.header.stamp.sec).zfill(10) #pad sec with leading zeros to ensure it has 10 digits
            timestamp_nanosec = str(stereo_image_message.pose.header.stamp.nanosec).zfill(9) #pad nanosec with leading zeros to ensure it has 9 digits
            timestamp_str = f"{timestamp_sec}_{timestamp_nanosec}"

            output_path_left = f"{self.output_directory}/left_{timestamp_str}.png"
            output_path_right = f"{self.output_directory}/right_{timestamp_str}.png"
            cv2.imwrite(output_path_left, image_left_color)
            cv2.imwrite(output_path_right, image_right_color)

            pose = stereo_image_message.pose.pose
            self.file_poses.write(f"{timestamp_sec},{timestamp_nanosec},{pose.position.x:.7f},{pose.position.y:.7f},{pose.position.z:.7f},{pose.orientation.w:.15f},{pose.orientation.x:.15f},{pose.orientation.y:.15f},{pose.orientation.z:.15f}\n")

            self.counter += 1
            self.get_logger().info(f"Saved image {self.counter}")


def main(args=None):
    rclpy.init(args=args)
    stereo_image_writer = StereoImageWriter()
    rclpy.spin(stereo_image_writer)
    stereo_image_writer.file_poses.close()
    stereo_image_writer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()