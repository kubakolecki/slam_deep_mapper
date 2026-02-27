import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msgs

from typing import Tuple, Dict, List

import os as os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.transform import RigidTransform as SE3
import onnxruntime as ort
from ultralytics import YOLO


from orbslam3.msg import GeoreferencedStereoImage
from depth_optimizer.msg import ObjectData

ONNX_INPUT_SIZE = (544, 1216) #These are values for Metric3D ConvNext Large model, according to: https://github.com/YvanYin/Metric3D/blob/main/onnx/test_onnx.py

def prepare_onnx_input(rgb_image: np.ndarray, input_size: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray], List[int]]:
    # implementation comes from: https://github.com/YvanYin/Metric3D/blob/main/onnx/test_onnx.py

    h, w = rgb_image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb: np.ndarray = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    onnx_input = {
        "image": np.ascontiguousarray(
            np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32
        ),  # 1, 3, H, W
    }
    return onnx_input, pad_info


class YoloMapper(Node):
    def __init__(self):
        super().__init__('yolo_mapper')

        #self.path_file_pointcloud = 'pointcloud.txt' #TODO only for debugging - remove later
        #self.file_pointcloud = open(self.path_file_pointcloud, 'w') #TODO only for debugging - remove later

        self.declare_parameter('name_of_stero_image_topic', '/orbslam3/georeferenced_stereo_image')
        self.declare_parameter('model_yolo_path', '/path/to/yolo/model')
        self.declare_parameter('model_depth_path', '/path/to/depth/model')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('output_directory', '/path/to/output/directory')
        self.declare_parameter('publish_visualizations', True)
        self.declare_parameter('save_visualizations', False)
        self.declare_parameter('depth_estimation_roi_rows', [28,929])
        self.declare_parameter('depth_estimation_roi_cols', [87,1280])
        self.declare_parameter('camera_constant_image_left', [562.32565, 562.17107]) 
        self.declare_parameter('camera_constant_image_right', [551.66086, 551.60775])
        self.declare_parameter('max_depth_meters', 10.0)

        model_yolo_path = self.get_parameter('model_yolo_path').get_parameter_value().string_value
        model_depth_path = self.get_parameter('model_depth_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        output_directory = self.get_parameter('output_directory').get_parameter_value().string_value
        self.do_publish_visualizations = self.get_parameter('publish_visualizations').get_parameter_value().bool_value
        self.do_save_visualizations = self.get_parameter('save_visualizations').get_parameter_value().bool_value
        self.depth_estimation_roi_rows = self.get_parameter('depth_estimation_roi_rows').get_parameter_value().integer_array_value
        self.depth_estimation_roi_cols = self.get_parameter('depth_estimation_roi_cols').get_parameter_value().integer_array_value
        self.camera_constant_image_left = self.get_parameter('camera_constant_image_left').get_parameter_value().double_array_value
        self.camera_constant_image_right = self.get_parameter('camera_constant_image_right').get_parameter_value().double_array_value
        self.max_depth_meters = self.get_parameter('max_depth_meters').get_parameter_value().double_value

        if not os.path.exists(output_directory):
            self.get_logger().fatal(f'Output directory {output_directory} does not exist.')
            raise FileNotFoundError(f'Output directory {output_directory} does not exist.') 
        
        if not os.path.isfile(model_yolo_path):
            self.get_logger().fatal(f'Model file {model_yolo_path} does not exist.')
            raise FileNotFoundError(f'Model file {model_yolo_path} does not exist.')
        
        if not os.path.isfile(model_depth_path):
            self.get_logger().fatal(f'Depth model file {model_depth_path} does not exist.')
            raise FileNotFoundError(f'Depth model file {model_depth_path} does not exist.')
        
        if self.depth_estimation_roi_rows[0] < 0 or self.depth_estimation_roi_rows[1] < 0 or self.depth_estimation_roi_cols[0] < 0 or self.depth_estimation_roi_cols[1] < 0:
            self.get_logger().fatal(f'ROI values for depth estimation must be non-negative.')
            raise ValueError(f'ROI values for depth estimation must be non-negative.')
        
        if self.depth_estimation_roi_rows[0] >= self.depth_estimation_roi_rows[1]:
            self.get_logger().fatal(f'Invalid ROI rows for depth estimation: start row must be less than end row.')
            raise ValueError(f'Invalid ROI rows for depth estimation: start row must be less than end row.')
        
        if self.depth_estimation_roi_cols[0] >= self.depth_estimation_roi_cols[1]:
            self.get_logger().fatal(f'Invalid ROI cols for depth estimation: start col must be less than end col.')
            raise ValueError(f'Invalid ROI cols for depth estimation: start col must be less than end col.')


        self.get_logger().info('YoloMapper node has been started.')
        self.get_logger().info(f'Using model path: {model_yolo_path}')


        self.get_logger().info(f'Setting up the directory structure')
        self.path_directory_visualizations = os.path.join(output_directory, "visualizations")
        #self.path_directory_images = os.path.join(output_directory, "images") #TODO only for debugging - remove later


        if not os.path.exists(self.path_directory_visualizations):
            os.makedirs(self.path_directory_visualizations)

        self.get_logger().info(f'Reading the YOLO model')
        self.model = YOLO(model_yolo_path)

        onnx_providers = [("CUDAExecutionProvider",{"cudnn_conv_use_max_workspace": "0", "device_id": str(0)})]
        self.onnx_session = ort.InferenceSession(model_depth_path, providers=onnx_providers)
   
        self.subscription_stereo_image = self.create_subscription(
            GeoreferencedStereoImage,
            self.get_parameter('name_of_stero_image_topic').get_parameter_value().string_value,
            self.process_stereo_image,
            10)
        
        self.publisher_detection_visualization = self.create_publisher(sensor_msgs.Image, 'slam_deep_mapper/yolo_visualization', 10)
        self.publisher_object_data = self.create_publisher(ObjectData, 'slam_deep_mapper/object_data', 10)
        self.bridge = CvBridge()
        self.get_logger().info('Waiting for GeoreferencedStereoImage messages...')

    def time_to_nanoseconds_string(self, stamp) -> str:
        total_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        return str(total_ns)
    
    def sample_depth_for_sparse_depth_information(self, sparse_depth_info:sensor_msgs.PointCloud, depth_map:np.ndarray):
        map_x = []
        map_y = []
        depth_values = []
        for point in sparse_depth_info.points:
            map_x.append(point.x)
            map_y.append(point.y)
            depth_values.append(point.z)
        interpolated_depth_values = cv2.remap(depth_map, np.asarray(map_x, dtype=np.float32), np.asarray(map_y, dtype=np.float32), interpolation=cv2.INTER_LINEAR)
        referece_depth_values = np.asarray(depth_values, dtype=interpolated_depth_values.dtype).reshape(interpolated_depth_values.shape)
        depth_differences = referece_depth_values - interpolated_depth_values

        #TODO: get rid of samples where depth is equal to zero!!!!!

        corrections = np.hstack((referece_depth_values, depth_differences))


        print(interpolated_depth_values.shape)
        print(referece_depth_values.shape)
        print(depth_differences.shape)
        print(corrections.shape)
        print(corrections)
        return interpolated_depth_values

    def process_stereo_image(self, stereo_image_message: GeoreferencedStereoImage):
        #self.get_logger().info('Processing GeoreferencedStereoImage message.')       
        image_left = self.bridge.imgmsg_to_cv2(stereo_image_message.image_left, desired_encoding='passthrough')
        if image_left.ndim == 2: #grayscale image
            image_left_color = np.stack((image_left, image_left, image_left), axis=-1)
        else: #color image
            image_left_color = image_left

  

        #self.get_logger().info(f'Got image with size:{image_left.shape}')
        #time_yolo_start = time.perf_counter()
        detection_results = self.model.predict(source=image_left_color, save = False, verbose = False, conf = self.confidence_threshold)
        #time_yolo_end = time.perf_counter()
        #self.get_logger().info(f"YOLO Runtime: {time_yolo_end - time_yolo_start:.4f} seconds")
        result = detection_results[0]
        has_detections = result.boxes.shape[0] > 0
        if has_detections:
            if self.do_publish_visualizations:
                #path_file_image = os.path.join(self.path_directory_images, 'image_' + self.time_to_nanoseconds_string(stero_image_message.pose.header.stamp)  + '.jpg' )
                # Save input image for debugging
                # cv2.imwrite(path_file_image, image_left_color)
                visualization_image = result.plot()
                # Publish visualization
                visualization_msg = self.bridge.cv2_to_imgmsg(visualization_image, encoding='bgr8')
                visualization_msg.header = stereo_image_message.pose.header
                self.publisher_detection_visualization.publish(visualization_msg)
                # Save visualization to file
                if self.do_save_visualizations:
                    path_file_results_visualization = os.path.join(self.path_directory_visualizations, 'visualization_' + self.time_to_nanoseconds_string(stereo_image_message.pose.header.stamp)  + '.jpg' )
                    cv2.imwrite(path_file_results_visualization, visualization_image)

            #depth estimation:
            image_input_depth_detection = image_left_color[self.depth_estimation_roi_rows[0]:self.depth_estimation_roi_rows[1], self.depth_estimation_roi_cols[0]:self.depth_estimation_roi_cols[1], :]


            #temporary code to save input images for debugging:
            #path_file_image = os.path.join(self.path_directory_visualizations, 'input_image_' + self.time_to_nanoseconds_string(stereo_image_message.pose.header.stamp)  + '.jpg' )
            #cv2.imwrite(path_file_image, image_input_depth_detection)  


            onnx_input, pad_info = prepare_onnx_input(image_input_depth_detection, ONNX_INPUT_SIZE)
            #time_depth_start = time.perf_counter()
            onnx_output = self.onnx_session.run(None, onnx_input)
            depth_map_result = onnx_output[0].squeeze()
            depth_map_result = depth_map_result[pad_info[0] : ONNX_INPUT_SIZE[0] - pad_info[1], pad_info[2] : ONNX_INPUT_SIZE[1] - pad_info[3]]
            depth_map_result = cv2.resize(depth_map_result, (image_input_depth_detection.shape[:2][1], image_input_depth_detection.shape[:2][0]), interpolation=cv2.INTER_LINEAR)
            depth_map = np.zeros(image_left_color.shape[:2], dtype=depth_map_result.dtype)
            depth_map[self.depth_estimation_roi_rows[0]:self.depth_estimation_roi_rows[1], self.depth_estimation_roi_cols[0]:self.depth_estimation_roi_cols[1]] = depth_map_result

            print(f"depth map is of type {depth_map.dtype} and has shape {depth_map.shape}")

            object_data_message = ObjectData()
            object_data_message.pose = stereo_image_message.pose
            object_data_message.sparse_depth_information = stereo_image_message.sparse_depth_information
            object_data_message.depth_map_left_row_major = depth_map.flatten().tolist() #we know that depth map has type float32
            object_data_message.depth_map_right_row_major = np.zeros(image_left_color.shape[:2], dtype=depth_map_result.dtype).flatten().tolist() #TODO: for now, we do not estimate depth for right image  
            object_data_message.rows = image_left_color.shape[0]
            object_data_message.columns = image_left_color.shape[1]
            object_data_message.depth_map_row_min = self.depth_estimation_roi_rows[0]
            object_data_message.depth_map_row_max = self.depth_estimation_roi_rows[1]
            object_data_message.depth_map_col_min = self.depth_estimation_roi_cols[0]
            object_data_message.depth_map_col_max = self.depth_estimation_roi_cols[1]
            object_data_message.focal_lenght_left = self.camera_constant_image_left
            object_data_message.focal_lenght_right = self.camera_constant_image_right

            image_segmented_by_classes = np.zeros(image_left_color.shape[:2], dtype=np.int32)
            list_of_classes = []

            for object_id, (class_id, mask) in enumerate(zip(result.boxes.cls, result.masks)):
                print(f"Processing object id {object_id} with class id {class_id.cpu()}")
                mask_array =  cv2.resize(np.squeeze(mask.data.cpu().numpy()), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                list_of_classes.append(np.int16(class_id.cpu()))
                image_segmented_by_classes[mask_array] = np.int16(object_id) + 1 #we start counting object ids from 1

     
            object_data_message.image_segmented_by_classes = image_segmented_by_classes.flatten().tolist()
            object_data_message.list_of_classes = list_of_classes
            object_data_message.number_of_objects = len(list_of_classes)
            print(f"Publishing ObjectData message with {len(list_of_classes)} objects detected.")
            self.publisher_object_data.publish(object_data_message)
            #the computations that follow will be done in anorther C++ node later on


            ##correction of depth map:
            ##depth_values = self.sample_depth_for_sparse_depth_information(stereo_image_message.sparse_depth_information, depth_map)
            ##print(depth_values)
#
#
            #valid_depth_mask = depth_map < self.max_depth_meters
            #
            #image_coordinates_y, image_coordinates_x = np.indices(depth_map.shape)
            #image_coordinates_y = image_coordinates_y.astype(np.float32)-(float(depth_map.shape[0])/2.0) + 0.5 #pixel coordinates relative to image center
            #image_coordinates_x = image_coordinates_x.astype(np.float32)-(float(depth_map.shape[1])/2.0) + 0.5 #pixel coordinates relative to image center
            #image_coordinates_y /= self.camera_constant_image_left[1] #these are normalized image coordinates
            #image_coordinates_x /= self.camera_constant_image_left[0] #these are normalized image coordinates
            #object_coordinates_y = depth_map * image_coordinates_y
            #object_coordinates_x = depth_map * image_coordinates_x
            #object_coordinates = np.stack((object_coordinates_x, object_coordinates_y, depth_map), axis=-1)
#
            ##self.get_logger().info(f'Shape of depth map: {depth_map.shape}')
#
            #rotation = Rotation.from_quat([
            #        stereo_image_message.pose.pose.orientation.x,
            #        stereo_image_message.pose.pose.orientation.y,
            #        stereo_image_message.pose.pose.orientation.z,
            #        stereo_image_message.pose.pose.orientation.w
            #    ], scalar_first = False )
#
            #se_3 = SE3.from_components(rotation = rotation, translation = [stereo_image_message.pose.pose.position.x, stereo_image_message.pose.pose.position.y, stereo_image_message.pose.pose.position.z])
            #transformation_matrix = se_3.as_matrix()
#
            ##self.get_logger().info(f'Transformation matrix:\n{transformation_matrix}')
#
#
            #for class_id, mask in zip(result.boxes.cls, result.masks):
            #    #class_id_int = int(class_id.item())
            #    mask_array =  cv2.resize(np.squeeze(mask.data.cpu().numpy()), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            #    points_in_3d_device_frame = object_coordinates[np.logical_and(mask_array, valid_depth_mask)]
            #    points_in_3d_device_frame_homogeneous = np.hstack((points_in_3d_device_frame, np.ones((points_in_3d_device_frame.shape[0], 1), dtype=points_in_3d_device_frame.dtype)))
            #    #TODO: here we can add depth filtering: removing some portion of points that are far
            #    points_in_world_frame_homogeneous = (transformation_matrix @ points_in_3d_device_frame_homogeneous.T).T
#
#
            #    #for point in points_in_world_frame_homogeneous[:, :3]:
            #    #    x, y, z = point
            #    #    self.file_pointcloud.write(f"{x:.4f},{y:.4f},{z:.4f},{int(class_id)}\n")
#
            #    #Here, you can save or process the object coordinates as needed
#
#
            ##At this point, we have depth_map, object_coordinates_x, object_coordinates_y arrays
#
#
            ##time_depth_end = time.perf_counter()
            ##self.get_logger().info(f"Depth ONNX Runtime: {time_depth_end - time_depth_start:.4f} seconds")





def main(args=None):
    rclpy.init(args=args)
    yolo_mapper = YoloMapper()
    rclpy.spin(yolo_mapper)
    yolo_mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()