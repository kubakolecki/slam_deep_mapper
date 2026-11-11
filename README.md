## About slam_deep_mapper
slam_deep_mapper is part of NEU_DEPTH project, which aim is a fusion of neural depth estimation and sparse map. Sparse map in the typical scenario is provided by visual (or visual-inertial) SLAM, like ORB-SLAM.
slam_deep_mapper is responsible for the depth inference. There is also YOLO object detection functionality available.

## Dependencies

slam_deep_mapper depends on some ROS2 messages that are defined in [ros_common_messages](https://github.com/kubakolecki/ros_common_messages) package. You need to build this package first.

slam_deep_mapper is ROS2 Python package. We tested in with ROS2 Jazzy and Python 3.12. It depends on Python packages listed below. In brackets we provide versions we tested the package with.  
- OpenCV (opencv-python==4.12.0.88)  used for image processing
- NumPy (numpy==2.2.6)
- SciPy (scipy==1.16.3)
- ONNX Rutime (onnxruntime-gpu==1.23.2) - for handling deep neural network models for depth inference
- Ultralytics (ultralytics==8.3.240) - for handling YOLO object detection and segementation

We also used NVIDIA CUDA for the real-time inference. On our machine the output of `nvcc --version` is:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```
Of course you can try using slam_deep_mapper with other version of CUDA

slam_deep_mapper requires also cv_bridge. See [this information](https://github.com/kubakolecki/depth_map_optimizer/blob/main/README.md#about-cv_bridge) about cv_bridge installation.

## Hardware Requirements
For the real time inference you need GPU. We tested the package with NVIDIA GeForce RTX 4070 Laptop GPU.

## Neural Depth Estimation models
We provide implementation of slam_deep_mapper that can use following models:  
- [Metric3D](https://github.com/YvanYin/Metric3D)
- [DepthPro](https://github.com/apple-aiml-research/ml-depth-pro)
- [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2)
- [UniDepth](https://github.com/lpiccinelli-eth/unidepth)

Each of those models requires different image preprocessing and different post processing of inference results. So for now we keep model specific implementations in different branches:
- `main`: Metric3D ConvNeXt models
- `feature/new_monodepth_models`: Metric3D ViT models
- `feature/depth_pro_models`: DepthPro models
- `feature/depth_anything`: DepthAnything models
- `feature/unidepth`: UniDepth models

In the future we will make this configurable via ROS2 launchfile.
slam_deep_mapper works only with models in ONNX format.
For now we also don't publish ONNX models we used. We will provide links soon. 
You can download [depth_pro models](https://huggingface.co/onnx-community/DepthPro-ONNX/tree/main/onnx) from Hugging Face.
Links for DepthPro ConvNeXt models should be available directly from DepthPro GitHub.
We were not able to use all models we wanted because of the limited RAM of the GPU. For example out of several DepthPro ONNX models published in Hugging Face we were able to use only model_fp16.onnx.

## Building
I assume you already have your ROS2 workspace. In the workspace you should have your packages located in the `src` directory, which is a standard way to organize ROS2 workspace.
In the terminal go to the workspace main directory and follow the commands below (of course if want test with dedicated model `git clone` with the proper branch).
```bash
cd src
git clone https://github.com/kubakolecki/slam_deep_mapper.git
cd ..
colcon build --packages-select slam_deep_mapper
source install/setup.bash
```

## Running
First you need to make sure that the node will see all required Python packages. If you use venv, remember to activate it.
There is a launchfile for starting the node: `ROS2_WORKSPACE/src/slam_deep_mapper/launch/run_dnn_mapper.py`. You need to set up some of the parameters before running the node:  
- `name_of_stero_image_topic` - Name of a topic the GeoreferencedStereoImage message is published to. This message is defined in [ros_common_messages](https://github.com/kubakolecki/ros_common_messages).
- `do_run_yolo_detection` - Set it to False if you are interested in depth inference only. Then you also don't need to provide `model_yolo_path` or adjust `confidence_threshold`.
- `publish_visualizations` and `save_visualizations` are for YOLO detection. Set them to false if you don't run object detection.
- `save_images` and `do_save_depth_maps` if set to true you will have depth maps along with source images saved to destination provided by: `path_to_save_images` and `path_to_save_depth_maps`. These options are useful for debuging.
- `depth_estimation_roi_rows` and `depth_estimation_roi_cols` determine the image ROI in which depth inference takes place. Setting ROI may be useful because some models, like Metric3D, are very sensitive to the presence of black pixels near the border of the image. Such pixels may appear as a result of prior image resampling like undistortion or stereo rectificaton. For example ORB-SLAM in the stereo mode runs both image resampling algorithms.
- `camera_constant_image_left`, `camera_constant_image_right`, `max_depth_meters` are currently not used

After running the node is waiting for the GeoreferencedStereoImage messages. 
The node will publish the results as the ImageBasedMappingData message in the `/slam_deep_mapper/mapping_data` topic.

We will provide some rosbags with GeoreferencedStereoImage messages soon.
