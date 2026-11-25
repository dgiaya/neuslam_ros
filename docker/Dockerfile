# neuslam_ros on ROS 2 Jazzy. Build with docker/build.sh (it exports NeuSLAM into the build context).
#   x86_64: BASE=ros:jazzy-ros-base (default); PyTorch and TensorRT from pip.
#   Jetson: BASE=<a jetson-containers image with ROS 2 Jazzy, PyTorch and TensorRT>, TORCH_FROM_PIP=0.
ARG BASE=ros:jazzy-ros-base
FROM ${BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TORCH_FROM_PIP=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG NAV2_MSGS_FROM_SOURCE=0
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git libopencv-dev python3-pip python3-dev \
      ros-${ROS_DISTRO}-message-filters ros-${ROS_DISTRO}-tf2-ros-py ros-${ROS_DISTRO}-rosbag2-py \
      ros-${ROS_DISTRO}-rosbag2-storage-mcap ros-${ROS_DISTRO}-diagnostic-msgs ros-${ROS_DISTRO}-geographic-msgs \
 && if [ "${NAV2_MSGS_FROM_SOURCE}" = "0" ]; then apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-nav2-msgs; fi \
 && rm -rf /var/lib/apt/lists/*

# numpy stays at the distribution's 1.x: ROS's message bindings are built against it.
COPY docker/constraints.txt /opt/neuslam/constraints.txt
RUN python3 -c "import numpy; print('numpy==' + numpy.__version__)" >> /opt/neuslam/constraints.txt \
 && if [ "${TORCH_FROM_PIP}" = "1" ]; then \
      pip install -c /opt/neuslam/constraints.txt torch --index-url ${TORCH_INDEX} \
      && pip install -c /opt/neuslam/constraints.txt "tensorrt-cu12==10.15.*"; fi

COPY neuslam_src /opt/neuslam/src
RUN pip install -c /opt/neuslam/constraints.txt "opencv-python<5" "scipy>=1.10" "pillow>=9" pyyaml munch loguru \
      "gtsam>=4.2" evo huggingface-hub onnx scikit-build-core pybind11 \
 && pip install -c /opt/neuslam/constraints.txt --no-build-isolation /opt/neuslam/src/third_party/pydbow3 \
 && pip install --no-deps /opt/neuslam/src \
 && python3 -c "import gtsam, pydbow3, tensorrt, torch, neuslam; print('torch', torch.__version__, 'tensorrt', tensorrt.__version__)"

WORKDIR /ws
RUN if [ "${NAV2_MSGS_FROM_SOURCE}" = "1" ]; then \
      git clone --depth 1 -b ${ROS_DISTRO} https://github.com/ros-navigation/navigation2.git /tmp/nav2 \
      && mkdir -p src && mv /tmp/nav2/nav2_msgs /tmp/nav2/nav2_common src/ && rm -rf /tmp/nav2; fi
COPY neuslam_msgs src/neuslam_msgs
COPY neuslam_ros src/neuslam_ros
RUN rm -rf src/neuslam_ros/models/* \
 && source /opt/ros/${ROS_DISTRO}/setup.bash \
 && colcon build --cmake-args -DBUILD_TESTING=OFF \
 && rm -rf build log

ARG NEUSLAM_COMMIT=unknown
LABEL neuslam.commit=${NEUSLAM_COMMIT}
ENV NEUSLAM_MODELS=/models NEUSLAM_COMMIT=${NEUSLAM_COMMIT}
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "neuslam_ros", "neuslam.launch.py"]
