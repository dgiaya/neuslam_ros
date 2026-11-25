#!/usr/bin/env bash
# Run a command in the image: GPU, host network and IPC (DDS), the package's models/ at /models.
#   docker/run.sh                                    the default launch
#   docker/run.sh ros2 launch neuslam_ros mavericks.launch.py
#   DATA=/path/to/bags docker/run.sh ros2 run neuslam_ros bag_runner --bag /data/<bag> ...
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tty=(); [[ -t 0 && -t 1 ]] && tty=(-it)
data=(); [[ -n "${DATA:-}" ]] && data=(-v "$DATA:/data")
exec docker run --rm "${tty[@]}" --init --gpus all --network host --ipc host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" ${RMW_IMPLEMENTATION:+-e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION"} \
  -v "$here/neuslam_ros/models:/models" -v "${RUNS:-$HOME/neuslam_runs}:/root/neuslam_runs" "${data[@]}" \
  "${IMAGE:-neuslam_ros:latest}" "$@"
