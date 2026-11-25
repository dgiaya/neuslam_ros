#!/usr/bin/env bash
# Build the image with NeuSLAM at the version in neuslam.repos (git archive: uncommitted changes never get in).
#   docker/build.sh                                   x86_64
#   BASE=<jetson image> TORCH_FROM_PIP=0 NAV2_MSGS_FROM_SOURCE=1 docker/build.sh    Jetson
#   NeuSLAM from: NEUSLAM_SRC=<dir>, else ./neuslam (a release archive), else NEUSLAM_REPO (default ~/Code/Dynamic_SLAM)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version=$(awk '/version:/{print $2; exit}' "$here/neuslam.repos")
url=$(awk '/url:/{print $2; exit}' "$here/neuslam.repos")
repo=${NEUSLAM_REPO:-$HOME/Code/Dynamic_SLAM}
tag=${IMAGE:-neuslam_ros:latest}
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/neuslam_src" "$ctx/docker"
src=${NEUSLAM_SRC:-}
[[ -z "$src" && -f "$here/neuslam/pyproject.toml" ]] && src="$here/neuslam"
if [[ -n "$src" ]]; then
  commit=$(cat "$src/NEUSLAM_COMMIT" 2>/dev/null || git -C "$src" rev-parse --short HEAD 2>/dev/null || echo source)
  tar -C "$src" --exclude=.git -cf - . | tar -x -C "$ctx/neuslam_src"
else
  [[ -d "$repo/.git" ]] || git clone "$url" "$repo"
  git -C "$repo" fetch --quiet origin || true
  commit=$(git -C "$repo" rev-parse "$version^{commit}" 2>/dev/null || git -C "$repo" rev-parse "origin/$version^{commit}")
  git -C "$repo" archive "$commit" | tar -x -C "$ctx/neuslam_src"
fi
cp -r "$here/neuslam_msgs" "$here/neuslam_ros" "$ctx/"
rm -rf "$ctx/neuslam_ros/models"/*
cp "$here/docker/Dockerfile" "$here/docker/constraints.txt" "$here/docker/entrypoint.sh" "$ctx/docker/"
docker build -f "$ctx/docker/Dockerfile" -t "$tag" \
  --build-arg BASE="${BASE:-ros:jazzy-ros-base}" --build-arg TORCH_FROM_PIP="${TORCH_FROM_PIP:-1}" \
  --build-arg NAV2_MSGS_FROM_SOURCE="${NAV2_MSGS_FROM_SOURCE:-0}" --build-arg NEUSLAM_COMMIT="$commit" "$ctx"
echo "built $tag with NeuSLAM $commit"
