from glob import glob

from setuptools import find_packages, setup

package = "neuslam_ros"

setup(
    name=package,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/launch", glob("launch/*.py")),
        (f"share/{package}/config", glob("config/*.yaml")),
        (f"share/{package}/models", [f for f in glob("models/*") if not f.endswith("README.md")]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dennis Giaya",
    maintainer_email="giaya.d@northeastern.edu",
    description="ROS 2 node for NeuSLAM",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "neuslam_node = neuslam_ros.node:main",
            "voxel_map = neuslam_ros.voxel_map:main",
            "bag_runner = neuslam_ros.bag_runner:main",
        ],
    },
)
