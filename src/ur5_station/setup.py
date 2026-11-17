from glob import glob

from setuptools import find_packages, setup

package_name = "ur5_station"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="si liu",
    maintainer_email="siliu64cs@gmail.com",
    description="Task-independent Python interface to the UR5 station.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "digit_live_view = ur5_station.tools.digit_live_view:main",
            "tactile_live_view = ur5_station.tools.tactile_live_view:main",
        ],
    },
)
