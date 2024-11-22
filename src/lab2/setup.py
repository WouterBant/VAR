from setuptools import find_packages, setup

package_name = "lab2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="woute",
    maintainer_email="99251015+WouterBant@users.noreply.github.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "curling = lab2.curling_node:main",
            "image_saver = lab2.img_saver_node:main",
            "wheel_odometry = lab2.wheel_odometry_node:main",
        ],
    },
)
