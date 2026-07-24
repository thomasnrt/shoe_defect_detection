import os
from glob import glob
from setuptools import setup

package_name = 'shoe_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ↓↓↓ this line is what installs launch files ↓↓↓
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yassine',
    maintainer_email='yassine@example.com',
    description='YOLO-based shoe/hole detector for UR3e + RealSense',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_node = shoe_detector.yolo_node:main','grasp_detector = shoe_detector.grasp_detector:main',
        ],
    },
)
