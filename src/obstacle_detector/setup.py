import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'obstacle_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haruaha',
    maintainer_email='cesty_cat@mail.ru',
    description='Robot (TB2) detection among lidar clusters: wall-segment vs robot-circle + tracking.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detector_node = obstacle_detector.detector_node:main'
        ],
    },
)
