from setuptools import find_packages, setup

package_name = 'ros2_change_detect'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='emartinso',
    maintainer_email='emartinso@ltu.edu',
    description='Associated tools for utilizing change detection in ROS2',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'change_server = ros2_change_detect.change_server:main',
        ],
    },
)
