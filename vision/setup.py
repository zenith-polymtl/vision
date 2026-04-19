from setuptools import find_packages, setup

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/models', [
        'models/yolo_m_100_epoch.pt',
        'models/world_model.pt',
    ]),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='colin',
    maintainer_email='colinc131@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_download = vision.image_download:main',
            'rgb_yolo = vision.rgb_yolo:main',
            'yolo_3D = vision.yolo_3D:main',
            'stereo_yolo = vision.stereo_yolo:main',
            'z2i_pipeline = vision.z2i_pipeline:main',
            'extract_relative_pose = vision.extract_relative_pose:main',
            'detect_circle = vision.detect_circle:main',
            'image_overlay = vision.image_overlay:main',
            'system_transformation = vision.system_transformation:main',
            'test_publisher = vision.test_publisher:main',
            'scene_description = vision.scene_description:main',
        ],
    },
)
