from setuptools import find_packages, setup

package_name = 'slam_deep_mapper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','ultralytics'],
    zip_safe=True,
    maintainer='kuba',
    maintainer_email='jjkolecki@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dnn_mapper = slam_deep_mapper.dnn_mapper:main',
            'stereo_image_writer = slam_deep_mapper.stereo_image_writer:main'
        ],
    },
)
