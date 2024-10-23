from setuptools import find_packages, setup

package_name = 'person_detection_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'torch', 'torchvision', 'numpy', 'pyrealsense2'],
    zip_safe=True,
    maintainer='anonione',
    maintainer_email='antonione@gmail.com',
    description='Pacote ROS 2 para detecção de pessoas e controle com Realsense e Detectron2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_detector_cpu = person_detection_control.person_detector_cpu:main',
            'person_detector_gpu = person_detection_control.person_detector_gpu:main',
        ],
    },
)
