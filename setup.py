from setuptools import setup

setup(
    name='robot-arm-kinematic-calibration',
    version='0.0.1',
    description='Simple Python Library for Kinematic Calibration of Robot Arms (Serial Manipulators).',
    author='Philippe Nadeau',
    author_email='philippe.nadeau@robotics.utias.utoronto.ca',
    license='MIT',
    packages=['RobotKineCal'],
    install_requires=['roboticstoolbox-python'],
    python_requires=">=3.8",
    include_package_data=True
)