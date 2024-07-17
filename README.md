# Robot Arm Kinematic Calibration
This simple Python library implements the method described in [Local POE model for robot kinematic calibration](https://doi.org/10.1016/S0094-114X(01)00048-9) to determine the kinematic parameters of a robot arm from the nominal robot model and a set of end-effector poses.

By virtue of relying on [the robotics toolbox](https://petercorke.github.io/robotics-toolbox-python/intro.html) (RTB), this library supports all robot models defined in RTB, such as:
- Puma 560
- Franka robots
- Kinova Gen 3
- UR 3/5/10 robots
- Kuka LBR
- Fetch
- etc.

It is also possible to define a custom robot model with RTB and use this library to calibrate it.

Since the method used to produce the calibration data varies greatly between different robot setups, this library does not provide any tools to collect the calibration data. Instead, it expects the user to provide the calibration data in the form of a list of joint configurations and measured end-effector poses.

For instance, end-effector poses can be obtained from a motion capture system with markers attached to the robot end-effector, from a laser tracker, from a camera attached to the robot end-effector and observing a known pattern, etc. In all cases, the resulting dataset should contain the joint configurations and the corresponding end-effector poses in SE(3).

To facilitate the management of the end-effector poses, the [with-respect-to](https://github.com/PhilNad/with-respect-to) library is used and the poses are assumed to be named *ee-0*, *ee-1*, ..., *ee-n*.

## Usage Examples
See [Examples/FrankaSimulation.py](Examples/FrankaSimulation.py) for an example on how to use this library to calibrate a Franka robot model using a simulated dataset, and [Examples/FrankaReal.py](Examples/FrankaReal.py) for an example on how to use this library to calibrate a real Franka robot using a dataset collected with a camera mounted on the end-effector of the robot (eye-in-hand).

## Technical Details
The method described in [Local POE model for robot kinematic calibration](https://doi.org/10.1016/S0094-114X(01)00048-9) and used in this library is based on twists and on the product of exponentials (POE) formula for forward kinematics. Through an iterative least-squares optimization scheme, the twists representing perturbations to the pose of each link relative to the previous one are determined. Since the perturbations are relative to the previous link, the method is deemd *local*. This formulation greatly simplifies the implementation of the calibration algorithm. However, as pointed out in [this paper](https://doi.org/10.1109/TRO.2016.2593042), an equivalent formulation exists where less parameters are required (avoiding the introduction of redundant parameters and possibly slightly improving convergence speed). In practice, very few iterations are required to converge to a solution and the over-parametrization of the problem is not an issue. 
