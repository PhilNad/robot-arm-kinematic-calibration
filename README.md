# Robot Arm Kinematic Calibration
This simple Python library implements the method described in [POE-based robot kinematic calibration using axis configuration space and the adjoint error model](https://doi.org/10.1109/TRO.2016.2593042) to determine the kinematic parameters of a robot arm from the nominal robot model and a set of end-effector observations.

**Features**
-------------
- Calibration of serial robots with revolute joints
- Observations can be position measurements or full SE(3) poses
- Many robot models supported out-of-the-box and custom URDF models can be used
- Minimal parametrization of the problem, yielding optimal convergence speed
- Can produce joint definitions for URDF files
----------------

By virtue of relying on [the robotics toolbox](https://petercorke.github.io/robotics-toolbox-python/intro.html) (RTB), this library supports all robot models defined in RTB, such as: Puma 560, Franka robots, Kinova Jaco, UR 3/5/10 robots, Kuka LBR, Fetch, etc.

It is also possible to define a custom robot model with RTB and use this library to calibrate it. For example with:
```python
import roboticstoolbox as rtb
from robotkinecal import SerialRobotKineCal
#Load custom robot model
nominal_robot_model = rtb.ERobot.URDF('Examples/gen3.urdf')
#Calibrate (assuming configurations and observed_ee_poses are defined)
cal = SerialRobotKineCal(nominal_robot_model, ee_name, verbose=True)
cal.set_observations(configurations, observed_ee_poses)
result = cal.solve()
```
See [Examples/FromURDF.py](Examples/FromURDF.py) for a complete example.

Since the method used to produce the calibration data varies greatly between different robot setups, this library does not provide any tools to collect the calibration data. Instead, it expects the user to provide input data in the form of a list of joint configurations and measured end-effector poses/positions.

For instance, observations can be obtained from a motion capture system with markers attached to the robot end-effector, from a laser tracker, from a camera attached to the robot end-effector and observing a known pattern, etc.

Each joint configuration must be paired with a measurement from the same stationary instant. Joint values must use radians, translations must use metres, and observations must be either `spatialmath.SE3` poses or NumPy position vectors with shape `(3,)`. Use actual encoder readings and configurations that exercise all calibrated joints across the intended workspace.

A recommended calibration workflow is:

1. Define the Base/EE frames, transform conventions, units, and calibration target.
2. Verify the nominal model and setup with simulated data.
3. Collect diverse, synchronized observations and reserve an independent validation set.
4. Run calibration and check convergence, rank, condition number, residuals, and parameter plausibility.
5. Validate the result before backing up and replacing the nominal model.

## Installation
Python 3.10 through 3.12 is supported.

Install from PyPI:
```bash
pip install robotkinecal
```
Or with [uv](https://docs.astral.sh/uv/):
```bash
uv add robotkinecal
```
You can then import the library in your Python scripts with:
```python
from robotkinecal import SerialRobotKineCal
```

## Usage Examples
See [Examples/FrankaSimulation.py](Examples/FrankaSimulation.py) for an example on how to use this library to calibrate a Franka robot model using a simulated dataset, and [Examples/FrankaReal.py](Examples/FrankaReal.py) for an example on how to use this library to calibrate a real Franka robot using a dataset collected with a camera mounted on the end-effector of the robot (eye-in-hand). Additionally, [Examples/FromURDF.py](Examples/FromURDF.py) shows how to calibrate a custom robot model defined in a URDF file.

### Minimal Example
```python
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb
from robotkinecal import SerialRobotKineCal

#Joint configurations reached during data collection
configurations = [
    [0.7379080192092227, 2.5894444647892367, 1.825416964205039, 3.091839038290101, 2.8827364908669937, 1.8344647650878239, -1.3493080127049217],
    [0.5485449070443003, -0.17823804356018647, -2.468495009792379, -1.7013699309548564, 2.513055435581019, -0.5230529481527988, 0.22526263925660972],
    [1.5460619969223375, 2.079778271432109, 0.840223786664084, -0.38761044852972537, -2.1829496374793194, 0.4298302893909538, 0.1773383662282102],
    [0.36292198850587143, 2.926937253360429, -2.216978582332016, -2.9553150554237244, 0.589950213299741, -2.424896731196723, 2.832521826380389],
    [-0.9549889194638133, -1.99435381806784, 2.524559046456991, 1.2976547203598257, 1.4241371154351716, 2.5138260178240195, 1.7540378912902277],
    [0.5685139443689309, 2.223610277195741, -1.3379262425863947, -2.0541791969127745, -2.2995125812073756, 3.10800166827796, -2.013774277469671],
    [1.4352597031044043, -2.2466448085858484, 0.32967206973427254, -1.4260112561208396, 2.9813408799580197, 1.0542362234276208, -1.5352756805346326],
    [1.2459328733337909, 2.8412275549266592, 2.4502115961101003, 3.101175202445268, 2.002473212659864, 0.28351093156853135, -0.30627980563416113],
    [-0.844340175819883, 1.8950250155197264, 1.7764801173164066, 1.2651531603415629, 0.7714280450108828, -0.03969310742199372, 2.1396614739252167],
    [-2.348621231134324, -2.0642446452075056, 1.4896583736744446, -2.3434434343157413, -0.8190139905243758, 0.6555498862249252, -2.4937683582669945]
]

#End-effector poses observed/measured by an external system
observed_ee_poses = [
    SE3.Rt(R=np.array([[-0.9324257885959856,-0.12674653717162304,-0.33840429086546275],[-0.20359466473222154,-0.5894297954308296,0.7817427510063303],[-0.29854875858681545,0.7978144092289713,0.523794623085135]]),t=[-0.20020845, 0.14357632, 0.54859908]),
    SE3.Rt(R=np.array([[0.7081515893700399,0.7050182461949774,0.03834838985986567],[0.09186072064425603,-0.14584834169510727,0.9850329279914987],[0.700059236404609,-0.6940299228109299,-0.1680462191411954]]),t=[-0.16699219, -0.36075783, 0.70381521]),
    SE3.Rt(R=np.array([[0.782493008434827,0.07492066800086444,0.6181355719071004],[0.4386386029173855,0.6382606030582646,-0.6326291003537413],[-0.4419285776924793,0.766165971634087,0.46657136231095203]]),t=[0.06005071, 0.49287995, -0.09075145]),
    SE3.Rt(R=np.array([[-0.7153746269464165,-0.24361396036021254,0.6548979931553375],[0.21015632376029245,-0.968872445715429,-0.13084534197655737],[0.6663883722761337,0.04402751697536291,0.7443037787383248]]),t=[0.29116534, -0.14646528, 0.40562032]),
    SE3.Rt(R=np.array([[-0.30895027371762956,0.0008401934860206525,-0.9510778214450745],[0.7035976408011462,-0.6726362368400867,-0.22915246616318408],[-0.6399219391682579,-0.7399728285312338,0.2072199913295606]]),t=[-0.47818308, 0.23775671, 0.59060084]),
    SE3.Rt(R=np.array([[0.31270451230506163,-0.9494252614904123,0.028417614747115325],[0.13217456104615904,0.013867386292998732,-0.9911294471509037],[0.940609256490871,0.3136867361576875,0.129826261451328]]),t=[0.20792547, -0.44522547, 0.08904329]),
    SE3.Rt(R=np.array([[0.8665039002980348,-0.49887545095794694,0.017154451311483177],[-0.48461686175594715,-0.8489816749303915,-0.21065757269608124],[0.11965570638339872,0.17422227200875492,-0.9774093880590661]]),t=[-0.32655286, -0.74353132, 0.26892729]),
    SE3.Rt(R=np.array([[0.40761686352115845,0.8548560838441313,0.3210444961182939],[0.7120194395680405,-0.5176684679841368,0.47439190015447263],[0.571731414433725,0.03521978378174277,-0.8196845469935249]]),t=[-0.00058408, 0.1862683,  0.39643306]),
    SE3.Rt(R=np.array([[0.11897928853624999,-0.8172971004352293,0.5637990586366514],[0.2728040551136579,0.5728877351413868,0.7729020574679025],[-0.9546841762661369,0.06184733255960661,0.2911237383696311]]),t=[0.08149727, -0.40112761, 0.27034067]),
    SE3.Rt(R=np.array([[-0.34883719892825377,-0.881267796852924,-0.3188725119446387],[0.16685186939773253,-0.3932098394115846,0.9041827668499315],[-0.9222109641087394,0.2622081089896144,0.28420739831629716]]),t=[0.34657502, -0.13572935, 0.49357116])
]

#Load the model of the robot
robot_model = rtb.models.URDF.Panda()

#Create the calibration object
cal = SerialRobotKineCal(robot_model, ee_name='panda_link8', verbose=True)

#Set the data
cal.set_observations(configurations, observed_ee_poses)

#Solve the calibration problem
result = cal.solve()

#Inspect and export the result
print("Termination reason:", result.termination_reason)
last_iteration = result.iteration_results[-1]
print(f"Regressor rank: {last_iteration.matrix_rank}/{last_iteration.N_PARAMS}")
screw_axes, zero_conf_ee_pose = cal.get_calibration(result)
urdf_joint_definitions = cal.get_urdf_xyzrpy(result)
```
produces
```
> python Examples/MinimalExample.py 
Iteration #1 result:
        Norm of twist errors: 0.2851
        Avg. Position error: 0.0566
        Max. Position error: 0.1169
        Avg. Orientation error: 0.0632
        Max. Orientation error: 0.0884
        Joints uncertainty: [1.7910e-07 1.3972e-07 3.2196e-07 3.6824e-07 5.1620e-07 6.2999e-07 7.8914e-07]
Iteration #2 result:
        Norm of twist errors: 0.0045
        Avg. Position error: 0.0012
        Max. Position error: 0.0019
        Avg. Orientation error: 0.0006
        Max. Orientation error: 0.0009
        Joints uncertainty: [1.8149e-14 1.4213e-14 3.2184e-14 3.7330e-14 5.2304e-14 6.3555e-14 7.9726e-14]
Iteration #3 result:
        Norm of twist errors: 0.0000
        Avg. Position error: 0.0000
        Max. Position error: 0.0000
        Avg. Orientation error: 0.0000
        Max. Orientation error: 0.0000
        Joints uncertainty: [2.7885e-18 2.1832e-18 4.9448e-18 5.7353e-18 8.0385e-18 9.7659e-18 1.2251e-17]
Iteration #4 result:
        Norm of twist errors: 0.0000
        Avg. Position error: 0.0000
        Max. Position error: 0.0000
        Avg. Orientation error: 0.0000
        Max. Orientation error: 0.0000
        Joints uncertainty: [2.3988e-18 1.8781e-18 4.2538e-18 4.9338e-18 6.9152e-18 8.4011e-18 1.0539e-17]
The kinematic calibration has converged.
Termination reason: converged
Regressor rank: 34/34
```

`solve()` updates the calibrator's internal model. Call `cal.reset()` before
running another calibration from the same nominal model; loaded observations
are retained. Before deployment, verify the calibrated model on observations
that were not used by the solver.

### Real Robot Example
This library was used to find the kinematic parameters of a real Franka Research 3 robot arm using a dataset collected with a RealSense D405 camera mounted on the robot end-effector and overlooking a calibration board, as shown in the following GIF:

![FR3_Calibration_Setup](https://github.com/user-attachments/assets/793d800d-7953-4063-a9cf-7074337a6916)

The dataset was collected by moving the robot to about 115 different poses, each having the camera (approximately) pointing at the centre of the calibration board. Assuming knowledge of the transform between the end-effector frame and the camera frame (a reasonable assumption since the camera mount was accurately 3D printed), end-effector poses were obtained from the camera images. The resulting dataset is [here](https://github.com/PhilNad/robot-arm-kinematic-calibration/blob/main/Examples/calib_data.pickle) and the code used to calibrate the robot is [here](https://github.com/PhilNad/robot-arm-kinematic-calibration/blob/main/Examples/FrankaReal.py).

Using `SerialRobotKineCal.print_urdf_joint_definitions()`, joint definitions that can directly be used in the [kinematics.yaml](https://github.com/frankaemika/franka_description/blob/main/robots/fr3/kinematics.yaml) file of the [franka_description](https://github.com/frankaemika/franka_description) package were obtained:
```
Definition of the joints in RPY-XYZ format for use in a URDF:
Joint panda_link0-panda_link1
        XYZ: [4.67752926e-03 1.33386504e-04 3.32977503e-01]
        RPY: [-7.88191822e-05  9.64086699e-03 -9.98194809e-07]
Joint panda_link1-panda_link2
        XYZ: [ 9.89653185e-05 -1.31223433e-04 -1.68527322e-03]
        RPY: [-1.56837511e+00 -9.64077762e-03 -3.51593821e-06]
Joint panda_link2-panda_link3
        XYZ: [ 0.00239793 -0.31767381  0.00077651]
        RPY: [1.56950118e+00 1.40891682e-05 9.46926952e-03]
Joint panda_link3-panda_link4
        XYZ: [ 8.28532697e-02 -3.26817314e-05 -3.92459010e-03]
        RPY: [ 1.56817596 -0.00946446  0.00264282]
Joint panda_link4-panda_link5
        XYZ: [-0.07844599  0.38868122 -0.00130173]
        RPY: [-1.57377615 -0.00264545 -0.01050378]
Joint panda_link5-panda_link6
        XYZ: [ 0.00134897 -0.00154525 -0.00648275]
        RPY: [ 1.56590009 -0.01049793  0.02455011]
Joint panda_link6-panda_link7
        XYZ: [ 0.08726884  0.00647582 -0.00217406]
        RPY: [ 1.5722515  -0.02449635 -0.01476014]
Joint panda_link7-panda_link8
        XYZ: [-0.00130561  0.00223521  0.11176487]
        RPY: [ 0.01627551  0.00927677 -0.05635541]
```

After independent validation, ROS nodes such as the [MoveIt](https://github.com/moveit/moveit) motion planner can use the updated robot model. Keep the nominal model for rollback, test the candidate URDF in simulation, and verify it on the robot at low speed before production use.

## Frequently Asked Questions
### My kinematic calibration is not converging. What can I do?
- First verify the frame conventions, units, synchronization, regressor rank, and pose diversity. Add observations that excite weakly observed joints and workspace directions; simply repeating similar poses usually does not help. You can vary `N_OBSERVATIONS` in [Examples/FrankaSimulation.py](Examples/FrankaSimulation.py) to study the effect of observation count.

## Technical Details
The method described in [POE-based robot kinematic calibration using axis configuration space and the adjoint error model](https://doi.org/10.1109/TRO.2016.2593042) and used in this library is based on twists and on the product of exponentials (POE) formula for forward kinematics. Through an iterative least-squares optimization scheme, screw axes corrections minimizing the error between the observations and the forward kinematics of the robot are found.
