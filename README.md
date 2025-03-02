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
from RobotKineCal import SerialRobotKineCal
#Load custom robot model
nominal_robot_model = rtb.ERobot.URDF(Path(__file__).parents[1] / 'Examples' / 'gen3.urdf')
#Calibrate (assuming configurations and observed_ee_poses are defined)
cal = SerialRobotKineCal(nominal_robot_model, ee_name, verbose=True)
cal.set_observations(configurations, observed_ee_poses)
result = cal.solve()
```
See [Examples/FromURDF.py](Examples/FromURDF.py) for a complete example.

Since the method used to produce the calibration data varies greatly between different robot setups, this library does not provide any tools to collect the calibration data. Instead, it expects the user to provide input data in the form of a list of joint configurations and measured end-effector poses/positions.

For instance, observations can be obtained from a motion capture system with markers attached to the robot end-effector, from a laser tracker, from a camera attached to the robot end-effector and observing a known pattern, etc.

## Usage Examples
See [Examples/FrankaSimulation.py](Examples/FrankaSimulation.py) for an example on how to use this library to calibrate a Franka robot model using a simulated dataset, and [Examples/FrankaReal.py](Examples/FrankaReal.py) for an example on how to use this library to calibrate a real Franka robot using a dataset collected with a camera mounted on the end-effector of the robot (eye-in-hand). Additionally, [Examples/FromURDF.py](Examples/FromURDF.py) shows how to calibrate a custom robot model defined in a URDF file.

### Minimal Example
```python
import numpy as np
from spatialmath import SE3
import roboticstoolbox as rtb

import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[1].as_posix())
from RobotKineCal import SerialRobotKineCal

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
```

### Real Robot Example
This library was used to find the kinematic parameters of a real Franka Research 3 robot arm using a dataset collected with a RealSense D405 camera mounted on the robot end-effector. A calibration board was precisely mounted on the robot table, as shown in the following picture:
![FR3_Calibration_Setup](https://github.com/user-attachments/assets/ac481fa2-d099-4bc2-a510-666ff133a48e)

The dataset was collected by moving the robot to about 115 different poses, each having the camera (approximately) pointing at the centre of the calibration board. Assuming knowledge of the transform between the end-effector frame and the camera frame (a reasonable assumption since the camera mount was accurately 3D printed), end-effector poses were obtained from the camera images. The resulting dataset is [here](https://github.com/PhilNad/robot-arm-kinematic-calibration/blob/main/Examples/calib_data.pickle) and the code used to calibrate the robot is [here](https://github.com/PhilNad/robot-arm-kinematic-calibration/blob/main/Examples/FrankaReal.py).

Using `SerialRobotKineCal.print_urdf_joint_definitions()`, joint definitions that can directly be used in the [kinematics.yaml](https://github.com/frankaemika/franka_description/blob/main/robots/fr3/kinematics.yaml) file of the [franka_description](https://github.com/frankaemika/franka_description) package were obtained:
```
Definition of the joints in RPY-XYZ format for use in a URDF:
Joint panda_link0-panda_link1
        XYZ: [ 1.46717027e-03  1.07142921e-04 -7.02042906e-06]
        RPY: [-7.88191822e-05  9.64086699e-03 -9.98194809e-07]
Joint panda_link1-panda_link2
        XYZ: [ 9.89653185e-05 -1.31223433e-04  3.31314727e-01]
        RPY: [ 2.42121831e-03 -9.64077762e-03 -3.51593821e-06]
Joint panda_link2-panda_link3
        XYZ: [-0.00059432  0.00036725  0.00168825]
        RPY: [-1.57209161e+00  9.46926952e-03 -1.40897999e-05]
Joint panda_link3-panda_link4
        XYZ: [ 3.57252808e-04 -3.11294604e-01 -2.50704407e-04]
        RPY: [ 1.56820097 -0.0026427  -0.00946449]
Joint panda_link4-panda_link5
        XYZ: [ 8.25127849e-02 -6.07637054e-05  3.83759295e-03]
        RPY: [ 1.56778872  0.01050374 -0.0026456 ]
Joint panda_link5-panda_link6
        XYZ: [-0.08115103  0.37751725  0.00154525]
        RPY: [-1.57543479  0.02454876  0.01050109]
Joint panda_link6-panda_link7
        XYZ: [-0.00069518  0.00432953  0.00777427]
        RPY: [ 1.57188994  0.01475571 -0.02449902]
Joint panda_link7-panda_link8
        XYZ: [ 0.07601011 -0.00010436  0.00450503]
        RPY: [ 1.57889554e+00 -5.65783890e-05  1.46424774e-02]
```

After replacing the nominal kinematic parameters in the URDF file with the calibrated ones, any ROS node should be able to benefit from the improved accuracy of the robot model. This includes the [MoveIt](https://github.com/moveit/moveit) motion planner, whose collision avoidance capabilities depend on accurate kinematic parameters. In our experiments, the robot was a lot less likely to collide with the environment after calibration.

## Frequently Asked Questions
### My kinematic calibration is not converging. What can I do?
- Assuming that the robot model you are using is correct, gather more observations. The more data you have, the more likely it is that the calibration will converge. You can play with `N_OBSERVATIONS` in the [Examples/FrankaSimulation.py](Examples/FrankaSimulation.py) file to see how the number of observations affects the calibration. 

## Technical Details
The method described in [POE-based robot kinematic calibration using axis configuration space and the adjoint error model](https://doi.org/10.1109/TRO.2016.2593042) and used in this library is based on twists and on the product of exponentials (POE) formula for forward kinematics. Through an iterative least-squares optimization scheme, screw axes corrections minimizing the error between the observations and the forward kinematics of the robot are found.