from spatialmath import SE3, SO3
from spatialmath.base import skew
import roboticstoolbox as rtb
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[1].as_posix())
from RobotKineCal import SerialRobotKineCal
                        
#Whether we want to use position observations instead of full poses
USE_POSITION_ONLY = False

#Perturbation magnitude, must be in [0, 1]
PERTURBATION_MAGNITUDE = 0.01

#Number of observations to generate
N_OBSERVATIONS = 10

#Standard deviation of the noise to add to the observations
# Set to zero to generate noise-free data.
OBS_NOISE_STD_DEV = 0#0.01

#Fixed seed for reproducibility
np.random.seed(1234)

#Load the model of the robot
nominal_robot_model = rtb.models.URDF.Panda()
ee_name = 'panda_link8'

#According to Theorem 2 of [1], when point measurements are taken,
# the measured point should not lie on the last joint axis if the last joint is revolute.
# Otherwise, some parameters will be unidentifiable. To avoid this, we offset
# the end-effector position slightly.
# [1] : "POE-Based Robot Kinematic Calibration Using Axis Configuration Space and the Adjoint Error Model", Li et al., 2016
if USE_POSITION_ONLY:
    #Create a new model with the same structure but with an end-effector offset
    # to ensure that the end-effector position is not on the last joint axis.
    links = []
    parent_link = None
    for i in range(len(nominal_robot_model.links)-1):
        ets = nominal_robot_model.links[i].ets
        link_name = nominal_robot_model.links[i].name
        new_link = rtb.Link(ets)
        links.append(new_link)

    ee_pose = SE3(nominal_robot_model.link_dict.get(ee_name).Ts)
    offset = SE3.Rt(R=np.eye(3), t=np.array([0.05, 0.05, 0]))
    ee_link = rtb.Link(rtb.ET.SE3(T= offset @ ee_pose), name=ee_name)
    links.append(ee_link)
    nominal_robot_model = rtb.Robot(links)

#Slightly perturb the screw axes of the model to build a new model that
# can be used to produce simulated perturbed data
nominal_screw_axes = []
perturbed_screw_axes = []
pertubed_model_links = []
for l in nominal_robot_model.links:
    #Pose of the joint when the robot is in the zero configuration
    zero_conf_joint_pose = nominal_robot_model.fkine(np.zeros(nominal_robot_model.n), l.name)
    if l.name == ee_name:
        #Record unperturbed/nominal EE pose
        nominal_zero_conf_ee_pose = zero_conf_joint_pose
        #Perturb the pose and record it
        # NOTE: If we are using position-only observations, the end-effector orientation
        # relative to its parent link is not identifiable. Therefore, we only perturb the position.
        perturb_rot = SO3.RPY(np.random.normal(scale=PERTURBATION_MAGNITUDE, size=3))
        if USE_POSITION_ONLY:
            perturb_rot = SO3()
        perturbed_zero_conf_ee_pose = zero_conf_joint_pose * SE3.Rt(R=perturb_rot, t=np.random.normal(scale=PERTURBATION_MAGNITUDE, size=3))
        break
    if l.isjoint:
        #Record the unperturbed/nominal screw axis
        nominal_screw_axis = np.block([skew(zero_conf_joint_pose.t) @ zero_conf_joint_pose.R[:, 2], zero_conf_joint_pose.R[:, 2]])
        nominal_screw_axes.append(nominal_screw_axis)

        #Perturb the screw axis and record it
        perturbed_axis = zero_conf_joint_pose.R[:, 2] + np.random.normal(scale=PERTURBATION_MAGNITUDE, size=3)
        perturbed_axis /= np.linalg.norm(perturbed_axis)
        perturbed_point = zero_conf_joint_pose.t + np.random.normal(scale=PERTURBATION_MAGNITUDE, size=3)
        perturbed_screw_axis = np.block([skew(perturbed_point) @ perturbed_axis, perturbed_axis])
        perturbed_screw_axes.append(perturbed_screw_axis)
        
        #Create a link with the perturbed screw axis
        # to integrate it into the new model.
        link = rtb.PoELink(perturbed_screw_axis)
        pertubed_model_links.append(link)

#Create the perturbed kinematic model of the robot
# that will be used to generate observations.
actual_robot_model = rtb.PoERobot(pertubed_model_links, perturbed_zero_conf_ee_pose)

#Produce simulated data
joint_positions = []
observations = []
for m in range(N_OBSERVATIONS):
    #Pick a random joint configuration
    # with each joint position in [-PI, PI]
    config = np.random.rand(actual_robot_model.nlinks)*2*np.pi - np.pi

    #Simulate noise in the joint positions
    config_noise = np.random.normal(scale=OBS_NOISE_STD_DEV, size=actual_robot_model.nlinks)

    #Compute the EE pose according to the perturbed model
    obs_ee_pose = actual_robot_model.fkine(config + config_noise)

    #Record the EE observation and joint positions
    if USE_POSITION_ONLY:
        observations.append(obs_ee_pose.t)
    else:
        observations.append(obs_ee_pose)
    joint_positions.append(config)

#Create the calibration object
# The calibration starts from the nominal model and iteratively improves it
# to converge to the actual model.
cal = SerialRobotKineCal(nominal_robot_model, ee_name, verbose=True)
#Set the data
cal.set_observations(joint_positions, observations)
#Solve the calibration problem
result = cal.solve()

#Compare the result with the true screw axes
np.set_printoptions(precision=4)
estimated_screw_axes, estimated_zero_conf_EE_pose = cal.get_calibration(result)
estimated_model = estimated_screw_axes + [estimated_zero_conf_EE_pose]
estimated_robot_model = rtb.PoERobot([rtb.PoELink(a) for a in estimated_screw_axes], estimated_zero_conf_EE_pose)

for i in range(len(estimated_screw_axes)):
    print(f'Error for screw axis {i}:')
    initial_error = np.linalg.norm(nominal_screw_axes[i] - perturbed_screw_axes[i])
    final_error = np.linalg.norm(estimated_screw_axes[i] - perturbed_screw_axes[i])
    print(f'\tInitial error: {initial_error:.4f}')
    print(f'\tFinal error: {final_error:.4f}')
    print(f'\tRelative Improvement: {100*(initial_error - final_error)/initial_error:.2f} %')
    print(f'Nominal screw axis:{nominal_screw_axes[i]}')
    print(f'Actual screw axis:{perturbed_screw_axes[i]}')
    print(f'Estimated screw axis:{estimated_screw_axes[i]}')

print(f'Error for end-effector:')
initial_error = np.linalg.norm((nominal_zero_conf_ee_pose.inv() @ perturbed_zero_conf_ee_pose).log(twist=True))
final_error = np.linalg.norm((estimated_zero_conf_EE_pose.inv() @ perturbed_zero_conf_ee_pose).log(twist=True))
print(f'\tInitial error: {initial_error:.4f}')
print(f'\tFinal error: {final_error:.4f}')
print(f'\tRelative Improvement: {100*(initial_error - final_error)/initial_error:.2f} %')

#Compute errors on new data to validate the calibration
position_errors = []
orientation_errors = []
for m in range(N_OBSERVATIONS):
    #Pick a random joint configuration
    # with each joint position in [-PI, PI]
    config = np.random.rand(nominal_robot_model.nlinks)*2*np.pi - np.pi
    #Compute the EE pose according to the unperturbed model
    # and compare it to the model resulting from the estimation
    nom_ee_pose = nominal_robot_model.fkine(config)     #Non-calibrated model
    true_ee_pose = actual_robot_model.fkine(config)     #True model 
    cal_ee_pose = estimated_robot_model.fkine(config)   #Calibrated model         

    #Errors of the model before calibration
    init_pose_diff = nom_ee_pose @ true_ee_pose.inv()
    init_position_error = np.linalg.norm(init_pose_diff.t)
    init_orientation_error = init_pose_diff.angvec()[0]

    #Errors of the model after calibration
    cal_pose_diff = cal_ee_pose @ true_ee_pose.inv()
    cal_position_error = np.linalg.norm(cal_pose_diff.t)
    cal_orientation_error = cal_pose_diff.angvec()[0]

    #Compute improvements
    position_error_improvement = (init_position_error - cal_position_error)/init_position_error
    orientation_error_improvement = (init_orientation_error - cal_orientation_error)/init_orientation_error

    position_errors.append(position_error_improvement)
    orientation_errors.append(orientation_error_improvement)

print(f'Average relative EE position error improvement: {100*np.mean(position_errors):.2f} %')
print(f'Average relative EE orientation error improvement: {100*np.mean(orientation_errors):.2f} %')