from spatialmath import SE3, Twist3, UnitQuaternion
import roboticstoolbox as rtb
import numpy as np
import with_respect_to as WRT
from RobotKineCal.RobotKineCal import SerialRobotKineCal

def fk_poe(joint_positions, screw_definitions):
    '''
    Forward Kinematics using the Product of Exponentials formula
    to be used to generate simulated data from perturbed screw axes.

    Parameters
    ----------
    joint_positions : list
        List of joint positions.
    screw_definitions : list
        List of screw axes (Twist3 objects) for each joint and the end-effector and the last element being the pose of the end-effector when the robot is in the zero configuration.
    '''
    n = len(screw_definitions) - 1
    T = SE3()
    for i in range(n):
        T *= Twist3(screw_definitions[i]).exp(joint_positions[i])
    T *= screw_definitions[-1]
    return T

#Load the model of the robot
robot_model = rtb.models.URDF.Panda()
#Create the calibration object
ee_name = 'panda_link8'
cal = SerialRobotKineCal(robot_model, ee_name, verbose=True)

#Slightly perturb the screw axes of the model to build a new model that
# can be used to produce simulated perturbed data
screw_definitions = []
unperturbed_screw_definitions = []
for l in robot_model.links:
    if l.isjoint:
        unperturbed_pose = Twist3(l.v.s).SE3()
        unperturbed_screw_definitions.append(l.v.s)

        #Perturb the screw axis
        q = np.random.normal(size=4)
        q /= np.linalg.norm(q)
        perturbation = SE3.Rt(R=UnitQuaternion(q).SO3(), t=np.random.rand(3)*2-1)
        largely_perturbed_pose = unperturbed_pose * perturbation
        slightly_perturbed_pose = unperturbed_pose.interp(largely_perturbed_pose, 0.01)
        perturbed_screw_axis = Twist3(slightly_perturbed_pose)
        screw_definitions.append(perturbed_screw_axis)
    if l.name == ee_name:
        #Pose of the EE relative to the last joint
        unperturbed_pose = l.A()
        unperturbed_screw_definitions.append(unperturbed_pose)

        #Perturb the pose
        largely_perturbed_pose = unperturbed_pose * perturbation
        slightly_perturbed_pose = unperturbed_pose.interp(largely_perturbed_pose, 0.01)
        #Compute the pose of the EE relative to the base in the zero configuration
        # using the product of exponentials formula
        ee_zero_pose = SE3()
        for s in screw_definitions:
            ee_zero_pose *= Twist3(s).exp(0)
        ee_zero_pose *= slightly_perturbed_pose
        screw_definitions.append(ee_zero_pose)

#Connect to the frame database that will be used to record
# end-effector poses.
TEMPORARY_DATABASE = 1
db = WRT.DbConnector(TEMPORARY_DATABASE)

#Produce simulated data
N_OBSERVATIONS = 100
joint_positions = []
for m in range(N_OBSERVATIONS):
    #Pick a random joint configuration
    config = np.random.rand(7)
    #Compute the EE pose according to the perturbed model
    obs_ee_pose = fk_poe(config, screw_definitions)
    #Record the EE pose and joint positions
    db.In('kine-cal').Set(f'ee-{m}').Wrt('world').Ei('world').As(obs_ee_pose)
    joint_positions.append(config)

#Set the data
cal.set_data('kine-cal', joint_positions)
#Solve the calibration problem
result = cal.solve()

#Compare the result with the true screw axes
estimated_screw_axes = cal.get_screw_axes(result)
for i in range(len(estimated_screw_axes)):
    print(f'Error for screw axis {i}:')
    if i < cal.N_JOINTS:
        initial_error = np.linalg.norm(screw_definitions[i] - unperturbed_screw_definitions[i])
        final_error = np.linalg.norm(screw_definitions[i] - estimated_screw_axes[i])
    else:
        initial_error = np.linalg.norm((screw_definitions[i].inv() @ unperturbed_screw_definitions[i]).log(twist=True))
        final_error = np.linalg.norm((screw_definitions[i].inv() @ estimated_screw_axes[i]).log(twist=True))

    print(f'\tInitial error: {initial_error}')
    print(f'\tFinal error: {final_error}')
    print(f'\tRelative Improvement: {100*(initial_error - final_error)/initial_error} %')