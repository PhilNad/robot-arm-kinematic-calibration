from spatialmath import SE3, Twist3
from spatialmath.base import *
import roboticstoolbox as rtb
import numpy as np
from numpy.linalg import norm
import with_respect_to as WRT

class SerialRobotKineCal:
    #
    # Implements the method described in:
    #   Local POE model for robot kinematic calibration (2001)
    #
    def __init__(self, 
                 robot_model:rtb.ERobot, 
                 ee_name:str=None):
        '''
        Parameters:
        -----------
        robot_model: roboticstoolbox.ERobot
            The roboticstoolbox model of the robot.
        ee_name: str
            The name of the end effector link. If None, the last link is assumed to be the end effector.
        '''
        assert(isinstance(robot_model, rtb.Robot))
        self.robot_model = robot_model

        #Each element of self.joints is a roboticstoolbox Link object
        self.joints = [l for l in self.robot_model.links if l.isjoint]

        if ee_name is None:
            #By default, assumes that the end effector is the first child of the last joint
            self.EE_link = self.joints[-1].children[0]
        else:
            #The user has specified the end effector link.

            #Verify that the end effector link exists
            ee_link = self.robot_model.link_dict.get(ee_name)
            if ee_link is None:
                raise ValueError(f"End-effector link '{ee_name}' does not exist.")
            #Verify that the end effector link is a child of the last joint
            if ee_link.parent_name != self.joints[-1].name:
                raise ValueError(f"End-effector link '{ee_name}' is not a child of the last joint.")
            
            self.EE_link = ee_link

        #Each element of joint_local_nominal_poses is the pose of the i-th link relative to the previous link
        # when the joint angles are zero.
        self.joint_local_nominal_poses = [l.A() for l in self.joints]

        #Each element of joint_local_screws is the screw axis of the joint
        self.joint_local_screws = [l.v.s for l in self.joints]

        #Add the end-effector fixed link to the lists.
        # The end-effector link is fixed relative to the previous one, so the screw axis is zero.
        self.joint_local_nominal_poses.append(self.EE_link.A())
        self.joint_local_screws.append(np.zeros(6))

        #Number of joints
        self.N_JOINTS = len(self.joint_local_screws) - 1

        #List of observed end-effector poses that will be populated by set_data
        self._obs_ee_poses = []

        #List of measured joint positions that will be populated by set_data
        self._joint_positions = []

    def set_data(self, wrt_world_name:str, joint_positions):
        '''
        Set the data for the calibration.

        Parameters:
        -----------
        wrt_world_name: str
            The name of the with_respect_to "world" in which the EE poses are written.
        joint_positions: list
            List of numpy arrays where each array has a shape of 1xN_JOINTS.
        '''
        #Set the number of observations
        self.N_OBSERVATIONS = len(joint_positions)

        #Build a list of SE3 objects representing the observed end-effector poses
        self._obs_ee_poses = []
        WRT_DB = WRT.DbConnector()
        for m in range(self.N_OBSERVATIONS):
            joint_pos = joint_positions[m]
            if len(joint_pos) != self.N_JOINTS:
                raise ValueError(f"The number of columns in joint_positions must be equal to the number of joints ({self.N_JOINTS}).")
            
            #Observed end-effector poses
            obs_name = f'ee-{m}'
            obs_ee_pose = SE3(WRT_DB.In(wrt_world_name).Get(obs_name).Wrt('world').Ei('world'))
            self._obs_ee_poses.append(obs_ee_pose)

        #Measured joint positions
        self._joint_positions = joint_positions

    def forward_kinematics(self, link_index:int, joint_positions:np.ndarray):
        '''
        Compute the forward kinematics of the i-th link.

        Parameters:
        -----------
        link_index: int
            The index of the link for which to compute the forward kinematics.
        joint_positions: numpy.ndarray
            The joint angles as a 1xN_JOINTS numpy array.
        
        Returns:
        --------
        The pose of the link_index-th link relative to the base frame as a spatialmath.SE3 object.
        '''

        #Verify that i is within the range of the number of joints
        if link_index < 0 or link_index > self.N_JOINTS+1:
            raise ValueError(f"Link index 'i' must be between 0 and {self.N_JOINTS+1}.")

        T = SE3()
        for j in range(min(link_index, self.N_JOINTS)):
            T *= self.joint_local_nominal_poses[j] @ Twist3(self.joint_local_screws[j]).exp(joint_positions[j])

        #If the user desire the pose of the end effector
        if link_index == self.N_JOINTS + 1:
            T *= self.joint_local_nominal_poses[-1]
        
        return T
    
    def update_twist_definitions(self, twist_corrections:np.ndarray, step_size:float=1.0):
        '''
        Update the twist definitions of the joints such as to account for the error in the EE pose.

        Parameters:
        -----------
        twist_corrections: numpy.ndarray
            The twist corrections to apply to each joint that, together, account for the error in the EE pose.
            The shape of the array is 6(N_JOINTS+1)x1.
        step_size: float
            The step size to apply to the twist corrections during the optimization (default = 1.0).
        '''
        #Verify that the shape of the array is correct
        if twist_corrections.shape != (6*(self.N_JOINTS+1), 1):
            raise ValueError(f"The shape of the twist corrections array must be (6*(N_JOINTS+1), 1).")
        
        #Verify that the step size is positive
        if step_size <= 0:
            raise ValueError("The step size must be positive.")

        #Update the pose of the joints
        for i in range(self.N_JOINTS+1):
            #The x[6*i:6*(i+1)] contains the twist that describes the perturbation to the pose
            # of the i-th link relative to the previous link and expressed in the local frame of the i-th link
            # such as to account for the error.
            delta_twist = step_size * twist_corrections[6*i:6*(i+1)].reshape(6)
            #Update the pose of the i-th link relative to the previous link
            # Equation (19) in the paper.
            self.joint_local_nominal_poses[i] = self.joint_local_nominal_poses[i] @ Twist3(delta_twist).SE3()

    def A_Matrix(self, joint_positions:np.ndarray):
        '''
        Build a matrix of elements that will be multiplied by the joint pose perturbations such as to account for the TCP error. See equations (16) and (25) in the paper.

        Parameters:
        -----------
        joint_positions: numpy.ndarray
            The joint angles as a 1xN_JOINTS numpy array.

        Returns:
        --------
        The A matrix for the current observation as a numpy.ndarray of shape 6x6(N_JOINTS+1).
        '''
        A = np.ndarray((6, 6*(self.N_JOINTS+1)))

        prev_T = self.joint_local_nominal_poses[0]
        A[:,0:6] = prev_T.Ad()
        for i in range(self.N_JOINTS):
            #Compute the pose of the (i+1)-th link relative to the base frame given the angles of the previous joints.
            #   prev_T: Pose of the i-th link relative to the base frame
            #   Twist3(self.joint_local_screws[i]).exp(joint_positions[i]) : Impact of the the i-th joint angle
            #   self.joint_local_nominal_poses[i+1]: Pose of the (i+1) link relative to the i-th link when the joint angles are zero
            T = prev_T @ Twist3(self.joint_local_screws[i]).exp(joint_positions[i]) @ self.joint_local_nominal_poses[i+1]
            #The reason why the adjoint is used here is quite involved.
            # For an explanation, see the paper titled "POE-Based Robot Kinematic Calibration Using Axis Configuration Space and the Adjoint Error Model (2016)".
            A[:,6*(i+1):6*(i+2)] = T.Ad()
            prev_T = T

        return A
    
    def solve(self, max_iterations:int=10, step_size:float=1.0):
        '''
        Solve the kinematic calibration problem via a iterative least squares method where at each iteration the screw axes are updated to minimize the error in the end effector pose. The process is stopped when the error has converged, when the error is increasing, or when the maximum number of iterations has been reached.

        Parameters:
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        
        step_size: float
            The step size to apply to the twist corrections during the optimization (default = 1.0).
        '''
        #Iterative least squares
        previous_error_mag = np.inf
        for it in range(max_iterations):
            y_all = np.zeros((6*self.N_OBSERVATIONS, 1))
            A_all = np.ndarray((6*self.N_OBSERVATIONS, 6*(self.N_JOINTS+1)))
            
            #Lists to store the position and orientation errors
            position_errors = []
            orientation_errors = []

            #Iterate over all observations
            for m in range(self.N_OBSERVATIONS):
                #End effector pose according to the observation
                obs_ee_pose = self._obs_ee_poses[m]
                
                #Joint positions as recorded by the robot
                joint_pos = self._joint_positions[m]

                #End effector pose according to the model
                mod_ee_pose = self.forward_kinematics(self.N_JOINTS+1, joint_pos)

                #Pose difference between observed and predicted end effector poses
                pose_diff = obs_ee_pose @ mod_ee_pose.inv()
                
                position_error = np.linalg.norm(pose_diff.t)
                orientation_error = pose_diff.angvec()[0]
                
                position_errors.append(position_error)
                orientation_errors.append(orientation_error)

                #Twist error vector
                y = pose_diff.log(twist=True) 

                #Jacobian matrix
                A = self.A_Matrix(joint_pos)

                #Populate the y_all and A_all matrices
                y_all[6*m:6*(m+1)] = y.reshape((6,1))
                A_all[6*m:6*(m+1), :] = A

            #Solve for the perturbation to the joint zero pose such that the EE error is accounted for
            x = np.linalg.lstsq(A_all, y_all, rcond=1e-12)[0]

            #Update the kinematic model taking into account the errors
            self.update_twist_definitions(x, step_size)

            y_norm = norm(y_all)
            print(f"Iteration {it}: {y_norm}")
            print(f"\tAvg. Position error: {np.mean(position_errors)}")
            print(f"\tAvg. Orientation error: {np.mean(orientation_errors)}")
            print(f"\tMax. Position error: {np.max(position_errors)}")
            print(f"\tMax. Orientation error: {np.max(orientation_errors)}")
            if y_norm > previous_error_mag:
                print('Stopping because the error is increasing.')
                break
            if np.abs(y_norm - previous_error_mag) < 1e-4:
                print('Stopping because the error has converged.')
                break
            previous_error_mag = y_norm

        #Compute screw definitions for use with the PoE formula
        print("Definition of the screw axes for use with the PoE formula:")
        T = SE3()
        screw_definitions = []
        for i in range(self.N_JOINTS):
            print(f"Link {self.joints[i].name}")
            #Since the self.joint_local_nominal_poses are describing the pose of the link relative to the previous link
            # we need to multiply them to get the pose of the link relative to the base frame.
            T *= self.joint_local_nominal_poses[i]

            #Rotation axis in the base frame
            rot_axis = T.R @ self.joint_local_screws[i][3:6]

            screw_v = np.cross(T.t, rot_axis)
            screw_w = rot_axis
            screw = np.hstack((screw_v, screw_w))
            print("\tScrew definition [v,w]: ", screw)

            screw_definitions.append(screw)

        #End effector pose wrt the base frame when all joint angles are zero
        print("End effector pose in zero configuration:")
        ee_zero_pose = T @ self.joint_local_nominal_poses[-1]
        screw_definitions.append(ee_zero_pose)
        print(ee_zero_pose.A)

        print('Definition of the joints in RPY-XYZ format for use in a URDF:')
        for i in range(self.N_JOINTS+1):
            if i < self.N_JOINTS:
                print(f"Joint {self.joints[i].parent.name}-{self.joints[i].name}")
            else:
                print(f"Joint {self.EE_link.parent.name}-{self.EE_link.name}")
                
            X = SE3(self.joint_local_nominal_poses[i])
            rpy = X.rpy(unit='rad',order='zyx')
            xyz = X.t
            print(f"\tXYZ: {xyz}")
            print(f"\tRPY: {rpy}")


if __name__ == '__main__':
    db = WRT.DbConnector()

    #Approximate pose of the camera relative to the link8 frame
    REALSENSE_X_OFFSET_MM        = 94.5     #millimeters
    REALSENSE_Y_OFFSET_MM        = -9       #millimeters
    REALSENSE_Z_OFFSET_MM        = 40.55    #millimeters
    REALSENSE_ROT_Z_OFFSET_DEG   = 90       #degrees

    #Pose of Realsense relative to TCP expressed in TCP frame
    X_C_T = SE3.Rz(REALSENSE_ROT_Z_OFFSET_DEG, t=[REALSENSE_X_OFFSET_MM/1000, REALSENSE_Y_OFFSET_MM/1000, REALSENSE_Z_OFFSET_MM/1000], unit="deg")


    db.In('kine-cal').Set('ee').Wrt('cam').Ei('cam').As(X_C_T.inv().A)

    #Pose of the board relative to robot's base frame (world)
    # Assuming a board thickness of 3 mm, and a robot mounting plate thickness of 12.7 mm
    # The origin of the board is close to the 0 marker.
    board_centre = ((82.25+4*25+3+250)/1000, -(4.5*25+3+250)/1000)
    board_dims = (0.5, 0.5)
    #Top left corner of the checkerboard
    # We need to add half the size of the board (including margin) and remove the margin.
    top_left_corner = (board_centre[0] + board_dims[0]/2 - 0.025, board_centre[1] - board_dims[1]/2 + 0.025)
    X_B_W = SE3.Ry(0,t=[top_left_corner[0], top_left_corner[1], -9.7/1000], unit="deg")
    db.In('kine-cal').Set('board').Wrt('world').Ei('world').As(X_B_W.A)

    import pickle
    fin = open('/home/phil/UofT/Research Coding/calib_data.pickle', 'rb')
    kine_cal = pickle.load(fin)

    joint_positions = kine_cal['joint_positions']
    camera_poses = kine_cal['camera_poses']

    for m in range(len(camera_poses)):
        #Camera pose relative to the board
        X_C_B = camera_poses[m]
        db.In('kine-cal').Set('cam').Wrt('board').Ei('board').As(X_C_B)
        #End-effector pose from cameral observation
        obs_ee_pose = db.In('kine-cal').Get('ee').Wrt('world').Ei('world')
        #Record the observed end-effector pose as a separate frame
        db.In('kine-cal').Set(f'ee-{m}').Wrt('world').Ei('world').As(obs_ee_pose)

    #Load the model of the robot
    panda = rtb.models.URDF.Panda()
    #Create the calibration object
    cal = SerialRobotKineCal(panda, 'panda_link8')
    #Set the data
    cal.set_data('kine-cal', joint_positions)
    #Solve the calibration problem
    cal.solve()