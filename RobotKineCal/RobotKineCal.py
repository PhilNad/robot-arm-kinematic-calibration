from spatialmath import SE3, Twist3
import roboticstoolbox as rtb
import numpy as np
from numpy.linalg import norm
import with_respect_to as WRT

class CalibrationResult:
    '''
    Class to store the result of a kinematic calibration.
    '''
    def __init__(self, joints, convergence_tolerance=1e-4) -> None:
        '''
        Initialize the calibration result.

        Parameters
        -----------
        joints: list
            List of roboticstoolbox.Link objects representing the joints of the robot.
        convergence_tolerance: float
            The tolerance to consider that the error has converged.
        '''
        self.joints = joints
        self.N_JOINTS = len(joints)
        self.convergence_tolerance = convergence_tolerance
        self.has_converged = False
        self.is_diverging = False
        self.nb_iterations_executed = 0
        self.iteration_results = []

    class IterationResult:
        '''
        Class to store the result of an iteration of the calibration.
        '''
        def __init__(self, joint_local_nominal_poses, joint_local_screw_definitions,
                     A_all=None,
                     twist_corrections=None, 
                     twist_errors=None, 
                     position_errors=None, 
                     orientation_errors=None) -> None:
            '''
            Initialize the iteration result.

            Parameters
            ----------
            joint_local_nominal_poses: list
                List of SE3 objects representing the pose of the i-th link relative to the previous link when the joint angles are zero.
            joint_local_screw_definitions: list
                List of screw axes for each joint and the end-effector.
            A_all: numpy.ndarray
                The data matrix matrix build from stacked A matrices when considered all observations as a numpy.ndarray of shape 6*N_OBSERVATIONS x 6*(N_JOINTS+1).
            twist_corrections: numpy.ndarray
                A 6*(N_JOINTS+1) x 1 vector containing the twist correction for each joint and for the end-effector. This is x in SerialRobotKineCal.solve.
            twist_errors: numpy.ndarray
                Vector of length 6(N_JOINTS+1) containing the twist errors for each joint and the end-effector. This is y_all in SerialRobotKineCal.solve.
            position_errors: numpy.ndarray
                Vector of length N_OBSERVATIONS containing the magnitude of the position difference between the observed and computed end-effector poses for each observation.
            orientation_errors: numpy.ndarray
                Vector of length N_OBSERVATIONS containing the magnitude of the orientation difference between the observed and computed end-effector poses for each observation.
            '''

            #The i-th element of joint_local_nominal_poses is the pose of the i-th link 
            # relative to the previous link when the joint angles are zero.
            self.joint_local_nominal_poses = joint_local_nominal_poses
            
            #The i-th element of joint_local_screws is the screw axis of the i-th joint relative
            # to the previous one. Since the end-effector link is fixed relative to the previous one, the last screw axis is zero.
            self.joint_local_screw_definitions = joint_local_screw_definitions

            self.A_all = A_all
            self.twist_corrections = twist_corrections
            self.twist_errors = twist_errors
            self.position_errors = position_errors
            self.orientation_errors = orientation_errors

            #Number of joints
            self.N_JOINTS = len(joint_local_screw_definitions) - 1
            #Number of observations
            self.N_OBSERVATIONS = len(position_errors)
        
        def compute_uncertainty_estimate(self, A_all, x_all, y_all):
            '''
            Compute the uncertainty estimate of the calibration.

            Parameters
            -----------
            A_all: numpy.ndarray
                The A matrix for all observations as a numpy.ndarray of shape 6*N_OBSERVATIONS x 6*(N_JOINTS+1).
            x_all: numpy.ndarray
                The twist corrections for all observations as a numpy.ndarray of shape 6*(N_JOINTS+1) x 1.
            y_all: numpy.ndarray
                The twist errors for all observations as a numpy.ndarray of shape 6*N_OBSERVATIONS x 1.
            
            Returns
            --------
            A numpy.ndarray of shape N_JOINTS+1 containing the norm of the twist variance for each joint and the end-effector (a joint uncertainty estimate).
            '''
            #Sum of squared residuals
            SSR = (y_all - A_all @ x_all).T @ (y_all - A_all @ x_all)
            #Statistical degrees of freedom
            df = 6*self.N_OBSERVATIONS - 6*(self.N_JOINTS+1)
            #Reduced chi-squared statistic
            rcss = SSR/df
            #Variance estimates
            sigma = rcss * np.diag(np.linalg.inv(A_all.T @ A_all))
            twist_variance_norms = np.linalg.norm(sigma.reshape((6,self.N_JOINTS+1)),axis=0)
            return twist_variance_norms

        def get_statistics(self):
            '''
            Compute statistics on the results of the iteration.

            Returns
            --------
            A dictionary containing the following statistics:
            - twist_errors_norm: The norm of the twist errors.
            - position_errors_mean: The mean of the position errors.
            - position_errors_max: The maximum of the position errors.
            - orientation_errors_mean: The mean of the orientation errors.
            - orientation_errors_max: The maximum of the orientation errors.
            - joints_uncertainty: The uncertainty estimate of the calibration.
            '''
            stats = {}
            stats['twist_errors_norm'] = norm(self.twist_errors)
            stats['position_errors_mean'] = np.mean(self.position_errors)
            stats['position_errors_max'] = np.max(self.position_errors)
            stats['orientation_errors_mean'] = np.mean(self.orientation_errors)
            stats['orientation_errors_max'] = np.max(self.orientation_errors)
            stats['joints_uncertainty'] = self.compute_uncertainty_estimate(self.A_all, self.twist_corrections, self.twist_errors)
            return stats

        def print(self):
            stats = self.get_statistics()

            print("Iteration result:")
            print(f"\tNorm of twist errors: {stats['twist_errors_norm']:.4f}")
            print(f"\tAvg. Position error: {stats['position_errors_mean']:.4f}")
            print(f"\tMax. Position error: {stats['position_errors_max']:.4f}")
            print(f"\tAvg. Orientation error: {stats['orientation_errors_mean']:.4f}")
            print(f"\tMax. Orientation error: {stats['orientation_errors_max']:.4f}")
            with np.printoptions(precision=2):
                print(f"\tJoints uncertainty: {stats['joints_uncertainty']}")

    def add_iteration_result(self, iteration_result:IterationResult):
        '''
        Add the results of an iteration to the list of iteration results.

        Parameters
        -----------
        iteration_result: IterationResult
            The result of an iteration of the calibration.
        '''
        self.iteration_results.append(iteration_result)
        self.nb_iterations_executed += 1

        #Check if the optimization process has converged or is diverging
        if self.nb_iterations_executed > 1:
            previous_result = self.iteration_results[-2]
            current_result = self.iteration_results[-1]

            if abs(norm(current_result.twist_errors) - norm(previous_result.twist_errors)) < self.convergence_tolerance:
                self.has_converged = True
            elif norm(current_result.twist_errors) > norm(previous_result.twist_errors):
                self.is_diverging = True

    def get_screw_axes(self):
        '''
        Compute screw definitions for use with the PoE formula.

        Returns
        --------
        A list in which the first N_JOINTS elements are the screw axes of the joints and the last element is the pose of the end effector when all joint angles are zero.
        '''
        T = SE3()
        screw_definitions = []
        result_from_last_iteration = self.iteration_results[-1]
        for i in range(self.N_JOINTS):
            #Since the joint_local_nominal_poses are describing the pose of the link relative to the previous link
            # we need to multiply them to get the pose of the link relative to the base frame.
            T *= result_from_last_iteration.joint_local_nominal_poses[i]

            #Rotation axis in the base frame
            rot_axis = T.R @ result_from_last_iteration.joint_local_screw_definitions[i][3:6]

            #Define the screw axis
            screw_v = np.cross(T.t, rot_axis)
            screw_w = rot_axis
            screw = np.hstack((screw_v, screw_w))

            screw_definitions.append(screw)

        #End effector pose wrt the base frame when all joint angles are zero
        ee_zero_pose = T @ result_from_last_iteration.joint_local_nominal_poses[-1]
        screw_definitions.append(ee_zero_pose)

        return screw_definitions
    
    def get_urdf_xyzrpy(self):
        '''
        Compute the RPY-XYZ format of the joint definitions for use in a URDF file.

        In a URDF file, each rotary joint is defined by its axis of rotation and the position of the joint relative to the previous link. 

        Returns
        --------
        A list of dictionaries where each dictionary contains the following keys:
        - name: The name of the joint (e.g. "link1-link2").
        - xyz: The position of the joint relative to the previous link.
        - rpy: The orientation of the joint relative to the previous link.
        '''
        result_from_last_iteration = self.iteration_results[-1]
        joint_definitions = []
        for i in range(self.N_JOINTS+1):
            if i < self.N_JOINTS:
                joint_name = f"{self.joints[i].parent.name}-{self.joints[i].name}"
            else:
                joint_name = f"{self.joints[-1].name}-{self.joints[-1].children[0].name}"
            
            X = SE3(result_from_last_iteration.joint_local_nominal_poses[i])
            rpy = X.rpy(unit='rad',order='zyx')
            xyz = X.t

            joint_def = {}
            joint_def['name'] = joint_name
            joint_def['xyz'] = xyz
            joint_def['rpy'] = rpy
            joint_definitions.append(joint_def)
        
        return joint_definitions


class SerialRobotKineCal:
    '''
    Class implementing the method described in:
        Local POE model for robot kinematic calibration (2001)
    which consists in an iterative least squares method where at each iteration the screw axes defining the pose of the i-th link relative to the previous link (hence local) are updated to minimize the error between the measured and computed end effector pose.
    '''
    def __init__(self, 
                 robot_model:rtb.Robot, 
                 ee_name:str=None,
                 verbose=False):
        '''
        Build lists of links and joints from the roboticstoolbox.Robot model and the selected end effector,
        assuming that the robot is a serial chain from the base to the end effector and that the pose of the
        end effector can be measured relative to the base.

        Parameters
        -----------
        robot_model: roboticstoolbox.ERobot
            The roboticstoolbox model of the robot.
        ee_name: str
            The name of the end effector link. If None, the last link is assumed to be the end effector.
        verbose: bool
            If True, print additional information during the calibration process.
        '''
        assert(isinstance(robot_model, rtb.Robot))
        self.robot_model = robot_model

        #If True, print additional information during the calibration process
        self.verbose = verbose

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

        Parameters
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

        Parameters
        -----------
        link_index: int
            The index of the link for which to compute the forward kinematics.
        joint_positions: numpy.ndarray
            The joint angles as a 1xN_JOINTS numpy array.
        
        Returns
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

        Parameters
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

        Parameters
        -----------
        joint_positions: numpy.ndarray
            The joint angles as a 1xN_JOINTS numpy array.

        Returns
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

        Parameters
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        
        step_size: float
            The step size to apply to the twist corrections during the optimization (default = 1.0).
        '''
        calibration_result = CalibrationResult(self.joints)

        #Iterative least squares
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

            #Store the results of the iteration
            it_result = CalibrationResult.IterationResult(self.joint_local_nominal_poses, 
                                                          self.joint_local_screws,
                                                          A_all,
                                                          x, 
                                                          y_all, 
                                                          position_errors, 
                                                          orientation_errors)
            calibration_result.add_iteration_result(it_result)

            if self.verbose:
                it_result.print()

            if calibration_result.has_converged or calibration_result.is_diverging:
                if self.verbose:
                    if calibration_result.has_converged:
                        print("The kinematic calibration has converged.")
                    else:
                        print("The kinametic calibration is diverging.")
                break

        return calibration_result

    def get_screw_axes(self, calibration_result:CalibrationResult):
        '''
        Compute the screw axes for use with the PoE formula.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.

        Returns
        --------
        A list in which the first N_JOINTS elements are the screw axes of the joints and the last element is the pose of the end effector when all joint angles are zero.
        '''
        return calibration_result.get_screw_axes()
    
    def get_urdf_xyzrpy(self, calibration_result:CalibrationResult):
        '''
        Compute the RPY-XYZ format of the joint definitions for use in a URDF file.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.

        Returns
        --------
        A list of dictionaries where each dictionary contains the following keys:
        - name: The name of the joint (e.g. "link1-link2").
        - xyz: The position of the joint relative to the previous link.
        - rpy: The orientation of the joint relative to the previous link.
        '''
        return calibration_result.get_urdf_xyzrpy()

    def print_screw_axes(self, calibration_result:CalibrationResult):
        '''
        Print the screw axes for use with the PoE formula.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.
        '''
        screw_definitions = self.get_screw_axes(calibration_result)

        print("Definition of the screw axes for use with the PoE formula:")
        for i,s in enumerate(screw_definitions):
            if i < self.N_JOINTS:
                v = s[0:3]
                w = s[3:6]
                print(f"\tJoint {i+1}:")
                print(f"\t\tv: {v}")
                print(f"\t\tw: {w}")
            else:
                print(f"\tZero configuration pose:")
                print(f"\t\t{s.A}")

    def print_urdf_joint_definitions(self, calibration_result:CalibrationResult):
        '''
        Print the joint definitions for use in a URDF file.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.
        '''
        urdf_joint_definitions = self.get_urdf_xyzrpy(calibration_result)

        print('Definition of the joints in RPY-XYZ format for use in a URDF:')
        for j in urdf_joint_definitions:
            print(f"\tJoint {j['name']}")
            print(f"\t\tXYZ: {j['xyz']}")
            print(f"\t\tRPY: {j['rpy']}")