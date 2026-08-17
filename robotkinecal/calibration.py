import numpy as np
import roboticstoolbox as rtb
from numpy.linalg import norm
from spatialmath import SE3, Twist3
from spatialmath.base import skew


class CalibrationResult:
    '''
    Class to store the result of a kinematic calibration.
    '''
    def __init__(
        self,
        robot_model: rtb.Robot,
        convergence_tolerance: float = 1e-4,
        joints: list | None = None,
        divergence_patience: int = 3,
    ) -> None:
        '''
        Initialize the calibration result.

        Parameters
        -----------
        robot_model: roboticstoolbox.Robot
            The robot model used for the calibration.
        convergence_tolerance: float
            The tolerance to consider that the error has converged.
        '''
        self.robot_model = robot_model
        self.joints = list(joints) if joints is not None else [
            link for link in self.robot_model.links if link.isjoint
        ]
        self.N_JOINTS = len(self.joints)
        self.convergence_tolerance = convergence_tolerance
        self.has_converged = False
        self.is_diverging = False
        self.termination_reason = None
        self.divergence_patience = divergence_patience
        self._consecutive_error_increases = 0
        self.nb_iterations_executed = 0
        self.iteration_results = []

    class IterationResult:
        '''
        Class to store the result of an iteration of the calibration.
        '''
        def __init__(self, 
                     joint_screw_definitions:list[np.ndarray],
                     zero_conf_EE_pose:SE3,
                     A_all:np.ndarray,
                     twist_corrections:np.ndarray,
                     twist_errors:np.ndarray,
                     position_errors:list[float],
                     orientation_errors:list[float],
                     step_size:float=1.0,
                     post_update_twist_errors_norm:float | None=None) -> None:
            '''
            Initialize the iteration result.

            Parameters
            ----------
            joint_screw_definitions: list
                List of 6D screw axes for each joint and the end-effector.
            zero_conf_EE_pose: SE3
                Pose of the EE when all joint positions are zero
            A_all: numpy.ndarray
                The data matrix matrix build from stacked A matrices when considered all observations as a numpy.ndarray of shape OBS_SIZE*N_OBSERVATIONS x 4*N_JOINTS+OBS_SIZE.
            twist_corrections: numpy.ndarray
                A 4*N_JOINTS+OBS_SIZE x 1 vector containing the twist correction for each joint and for the end-effector. This is k in SerialRobotKineCal.solve.
            twist_errors: numpy.ndarray
                Vector of length OBS_SIZE*N_OBSERVATIONS containing the twist errors for each joint and the end-effector. This is y_all in SerialRobotKineCal.solve.
            position_errors: List[float]
                List of N_OBSERVATIONS containing the magnitude of the position difference between the observed and computed end-effector poses for each observation.
            orientation_errors: List[float]
                List of N_OBSERVATIONS containing the magnitude of the orientation difference between the observed and computed end-effector poses for each observation.
            '''
            # Store a snapshot: the optimizer continues mutating its model after
            # this result has been created.
            self.zero_conf_EE_pose = SE3(zero_conf_EE_pose.A.copy())
            self.joint_screw_definitions = [axis.copy() for axis in joint_screw_definitions]
            self.A_all = A_all.copy()
            self.twist_corrections = twist_corrections.copy()
            self.step_size = step_size
            self.applied_twist_corrections = step_size * twist_corrections.copy()
            self.twist_errors = twist_errors.copy()
            self.pre_update_twist_errors = self.twist_errors.copy()
            self.pre_update_twist_errors_norm = norm(self.twist_errors)
            self.post_update_twist_errors_norm = post_update_twist_errors_norm
            self.position_errors = list(position_errors)
            self.orientation_errors = list(orientation_errors)

            #Number of joints
            self.N_JOINTS = len(joint_screw_definitions)
            
            #Number of observations
            self.N_OBSERVATIONS = len(position_errors)
            
            # Length of a single observation
            # Will be 3 for position only observations and 6 for pose observations
            self.OBS_SIZE = int(len(self.twist_errors) / self.N_OBSERVATIONS)

            #Number of kinematic parameters in the model
            self.N_PARAMS = 4*self.N_JOINTS+self.OBS_SIZE

            #Whether the observations are 3D/positions or 6D/poses
            self.position_only = self.OBS_SIZE == 3

            # Observability diagnostics for the linearized calibration problem.
            self.matrix_rank = np.linalg.matrix_rank(self.A_all)
            self.singular_values = np.linalg.svd(self.A_all, compute_uv=False)
            self.condition_number = np.linalg.cond(self.A_all)
        
        def compute_uncertainty_estimate(self, A_all, x_all, y_all):
            '''
            Compute the uncertainty estimate of the calibration.

            Parameters
            -----------
            A_all: numpy.ndarray
                The A matrix for all observations as a numpy.ndarray of shape OBS_SIZE*N_OBSERVATIONS x 4*N_JOINTS+OBS_SIZE.
            x_all: numpy.ndarray
                The twist corrections for all observations as a numpy.ndarray of shape 4*N_JOINTS+OBS_SIZE x 1.
            y_all: numpy.ndarray
                The twist errors for all observations as a numpy.ndarray of shape OBS_SIZE*N_OBSERVATIONS x 1.
            
            Returns
            --------
            A numpy.ndarray of shape N_JOINTS+1 containing the norm of the twist variance for each joint and the end-effector (a joint uncertainty estimate).
            '''
            #Sum of squared residuals
            residual = y_all - A_all @ x_all
            SSR = float(np.sum(np.square(residual)))
            #Statistical degrees of freedom
            df = self.OBS_SIZE*self.N_OBSERVATIONS - self.N_PARAMS
            if df <= 0:
                return np.full(self.N_JOINTS, np.nan)
            #Reduced chi-squared statistic
            rcss = SSR/df
            # Use a pseudoinverse so rank-deficient or ill-conditioned datasets
            # produce a best-effort estimate instead of raising LinAlgError.
            sigma = rcss * np.diag(np.linalg.pinv(A_all.T @ A_all))
            #Twist variance norms
            twist_variance_norms = np.linalg.norm(sigma[:-self.OBS_SIZE].reshape((-1,4)), axis=1)
                
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
            stats['matrix_rank'] = self.matrix_rank
            stats['condition_number'] = self.condition_number
            stats['singular_values'] = self.singular_values.copy()
            return stats

        def print(self):
            stats = self.get_statistics()

            print(f"\tNorm of twist errors: {stats['twist_errors_norm']:.4f}")
            print(f"\tAvg. Position error: {stats['position_errors_mean']:.4f}")
            print(f"\tMax. Position error: {stats['position_errors_max']:.4f}")
            print(f"\tAvg. Orientation error: {stats['orientation_errors_mean']:.4f}")
            print(f"\tMax. Orientation error: {stats['orientation_errors_max']:.4f}")
            with np.printoptions(precision=4):
                print(f"\tJoints uncertainty: {stats['joints_uncertainty']}")
            print(f"\tRegressor rank: {stats['matrix_rank']}/{self.N_PARAMS}")
            print(f"\tRegressor condition number: {stats['condition_number']:.4e}")
            print(f"\tApplied step size: {self.step_size:.4e}")
            if self.post_update_twist_errors_norm is not None:
                print(
                    "\tPost-update twist error norm: "
                    f"{self.post_update_twist_errors_norm:.4e}"
                )

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
                self.termination_reason = "converged"
            elif norm(current_result.twist_errors) > norm(previous_result.twist_errors):
                self._consecutive_error_increases += 1
                if self._consecutive_error_increases >= self.divergence_patience:
                    self.is_diverging = True
                    self.termination_reason = "diverged"
            else:
                self._consecutive_error_increases = 0

    def get_screw_axes(self):
        '''
        Compute screw definitions for use with the PoE formula.

        Returns
        --------
        A list of joint screw axes.
        '''
        result_from_last_iteration = self.iteration_results[-1]
        return [axis.copy() for axis in result_from_last_iteration.joint_screw_definitions]

    def get_zero_conf_EE_pose(self):
        '''
        Returns
        --------
        A SE3 object defining the pose of the robot end-effector when all joint positions are zero.
        '''
        result_from_last_iteration = self.iteration_results[-1]
        return SE3(result_from_last_iteration.zero_conf_EE_pose.A.copy())
    
    def get_urdf_xyzrpy(self, zero_conf_joint_poses:list[SE3]):
        '''
        Compute the RPY-XYZ format of the joint definitions for use in a URDF file.

        In a URDF file, each rotary joint is defined by its axis of rotation and the position of the joint relative to the previous link. 

        Parameters
        -----------
        zero_conf_joint_poses: list
            List of SE3 objects defining the pose of each joint and the end-effector when the robot is in the zero configuration.

        Returns
        --------
        A list of dictionaries where each dictionary contains the following keys:
        - name: The name of the joint (e.g. "link1-link2").
        - xyz: The position of the joint relative to the previous link.
        - rpy: The orientation of the joint relative to the previous link.
        '''

        #A PoE chain does not describe joint position, it only describes
        # how the end-effector moves when the joints move. Hence, we cannot
        # use a PoERobot to generate URDF joint definitions.
        if isinstance(self.robot_model, rtb.PoERobot):
            raise TypeError("A PoERobot does not describe joint positions, use a DHRobot or ERobot instead.")

        joint_definitions = []
        previous_link_pose = SE3()
        for i,joint in enumerate(self.joints):
            joint_name = f"{joint.parent.name}-{joint.name}"
            zero_conf_joint_pose = zero_conf_joint_poses[i]

            #Pose of the joint, relative to the parent frame, 
            # when the robot is in the zero configuration
            X = previous_link_pose.inv() * zero_conf_joint_pose
            rpy = X.rpy(unit='rad',order='zyx')
            xyz = X.t

            joint_def = {}
            joint_def['name'] = joint_name
            joint_def['xyz'] = xyz
            joint_def['rpy'] = rpy
            joint_definitions.append(joint_def)

            previous_link_pose = zero_conf_joint_pose

        #End-effector
        joint_name = f"{self.joints[-1].name}-{self.joints[-1].children[0].name}"
        zero_conf_ee_pose = self.iteration_results[-1].zero_conf_EE_pose
        X = previous_link_pose.inv() * zero_conf_ee_pose
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
        Li, C., Wu, Y., Löwe, H., & Li, Z. (2016). POE-based robot kinematic calibration using axis configuration space and the adjoint error model. 
        IEEE Transactions on Robotics, 32(5), 1264-1279.
    which consists in an iterative least squares method where at each iteration the screw axes of the joints are 
    updated to minimize the error between the measured and computed end effector pose.

    The number of parameters to estimate is minimal (4 per joint), yielding greater calibration speed. In contrast to DH parameters,
    screw axes are much easier to optimize over, and the calibration is less likely to get stuck in local minima.

    The input data consists of joint positions and end-effector poses or points. When supplying end-effector points as observations,
    a greater number of observations is required to yield the same accuracy as when supplying end-effector poses. Furthermore, the
    orientation of the end-effector is not identifiable when using points, only its position is identifiable.
    '''
    def __init__(self, 
                 robot_model:rtb.Robot, 
                 ee_name:str | None=None,
                 verbose=False):
        '''
        Build lists of links and joints from the roboticstoolbox.Robot model and the selected end effector,
        assuming that the robot is a serial chain from the base to the end effector and that the pose of the
        end effector can be measured relative to the base.

        Parameters
        -----------
        robot_model: roboticstoolbox.Robot
            The roboticstoolbox model of the robot.
        ee_name: str
            The name of the end effector link. If None, the last link is assumed to be the end effector.
        verbose: bool
            If True, print additional information during the calibration process.
        '''
        if not isinstance(robot_model, rtb.Robot):
            raise TypeError("robot_model must be an instance of roboticstoolbox.Robot.")
        self.robot_model = robot_model

        #If True, print additional information during the calibration process
        self.verbose = verbose

        #If True, only the position information is used for calibration
        self.position_only = False

        #Each element of self.joints is a roboticstoolbox Link object
        self.joints = []

        #Each element of joint_screw_axis is the screw axis of the joint
        self.joint_screw_axis = []

        #Pose of the end-effector when all joint positions are zero
        self.zero_conf_EE_pose = None

        #The structure of PoERobot is different from the one DHRobot or ERobot
        if isinstance(robot_model, rtb.PoERobot):
            # RTB 1.3 inserts fixed base and tool sentinel links but reports
            # robot.n == 0 because PoELink ETS objects are world-frame constants.
            # Older releases exposed only the actual POE links. Support both
            # representations without relying on the broken ``n`` property.
            if robot_model.n == 0 and len(robot_model.links) >= 2:
                self.joints = list(robot_model.links[1:-1])
            else:
                self.joints = list(robot_model.links)
            self.joint_screw_axis = [link.S.A.copy() for link in self.joints]
            self.zero_conf_EE_pose = self.robot_model.T0
            # With a PoERobot, local poses are undefined since
            # the PoE formula does not describe joint poses.
            self.joint_local_nominal_poses = None
            self.joint_zero_conf_poses = None
        elif isinstance(robot_model, rtb.DHRobot):
            # DHRobot does not support fkine(q, end=link). Robotics Toolbox can
            # provide its equivalent POE representation directly, which is both
            # simpler and less error-prone than reconstructing screw axes from DH.
            screw_axes, self.zero_conf_EE_pose = robot_model.twists()
            self.joints = list(robot_model.links)
            self.joint_screw_axis = [axis.S.copy() for axis in screw_axes]
            self.ee_name = ee_name or self.joints[-1].name
            if ee_name is not None and ee_name != self.joints[-1].name:
                raise ValueError(
                    "Selecting an intermediate end effector is not supported for DHRobot."
                )
            self.joint_local_nominal_poses = [link.A(0) for link in self.joints]
            self.joint_zero_conf_poses = []
            zero_pose = SE3(robot_model.base)
            for link in self.joints:
                zero_pose *= link.A(0)
                self.joint_zero_conf_poses.append(SE3(zero_pose.A.copy()))
        else:
            #If the user has not specified the end effector link,
            # assume that the last link is the end effector.
            if ee_name is None:
                if not self.robot_model.links:
                    raise ValueError("The robot model does not contain any links.")
                ee_name = self.robot_model.links[-1].name
            self.ee_name = ee_name

            #Verify that the end effector link exists
            ee_link = self.robot_model.link_dict.get(ee_name)
            if ee_link is None:
                raise ValueError(f"End-effector link '{ee_name}' does not exist.")

            # Follow parent links from the selected end effector to the base.
            # This excludes gripper or other branch joints that do not affect the
            # selected end effector, and provides an unambiguous serial order.
            chain_links = []
            link = ee_link
            visited_links = set()
            while link is not None:
                if id(link) in visited_links:
                    raise ValueError("The robot model contains a cycle in its link graph.")
                visited_links.add(id(link))
                chain_links.append(link)
                link = link.parent
            chain_links.reverse()

            if self.robot_model.base_link not in chain_links:
                raise ValueError(
                    f"End-effector link '{ee_name}' is not connected to the robot base."
                )

            #Each element of self.joints is a roboticstoolbox Link object
            self.joints = [link for link in chain_links if link.isjoint]
            if not self.joints:
                raise ValueError("The selected end-effector chain contains no joints.")

            #Each element of joint_local_nominal_poses is the pose of the i-th link relative to the previous link
            # when the joint angles are zero.
            self.joint_local_nominal_poses = [l.A() for l in self.joints]

            #For each joint, define the screw axis of a revolute joint rotating about Z
            # and record the pose of the joint when the robot is in the zero configuration.
            self.joint_zero_conf_poses = []
            for l in chain_links:
                zero_conf_joint_pose = robot_model.fkine(np.zeros(robot_model.n), l.name)
                if l.isjoint:
                    # NOTE: Assumes that the joint rotates about Z.
                    # This is a limitation of the roboticstolbox, which might be fixed in the future.
                    # See: https://github.com/petercorke/robotics-toolbox-python/pull/441
                    screw_axis = np.block([skew(zero_conf_joint_pose.t) @ zero_conf_joint_pose.R[:, 2], zero_conf_joint_pose.R[:, 2]])
                    self.joint_screw_axis.append(screw_axis)
                    #Record the pose of the joint when the robot is in the zero configuration
                    self.joint_zero_conf_poses.append(zero_conf_joint_pose)
                if l == ee_link:
                    self.zero_conf_EE_pose = zero_conf_joint_pose

            if self.zero_conf_EE_pose is None:
                raise ValueError(f"Could not compute the zero pose of end-effector '{ee_name}'.")

        #Verify that the robot has only revolute joints
        # A future version could support prismatic joints.
        for j in self.joints:
            if j.isprismatic:
                raise ValueError("Only revolute joints are currently supported.")

        #Number of joints
        self.N_JOINTS = len(self.joint_screw_axis)

        #List of observed end-effector poses that will be populated by set_data
        self._observations = []

        #List of measured joint positions that will be populated by set_data
        self._joint_positions = []

        #Placeholder for the B matrix that has to be computed only once.
        self._B_Matrix = None

        # Preserve the nominal state so callers can run independent calibration
        # experiments with the same object.
        self._initial_model_state = self._model_snapshot()

    def reset(self):
        """Restore the nominal kinematic model used at construction time.

        Observations are retained, allowing another solve with different
        numerical options without reloading the dataset.
        """
        self._restore_model_snapshot(self._initial_model_state)
        self._B_Matrix = None

    def set_observations(self, joint_positions, observations):
        '''
        Parameters
        ----------
        joint_positions: list
            List of numpy arrays where each array has a shape of 1xN_JOINTS.
        observations: list
            List of SE3 objects in the case where observations are poses 
            OR of 3x1 arrays in the case where observations are points.
        '''
        if len(joint_positions) != len(observations):
            raise ValueError("The size of both lists must be equal.")
        if not observations:
            raise ValueError("At least one observation is required.")

        normalized_joint_positions = []
        for index, positions in enumerate(joint_positions):
            positions = np.asarray(positions, dtype=float).reshape(-1)
            if positions.shape != (self.N_JOINTS,):
                raise ValueError(
                    f"Joint configuration {index} must contain exactly "
                    f"{self.N_JOINTS} values."
                )
            if not np.all(np.isfinite(positions)):
                raise ValueError(f"Joint configuration {index} contains non-finite values.")
            normalized_joint_positions.append(positions.copy())

        if isinstance(observations[0], SE3):
            self.position_only = False
            self.OBS_SIZE = 6
            if not all(isinstance(observation, SE3) for observation in observations):
                raise ValueError("All observations must have the same type.")
            normalized_observations = [SE3(observation.A.copy()) for observation in observations]
        else:
            self.position_only = True
            self.OBS_SIZE = 3
            normalized_observations = []
            for index, observation in enumerate(observations):
                try:
                    point = np.asarray(observation, dtype=float).reshape(-1)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Observations must be SE3 poses or 3D points.") from exc
                if point.shape != (3,):
                    raise ValueError(f"Position observation {index} must contain exactly 3 values.")
                if not np.all(np.isfinite(point)):
                    raise ValueError(f"Position observation {index} contains non-finite values.")
                normalized_observations.append(point.copy())

        #Number of kinematic parameters in the model
        self.N_PARAMS = 4*self.N_JOINTS+self.OBS_SIZE

        #Number of observations (not equals to the length of the observation vector)
        self.N_OBSERVATIONS = len(observations)

        self._observations = normalized_observations
        self._joint_positions = normalized_joint_positions

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

        joint_positions = np.asarray(joint_positions, dtype=float).reshape(-1)
        if joint_positions.shape != (self.N_JOINTS,):
            raise ValueError(
                f"joint_positions must contain exactly {self.N_JOINTS} values."
            )
        if not np.all(np.isfinite(joint_positions)):
            raise ValueError("joint_positions contains non-finite values.")

        T = SE3()
        for i in range(min(link_index, self.N_JOINTS)):
            T *= Twist3(self.joint_screw_axis[i]).exp(joint_positions[i])

        #If the user desire the pose of the end effector
        if link_index == self.N_JOINTS + 1:
            T *= self.zero_conf_EE_pose
        
        return T

    def update_twist_definitions(self, param_corrections:np.ndarray, step_size:float=1.0):
        '''
        Update the twist definitions of the joints such as to account for the error in the EE pose.

        Parameters
        -----------
        param_corrections: numpy.ndarray
            The parameter corrections to apply to each joint that, together, account for the error in the EE pose.
            The shape of the array is 4*N_JOINTS+OBS_SIZE x 1.
        step_size: float
            The step size to apply to the twist corrections during the optimization (default = 1.0).
        '''
        #Verify that the shape of the array is correct
        param_corrections = np.asarray(param_corrections, dtype=float).reshape(-1)
        if param_corrections.shape != (4*self.N_JOINTS+self.OBS_SIZE,):
            raise ValueError(f"The shape of the parameter corrections array must be (4*N_JOINTS+OBS_SIZE, 1)=({4*self.N_JOINTS+self.OBS_SIZE}, 1).")
        
        #Verify that the step size is positive
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("The step size must be positive.")
        param_corrections = step_size * param_corrections

        #Get the B matrix that maps the 4 parameters of each joint to a 6-dimensional twist
        B_all = self.B_Matrix()

        #Update the pose of the joints
        for i in range(self.N_JOINTS):
            B_i = B_all[6*i:6*(i+1), 4*i:4*(i+1)]
            k_i = param_corrections[4*i:4*(i+1)].reshape(4)
            # Eq 29 in Adjoint Error Model paper
            # This produces a 6-dimensional twist correction from the 4 parameters
            n_i = B_i @ k_i

            #Update the screw axis of the joint
            new_screw_axis_i = Twist3(n_i).SE3().Ad() @ self.joint_screw_axis[i]
            self.joint_screw_axis[i] = new_screw_axis_i

            if self.joint_zero_conf_poses is not None:
                #Update the pose of the joint when the robot is in the zero configuration
                self.joint_zero_conf_poses[i] = Twist3(n_i).SE3() @ self.joint_zero_conf_poses[i]
        
        #Update the pose of the end effector
        # Eq 29 in Adjoint Error Model paper
        if self.position_only:
            self.zero_conf_EE_pose.t += param_corrections[-3:].reshape(3)
        else:
            k_st = param_corrections[-6:].reshape(6)
            new_ee_pose = Twist3(k_st).SE3() @ self.zero_conf_EE_pose
            self.zero_conf_EE_pose = new_ee_pose
        

    def B_Matrix(self):
        '''
        Matrix of bases. Each B is a matrix mapping 4 parameters to a twist about a given screw axis.

        The minimal parametrization of a joint is 4-dimensional (as with DH parameters) but a twist is 6-dimensional.
        Hence, identifying twist corrections directly results in an over parametrization (although it is possible).
        This algorithm uses a minimal parametrization (4 parameters per joint) and B is used to map between a set of
        4 parameters to a 6 dimensional twist.
        '''

        #NOTE: Since the definition of the B matrix is partly random
        # we need to compute it only once and store it.

        #If the B matrix has already been computed, return it
        if self._B_Matrix is not None:
            return self._B_Matrix

        B_all = np.zeros((6*self.N_JOINTS+self.OBS_SIZE, 4*self.N_JOINTS+self.OBS_SIZE))
        
        for i in range(self.N_JOINTS):
            joint_screw_axis = self.joint_screw_axis[i]
            v = joint_screw_axis[0:3]
            w = joint_screw_axis[3:6]
            q = np.cross(w, v)
            #Find mutually perpendicular vectors w_1 and w_2 that are perpendicular to w_n
            # Deterministically choose the Cartesian axis least aligned with w.
            # This avoids random results and near-zero cross products.
            reference_axis = np.eye(3)[np.argmin(np.abs(w))]
            w_1 = np.cross(w, reference_axis)
            w_1 /= np.linalg.norm(w_1)
            w_2 = np.cross(w, w_1)
            w_2 /= np.linalg.norm(w_2)
            #Fill in the B matrix with column basis vectors
            B = np.zeros((6, 4))
            B[0:3, 0] = w_1
            B[0:3, 1] = w_2
            B[0:3, 2] = np.cross(q, w_1)
            B[3:6, 2] = w_1
            B[0:3, 3] = np.cross(q, w_2)
            B[3:6, 3] = w_2
            B_all[6*i:6*(i+1), 4*i:4*(i+1)] = B
        
        #Set E_st
        B_all[-self.OBS_SIZE:, -self.OBS_SIZE:] = np.eye(self.OBS_SIZE)

        # If position_only, B is a (6*N_JOINTS+3) x (4*N_JOINTS+3) matrix
        # otherwise, B is a (6*N_JOINTS+6) x (4*N_JOINTS+6) matrix
        self._B_Matrix = B_all
        return B_all 

    def Q_Matrix(self, joint_positions:np.ndarray):

        Q = np.ndarray((6, 6*self.N_JOINTS+self.OBS_SIZE))
        prev_Ad = np.eye(6)
        for i in range(self.N_JOINTS):
            # Eq 20 of Adjoint Error Model paper
            Ad_i = prev_Ad @ Twist3(self.joint_screw_axis[i]).exp(joint_positions[i]).Ad()
            # Eq 25 of Adjoint Error Model paper
            Q_i = prev_Ad - Ad_i
            Q[:,6*i:6*(i+1)] = Q_i
            prev_Ad = Ad_i

        # Set Q_st according to Eq 25 of Adjoint Error Model paper
        Q[:,-self.OBS_SIZE:] = prev_Ad[:,:self.OBS_SIZE]

        return Q # A 6 x (6*N_JOINTS+6) matrix


    def A_Matrix(self, joint_positions:np.ndarray):
        '''
        Build a matrix of elements that will be multiplied by the joint pose perturbations such as to account for the TCP error. 
        See equations (16) and (25) in the paper.

        Parameters
        -----------
        joint_positions: numpy.ndarray
            The joint angles as a 1xN_JOINTS numpy array.

        Returns
        --------
        The A matrix for the current observation as a numpy.ndarray of shape 6x6(N_JOINTS+1).
        '''
        Q = self.Q_Matrix(joint_positions)
        B = self.B_Matrix()

        A = Q @ B
        
        # If position_only, A is a 6 x (4*N_JOINTS+3) matrix
        # otherwise, A is a 6 x (4*N_JOINTS+6) matrix
        return A

    def _twist_error_vector(self):
        """Return the stacked residual vector for the current model state."""
        errors = np.zeros((self.OBS_SIZE*self.N_OBSERVATIONS, 1))
        for index, (joint_positions, observation) in enumerate(
            zip(self._joint_positions, self._observations)
        ):
            model_pose = self.forward_kinematics(
                self.N_JOINTS + 1, joint_positions
            )
            if self.position_only:
                error = observation - model_pose.t
            else:
                error = (observation @ model_pose.inv()).log(twist=True)
            errors[
                self.OBS_SIZE*index:self.OBS_SIZE*(index+1)
            ] = error.reshape((self.OBS_SIZE, 1))
        return errors

    def _model_snapshot(self):
        """Capture mutable calibration state for a recoverable trial update."""
        joint_poses = None
        if self.joint_zero_conf_poses is not None:
            joint_poses = [SE3(pose.A.copy()) for pose in self.joint_zero_conf_poses]
        return (
            [axis.copy() for axis in self.joint_screw_axis],
            SE3(self.zero_conf_EE_pose.A.copy()),
            joint_poses,
        )

    def _restore_model_snapshot(self, snapshot):
        """Restore state captured by :meth:`_model_snapshot`."""
        screw_axes, zero_pose, joint_poses = snapshot
        self.joint_screw_axis = [axis.copy() for axis in screw_axes]
        self.zero_conf_EE_pose = SE3(zero_pose.A.copy())
        self.joint_zero_conf_poses = None if joint_poses is None else [
            SE3(pose.A.copy()) for pose in joint_poses
        ]

    def solve(
        self,
        max_iterations:int=100,
        step_size:float=1.0,
        position_weight:float=1.0,
        orientation_weight:float=1.0,
        damping:float=0.0,
        backtracking:bool=True,
        min_step_size:float=1e-4,
    ):
        '''
        Solve the kinematic calibration problem via a iterative least squares method where at each iteration the screw axes are updated to minimize the error in the end effector pose. The process is stopped when the error has converged, when the error is increasing, or when the maximum number of iterations has been reached.

        Parameters
        -----------
        max_iterations: int
            The maximum number of iterations to perform.
        
        step_size: float
            The step size to apply to the twist corrections during the optimization (default = 1.0).
        position_weight: float
            Weight applied to position residuals (default = 1.0).
        orientation_weight: float
            Weight applied to orientation residuals for pose observations (default = 1.0).
        damping: float
            Non-negative Tikhonov damping coefficient for the least-squares
            problem (default = 0.0).
        backtracking: bool
            If True, halve the step until it reduces the residual (default = True).
        min_step_size: float
            Smallest step considered during backtracking (default = 1e-4).
        '''

        if not self._observations:
            raise RuntimeError("No observations have been set. Call set_observations() first.")
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be a positive finite number.")
        if not np.isfinite(position_weight) or position_weight <= 0:
            raise ValueError("position_weight must be a positive finite number.")
        if not np.isfinite(orientation_weight) or orientation_weight <= 0:
            raise ValueError("orientation_weight must be a positive finite number.")
        if not np.isfinite(damping) or damping < 0:
            raise ValueError("damping must be a non-negative finite number.")
        if not isinstance(backtracking, bool):
            raise TypeError("backtracking must be a boolean.")
        if not np.isfinite(min_step_size) or min_step_size <= 0:
            raise ValueError("min_step_size must be a positive finite number.")
        if min_step_size > step_size:
            raise ValueError("min_step_size cannot be greater than step_size.")

        if self.position_only:
            if 3*self.N_OBSERVATIONS < 4*self.N_JOINTS+3:
                raise ValueError("Not enough observations to solve.")
        else:
            if 6*self.N_OBSERVATIONS < 4*self.N_JOINTS+6:
                raise ValueError("Not enough observations to solve.")
            
        calibration_result = CalibrationResult(self.robot_model, joints=self.joints)

        #Iterative least squares
        for it in range(max_iterations):
            y_all = np.zeros((self.OBS_SIZE*self.N_OBSERVATIONS, 1))
            A_all = np.ndarray((self.OBS_SIZE*self.N_OBSERVATIONS, self.N_PARAMS))
            
            #Lists to store the position and orientation errors
            position_errors = []
            orientation_errors = []

            #Iterate over all observations
            for m in range(self.N_OBSERVATIONS):
                #End effector pose according to the observation
                obs_ee_pose = self._observations[m]
                
                #Joint positions as recorded by the robot
                joint_pos = self._joint_positions[m]

                #End effector pose according to the model
                mod_ee_pose = self.forward_kinematics(self.N_JOINTS+1, joint_pos)

                if self.position_only:
                    # EE position error vector
                    y = obs_ee_pose - mod_ee_pose.t
                else:
                    #Pose difference between observed and predicted end effector poses
                    pose_diff = obs_ee_pose @ mod_ee_pose.inv()

                    # EE twist error vector
                    y = pose_diff.log(twist=True) 

                #Record position/orientation errors for convergence detection
                if self.position_only:
                    position_error = np.linalg.norm(y)
                    orientation_error = 0
                else:
                    position_error = np.linalg.norm(pose_diff.t)
                    orientation_error = pose_diff.angvec()[0]
                    
                position_errors.append(position_error)
                orientation_errors.append(orientation_error)

                #Jacobian matrix
                A = self.A_Matrix(joint_pos)

                if self.position_only:
                    #Eq. (26) in the paper
                    A = np.block([np.eye(3), -skew(mod_ee_pose.t)]) @ A

                #Populate the y_all and A_all matrices
                y_all[self.OBS_SIZE*m:self.OBS_SIZE*(m+1)] = y.reshape((self.OBS_SIZE,1))
                A_all[self.OBS_SIZE*m:self.OBS_SIZE*(m+1), :] = A

            if self.position_only:
                row_weights = np.full(self.OBS_SIZE*self.N_OBSERVATIONS, position_weight)
            else:
                observation_weights = np.array(
                    [position_weight] * 3 + [orientation_weight] * 3
                )
                row_weights = np.tile(observation_weights, self.N_OBSERVATIONS)
            weighted_A_all = row_weights[:, None] * A_all
            weighted_y_all = row_weights[:, None] * y_all

            identifiable_parameters = np.array(range(self.N_PARAMS))
            #Compute the column rank of A_all
            A_rank = np.linalg.matrix_rank(weighted_A_all)
            if A_rank < len(identifiable_parameters):
                if self.position_only and len(identifiable_parameters) - A_rank == 2:
                    #In this case, it is likely that the measurement point lie on the axis
                    # of the last revolute joint. Consequently, the end-effector and last
                    # joint poses are confounded.
                    print('WARNING: Verify that the measurement point does not lie on the axis of the last revolute joint.')
                print(f'WARNING: Not all parameters are identifiable (rank of regressor is {A_rank}/{len(identifiable_parameters)}). You might be lacking diverse data.')
                
                #Iteratively detect and record parameters that contribute to the the rank (identifiable ones)
                identifiable_parameters = []
                for i in range(self.N_PARAMS):
                    if np.linalg.matrix_rank(weighted_A_all[:,identifiable_parameters+[i]]) == len(identifiable_parameters)+1:
                        identifiable_parameters.append(i)
                    else:
                        joint_nb, param_nb = divmod(i,4)
                        print(f"Parameter {i} (joint {joint_nb+1}, param {param_nb+1}) is not identifiable.")
                #Remove unidentifiable parameters from the A matrix
                A_ident = weighted_A_all[:,identifiable_parameters]
                A_rank = np.linalg.matrix_rank(A_ident)
                if A_rank != len(identifiable_parameters):
                    raise RuntimeError("Failed to isolate a full-rank set of identifiable parameters.")
            else:
                A_ident = weighted_A_all


            #Solve for the perturbation to the joint zero pose such that the EE error is accounted for
            if damping > 0:
                A_solve = np.vstack(
                    [A_ident, np.sqrt(damping) * np.eye(A_ident.shape[1])]
                )
                y_solve = np.vstack(
                    [weighted_y_all, np.zeros((A_ident.shape[1], 1))]
                )
            else:
                A_solve = A_ident
                y_solve = weighted_y_all
            k = np.linalg.lstsq(A_solve, y_solve, rcond=1e-12)[0]

            #If some kinematic parameters were removed due to a rank deficiency
            # we set them to zero here such that k_full is complete again.
            k_full = np.zeros((self.N_PARAMS, 1))
            for k_idx, k_full_idx in enumerate(identifiable_parameters):
                k_full[k_full_idx] = k[k_idx]
            k = k_full

            # Update the model. With backtracking enabled, reject trial steps
            # that increase the residual and restore the exact previous state.
            model_before_update = self._model_snapshot()
            pre_update_error_norm = norm(y_all)
            applied_step_size = step_size
            line_search_failed = False
            while True:
                self._restore_model_snapshot(model_before_update)
                self.update_twist_definitions(k, applied_step_size)
                post_update_error_norm = norm(self._twist_error_vector())
                if not backtracking or post_update_error_norm <= pre_update_error_norm:
                    break
                applied_step_size *= 0.5
                if applied_step_size < min_step_size:
                    self._restore_model_snapshot(model_before_update)
                    applied_step_size = 0.0
                    post_update_error_norm = pre_update_error_norm
                    line_search_failed = True
                    break

            #Store the results of the iteration
            it_result = CalibrationResult.IterationResult(self.joint_screw_axis,
                                                          self.zero_conf_EE_pose,
                                                          A_all,
                                                          k, 
                                                          y_all, 
                                                          position_errors, 
                                                          orientation_errors,
                                                          applied_step_size,
                                                          post_update_error_norm)
            calibration_result.add_iteration_result(it_result)

            if line_search_failed:
                calibration_result.is_diverging = False
                if pre_update_error_norm <= calibration_result.convergence_tolerance:
                    # At machine precision, no trial step may be measurably
                    # better. Treat this as convergence instead of a failed
                    # search caused by floating-point noise.
                    calibration_result.has_converged = True
                    calibration_result.termination_reason = "converged"
                else:
                    calibration_result.has_converged = False
                    calibration_result.termination_reason = "line_search_failed"

            if self.verbose:
                print(f"Iteration #{calibration_result.nb_iterations_executed} result:")
                it_result.print()

            if line_search_failed or calibration_result.has_converged or calibration_result.is_diverging:
                if self.verbose:
                    if line_search_failed and not calibration_result.has_converged:
                        print("The line search could not find an error-reducing step.")
                    elif calibration_result.has_converged:
                        print("The kinematic calibration has converged.")
                    else:
                        print("The kinematic calibration is diverging.")
                break

        if calibration_result.termination_reason is None:
            calibration_result.termination_reason = "max_iterations"

        return calibration_result

    def get_calibration(self, calibration_result:CalibrationResult):
        '''
        Compute the screw axes for use with the PoE formula.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.

        Returns
        --------
            screw_axes, zero_conf_EE_pose
        '''
        return calibration_result.get_screw_axes(), calibration_result.get_zero_conf_EE_pose()

    
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
        return calibration_result.get_urdf_xyzrpy(self.joint_zero_conf_poses)

    def print_screw_axes(self, calibration_result:CalibrationResult):
        '''
        Print the screw axes for use with the PoE formula.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.
        '''
        screw_definitions = calibration_result.get_screw_axes()
        zero_conf_EE_pose = calibration_result.get_zero_conf_EE_pose()

        print("Definition of the screw axes for use with the PoE formula:")
        for i,s in enumerate(screw_definitions):
            v = s[0:3]
            w = s[3:6]
            print(f"\tJoint {i+1}:")
            print(f"\t\tv: {v}")
            print(f"\t\tw: {w}")
        print("\tZero configuration pose:")
        print(f"\t\t{zero_conf_EE_pose.A[0,:]}")
        print(f"\t\t{zero_conf_EE_pose.A[1,:]}")
        print(f"\t\t{zero_conf_EE_pose.A[2,:]}")
        print(f"\t\t{zero_conf_EE_pose.A[3,:]}")

    def print_urdf_joint_definitions(self, calibration_result:CalibrationResult):
        '''
        Print the joint definitions for use in a URDF file.

        Parameters
        -----------
        calibration_result: CalibrationResult
            The result of the calibration.
        '''
        if self.joint_local_nominal_poses is not None:
            urdf_joint_definitions = self.get_urdf_xyzrpy(calibration_result)

            print('Definition of the joints in RPY-XYZ format for use in a URDF:')
            for j in urdf_joint_definitions:
                print(f"\tJoint {j['name']}")
                print(f"\t\tXYZ: {j['xyz']}")
                print(f"\t\tRPY: {j['rpy']}")
        else:
            print("Cannot print URDF joint definitions for a PoERobot.")
