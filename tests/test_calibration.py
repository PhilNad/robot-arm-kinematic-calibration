import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from robotkinecal import CalibrationResult, SerialRobotKineCal

CONFIGURATIONS = [
    [0.74, 2.59, 1.83, 3.09, 2.88, 1.83, -1.35],
    [0.55, -0.18, -2.47, -1.70, 2.51, -0.52, 0.23],
    [1.55, 2.08, 0.84, -0.39, -2.18, 0.43, 0.18],
    [0.36, 2.93, -2.22, -2.96, 0.59, -2.42, 2.83],
    [-0.95, -1.99, 2.52, 1.30, 1.42, 2.51, 1.75],
    [0.57, 2.22, -1.34, -2.05, -2.30, 3.11, -2.01],
    [1.44, -2.25, 0.33, -1.43, 2.98, 1.05, -1.54],
    [1.25, 2.84, 2.45, 3.10, 2.00, 0.28, -0.31],
    [-0.84, 1.90, 1.78, 1.27, 0.77, -0.04, 2.14],
    [-2.35, -2.06, 1.49, -2.34, -0.82, 0.66, -2.49],
]


def make_calibrator():
    robot = rtb.models.URDF.Panda()
    terminal_offset = SE3.RPY([0.02, -0.01, 0.03]) * SE3(0.01, -0.02, 0.015)
    observations = [
        robot.fkine(q, end="panda_link8") @ terminal_offset
        for q in CONFIGURATIONS
    ]
    calibrator = SerialRobotKineCal(robot, ee_name="panda_link8")
    calibrator.set_observations(CONFIGURATIONS, observations)
    return calibrator


def test_minimal_calibration_converges():
    calibrator = make_calibrator()
    result = calibrator.solve(max_iterations=20)

    assert isinstance(result, CalibrationResult)
    assert result.has_converged
    assert result.termination_reason == "converged"
    assert result.iteration_results[-1].post_update_twist_errors_norm < 1e-6


def test_weighted_calibration_uses_backtracking_objective():
    calibrator = make_calibrator()
    result = calibrator.solve(
        max_iterations=20,
        position_weight=100.0,
        orientation_weight=0.01,
    )

    assert result.termination_reason == "converged"
    assert result.iteration_results[-1].post_update_twist_errors_norm < 1e-6


def test_reset_restores_nominal_model_and_retains_observations():
    calibrator = make_calibrator()
    nominal_axes = [axis.copy() for axis in calibrator.joint_screw_axis]
    nominal_pose = calibrator.zero_conf_EE_pose.A.copy()

    calibrator.solve(max_iterations=2)
    calibrator.reset()

    for restored, nominal in zip(calibrator.joint_screw_axis, nominal_axes):
        np.testing.assert_allclose(restored, nominal)
    np.testing.assert_allclose(calibrator.zero_conf_EE_pose.A, nominal_pose)
    assert len(calibrator._observations) == len(CONFIGURATIONS)


def test_public_api_surface():
    assert SerialRobotKineCal is not None
    assert CalibrationResult is not None
