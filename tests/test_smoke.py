import numpy as np
import pytest
import roboticstoolbox as rtb
from spatialmath import SE3

from robotkinecal import CalibrationResult, SerialRobotKineCal

CONFIGURATIONS = [
    [0.7379080192092227, 2.5894444647892367, 1.825416964205039, 3.091839038290101, 2.8827364908669937, 1.8344647650878239, -1.3493080127049217],
    [0.5485449070443003, -0.17823804356018647, -2.468495009792379, -1.7013699309548564, 2.513055435581019, -0.5230529481527988, 0.22526263925660972],
    [1.5460619969223375, 2.079778271432109, 0.840223786664084, -0.38761044852972537, -2.1829496374793194, 0.4298302893909538, 0.1773383662282102],
    [0.36292198850587143, 2.926937253360429, -2.216978582332016, -2.9553150554237244, 0.589950213299741, -2.424896731196723, 2.832521826380389],
    [-0.9549889194638133, -1.99435381806784, 2.524559046456991, 1.2976547203598257, 1.4241371154351716, 2.5138260178240195, 1.7540378912902277],
    [0.5685139443689309, 2.223610277195741, -1.3379262425863947, -2.0541791969127745, -2.2995125812073756, 3.10800166827796, -2.013774277469671],
    [1.4352597031044043, -2.2466448085858484, 0.32967206973427254, -1.4260112561208396, 2.9813408799580197, 1.0542362234276208, -1.5352756805346326],
    [1.2459328733337909, 2.8412275549266592, 2.4502115961101003, 3.101175202445268, 2.002473212659864, 0.28351093156853135, -0.30627980563416113],
    [-0.844340175819883, 1.8950250155197264, 1.7764801173164066, 1.2651531603415629, 0.7714280450108828, -0.03969310742199372, 2.1396614739252167],
    [-2.348621231134324, -2.0642446452075056, 1.4896583736744446, -2.3434434343157413, -0.8190139905243758, 0.6555498862249252, -2.4937683582669945],
]


def test_minimal_calibration_converges():
    robot_model = rtb.models.URDF.Panda()
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]

    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8", verbose=False)
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)
    result = cal.solve(max_iterations=20)

    assert isinstance(result, CalibrationResult)
    assert result.has_converged
    last_stats = result.iteration_results[-1].get_statistics()
    assert last_stats["position_errors_max"] < 1e-3
    assert last_stats["orientation_errors_max"] < 1e-3


def test_fixed_measurement_target_transform_is_absorbed_by_terminal_pose():
    """A rigid EE-to-target offset must not change the calibrated joint axes."""
    reference_model = rtb.models.URDF.Panda()
    observed_ee_poses = [
        reference_model.fkine(q, end="panda_link8")
        for q in CONFIGURATIONS
    ]
    ee_T_target = SE3.RPY([0.20, -0.15, 0.10], order="zyx") * SE3(
        0.04, -0.03, 0.08
    )
    observed_target_poses = [pose @ ee_T_target for pose in observed_ee_poses]

    ee_cal = SerialRobotKineCal(
        rtb.models.URDF.Panda(), ee_name="panda_link8"
    )
    ee_cal.set_observations(CONFIGURATIONS, observed_ee_poses)
    ee_result = ee_cal.solve(max_iterations=20)

    target_cal = SerialRobotKineCal(
        rtb.models.URDF.Panda(), ee_name="panda_link8"
    )
    target_cal.set_observations(CONFIGURATIONS, observed_target_poses)
    target_result = target_cal.solve(max_iterations=20)

    assert ee_result.has_converged
    assert target_result.has_converged
    for ee_axis, target_axis in zip(
        ee_result.get_screw_axes(), target_result.get_screw_axes()
    ):
        np.testing.assert_allclose(target_axis, ee_axis, atol=1e-7, rtol=1e-7)

    expected_target_zero_pose = ee_result.get_zero_conf_EE_pose() @ ee_T_target
    np.testing.assert_allclose(
        target_result.get_zero_conf_EE_pose().A,
        expected_target_zero_pose.A,
        atol=1e-7,
        rtol=1e-7,
    )

    for q, observed_target_pose in zip(CONFIGURATIONS, observed_target_poses):
        predicted_target_pose = target_cal.forward_kinematics(
            target_cal.N_JOINTS + 1, q
        )
        np.testing.assert_allclose(
            predicted_target_pose.A,
            observed_target_pose.A,
            atol=1e-7,
            rtol=1e-7,
        )


def test_camera_and_target_transforms_define_a_camera_frame_poe_model():
    """Fixed eye-to-hand transforms can be absorbed into a composite POE model."""
    reference_model = rtb.models.URDF.Panda()
    base_T_ee_poses = [
        reference_model.fkine(q, end="panda_link8")
        for q in CONFIGURATIONS
    ]
    camera_T_base = SE3.RPY([-0.10, 0.25, -0.18], order="zyx") * SE3(
        0.6, -0.4, 1.2
    )
    ee_T_target = SE3.RPY([0.20, -0.15, 0.10], order="zyx") * SE3(
        0.04, -0.03, 0.08
    )
    camera_T_target_poses = [
        camera_T_base @ base_T_ee @ ee_T_target
        for base_T_ee in base_T_ee_poses
    ]

    cal = SerialRobotKineCal(
        rtb.models.URDF.Panda(), ee_name="panda_link8"
    )
    cal.set_observations(CONFIGURATIONS, camera_T_target_poses)
    result = cal.solve(max_iterations=30)

    assert result.has_converged
    assert result.iteration_results[-1].post_update_twist_errors_norm < 1e-7
    nominal_cal = SerialRobotKineCal(
        rtb.models.URDF.Panda(), ee_name="panda_link8"
    )
    for nominal_axis, camera_axis in zip(
        nominal_cal.joint_screw_axis, result.get_screw_axes()
    ):
        expected_camera_axis = camera_T_base.Ad() @ nominal_axis
        np.testing.assert_allclose(
            camera_axis, expected_camera_axis, atol=1e-7, rtol=1e-7
        )

    expected_terminal_pose = (
        camera_T_base @ nominal_cal.zero_conf_EE_pose @ ee_T_target
    )
    np.testing.assert_allclose(
        result.get_zero_conf_EE_pose().A,
        expected_terminal_pose.A,
        atol=1e-7,
        rtol=1e-7,
    )


def test_position_only_calibration_converges():
    robot_model = rtb.models.URDF.Panda()
    rng = np.random.default_rng(1234)
    configurations = rng.uniform(-1.5, 1.5, size=(12, robot_model.n))
    observed_positions = [robot_model.fkine(q).t.copy() for q in configurations]
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    cal.set_observations(configurations, observed_positions)

    result = cal.solve(max_iterations=5)

    assert result.has_converged
    assert result.termination_reason == "converged"
    iteration = result.iteration_results[-1]
    assert iteration.position_only
    assert iteration.OBS_SIZE == 3
    assert max(iteration.position_errors) < 1e-10
    assert np.all(np.isfinite(iteration.get_statistics()["joints_uncertainty"]))


def test_poe_robot_calibration_and_urdf_rejection():
    source_model = rtb.models.DH.Puma560()
    screw_axes, zero_pose = source_model.twists()
    poe_model = rtb.PoERobot(
        [rtb.PoELink(axis) for axis in screw_axes],
        zero_pose,
    )
    cal = SerialRobotKineCal(poe_model)
    rng = np.random.default_rng(5678)
    configurations = rng.uniform(-1.0, 1.0, size=(6, cal.N_JOINTS))
    observations = [
        cal.forward_kinematics(cal.N_JOINTS + 1, q)
        for q in configurations
    ]
    cal.set_observations(configurations, observations)

    result = cal.solve(max_iterations=3)

    assert result.has_converged
    with pytest.raises(TypeError, match="PoERobot"):
        cal.get_urdf_xyzrpy(result)


def test_public_api_surface():
    assert SerialRobotKineCal is not None
    assert CalibrationResult is not None


def test_default_end_effector_is_supported():
    robot_model = rtb.models.URDF.Panda()

    cal = SerialRobotKineCal(robot_model)

    assert cal.zero_conf_EE_pose is not None
    assert cal.ee_name == robot_model.links[-1].name


def test_selected_end_effector_uses_only_its_ancestor_chain():
    robot_model = rtb.models.URDF.Panda()

    cal = SerialRobotKineCal(robot_model, ee_name="panda_link4")

    assert [joint.name for joint in cal.joints] == [
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
    ]
    assert cal.N_JOINTS == 4


def test_dh_robot_is_supported():
    robot_model = rtb.models.DH.Puma560()

    cal = SerialRobotKineCal(robot_model)

    assert cal.N_JOINTS == robot_model.n
    assert cal.zero_conf_EE_pose is not None


def test_observation_validation():
    cal = SerialRobotKineCal(rtb.models.URDF.Panda(), ee_name="panda_link8")

    with pytest.raises(ValueError, match="At least one observation"):
        cal.set_observations([], [])

    with pytest.raises(ValueError, match="exactly 7 values"):
        cal.set_observations([[0.0] * 6], [np.zeros(3)])


def test_forward_kinematics_validates_joint_positions():
    cal = SerialRobotKineCal(rtb.models.URDF.Panda(), ee_name="panda_link8")

    with pytest.raises(ValueError, match="exactly 7 values"):
        cal.forward_kinematics(cal.N_JOINTS + 1, [0.0] * 6)

    with pytest.raises(ValueError, match="non-finite"):
        cal.forward_kinematics(cal.N_JOINTS + 1, [0.0] * 6 + [np.nan])


def test_solve_requires_observations():
    cal = SerialRobotKineCal(rtb.models.URDF.Panda(), ee_name="panda_link8")

    with pytest.raises(RuntimeError, match="set_observations"):
        cal.solve()


def test_step_size_scales_position_update():
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    cal.set_observations([[0.0] * 7], [np.zeros(3)])
    initial_position = cal.zero_conf_EE_pose.t.copy()
    correction = np.zeros(4 * cal.N_JOINTS + cal.OBS_SIZE)
    correction[-3:] = [1.0, 2.0, 3.0]

    cal.update_twist_definitions(correction, step_size=0.25)

    np.testing.assert_allclose(
        cal.zero_conf_EE_pose.t - initial_position,
        [0.25, 0.5, 0.75],
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("position_weight", 0.0, "position_weight"),
        ("orientation_weight", np.inf, "orientation_weight"),
        ("damping", -1.0, "damping"),
        ("min_step_size", 0.0, "min_step_size"),
    ],
)
def test_solve_validates_numerical_options(keyword, value, message):
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)

    with pytest.raises(ValueError, match=message):
        cal.solve(**{keyword: value})


def test_iteration_result_stores_a_snapshot():
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)
    result = cal.solve(max_iterations=1)
    stored_axis = result.iteration_results[0].joint_screw_definitions[0].copy()

    cal.joint_screw_axis[0][0] += 1.0

    np.testing.assert_array_equal(
        result.iteration_results[0].joint_screw_definitions[0],
        stored_axis,
    )


def test_result_accessors_return_copies():
    robot_model = rtb.models.URDF.Panda()
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)
    result = cal.solve(max_iterations=1)

    screw_axes = result.get_screw_axes()
    zero_pose = result.get_zero_conf_EE_pose()
    screw_axes[0][0] += 1.0
    zero_pose.t[0] += 1.0

    assert screw_axes[0][0] != result.get_screw_axes()[0][0]
    assert zero_pose.t[0] != result.get_zero_conf_EE_pose().t[0]


def test_reset_restores_nominal_model_and_keeps_observations():
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    initial_axes = [axis.copy() for axis in cal.joint_screw_axis]
    initial_pose = cal.zero_conf_EE_pose.A.copy()
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)
    correction = np.ones(4 * cal.N_JOINTS + cal.OBS_SIZE) * 1e-3
    cal.update_twist_definitions(correction)

    cal.reset()

    for restored_axis, initial_axis in zip(cal.joint_screw_axis, initial_axes):
        np.testing.assert_array_equal(restored_axis, initial_axis)
    np.testing.assert_array_equal(cal.zero_conf_EE_pose.A, initial_pose)
    assert len(cal._observations) == len(CONFIGURATIONS)
    assert cal._B_Matrix is None


def test_result_reports_max_iterations():
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)

    result = cal.solve(max_iterations=1)

    assert result.termination_reason == "max_iterations"


def test_iteration_exposes_solver_diagnostics_and_applied_correction():
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)

    result = cal.solve(max_iterations=1, step_size=0.25, damping=1e-8)
    iteration = result.iteration_results[0]

    np.testing.assert_allclose(
        iteration.applied_twist_corrections,
        0.25 * iteration.twist_corrections,
    )
    assert iteration.matrix_rank <= iteration.N_PARAMS
    assert iteration.singular_values.ndim == 1
    assert np.isfinite(iteration.condition_number)
    assert (
        iteration.post_update_twist_errors_norm
        <= iteration.pre_update_twist_errors_norm
    )


def test_solve_validates_backtracking_options():
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)

    with pytest.raises(TypeError, match="backtracking"):
        cal.solve(backtracking=1)

    with pytest.raises(ValueError, match="greater than step_size"):
        cal.solve(step_size=0.1, min_step_size=0.2)


def test_backtracking_reduces_an_excessive_step():
    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    observed_ee_poses = [robot_model.fkine(q) for q in CONFIGURATIONS]
    cal.set_observations(CONFIGURATIONS, observed_ee_poses)

    result = cal.solve(max_iterations=1, step_size=10.0)
    iteration = result.iteration_results[0]

    assert 0 < iteration.step_size < 10.0
    assert (
        iteration.post_update_twist_errors_norm
        <= iteration.pre_update_twist_errors_norm
    )
