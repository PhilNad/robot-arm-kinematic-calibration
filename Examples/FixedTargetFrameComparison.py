"""Compare calibration results before and after a fixed target-frame offset."""

import numpy as np
import roboticstoolbox as rtb
from MinimalExample import configurations, observed_ee_poses
from spatialmath import SE3

from robotkinecal import SerialRobotKineCal


def solve(observations):
    calibrator = SerialRobotKineCal(
        rtb.models.URDF.Panda(),
        ee_name="panda_link8",
    )
    calibrator.set_observations(configurations, observations)
    result = calibrator.solve(max_iterations=20)
    return calibrator, result


if __name__ == "__main__":
    # A deterministic rigid transform from panda_link8 to a hypothetical
    # board/marker/tracker frame. Target observations are therefore
    # Base_T_Target = Base_T_EE * EE_T_Target.
    ee_T_target = SE3.RPY([0.20, -0.15, 0.10], order="zyx") * SE3(
        0.04, -0.03, 0.08
    )
    observed_target_poses = [
        base_T_ee @ ee_T_target for base_T_ee in observed_ee_poses
    ]

    ee_cal, ee_result = solve(observed_ee_poses)
    target_cal, target_result = solve(observed_target_poses)

    axis_differences = np.array(
        [
            np.linalg.norm(target_axis - ee_axis)
            for ee_axis, target_axis in zip(
                ee_result.get_screw_axes(), target_result.get_screw_axes()
            )
        ]
    )
    expected_target_zero_pose = (
        ee_result.get_zero_conf_EE_pose() @ ee_T_target
    )
    terminal_pose_error = np.linalg.norm(
        (
            target_result.get_zero_conf_EE_pose()
            @ expected_target_zero_pose.inv()
        ).log(twist=True)
    )

    print("Fixed target-frame comparison")
    print("-----------------------------")
    print(f"EE solve:     {ee_result.termination_reason}")
    print(f"Target solve: {target_result.termination_reason}")
    print("\nJoint screw-axis differences")
    for index, difference in enumerate(axis_differences, start=1):
        print(f"Joint {index}: {difference:.6e}")
    print(f"Maximum joint-axis difference: {axis_differences.max():.6e}")
    print(
        "Target terminal-pose consistency error: "
        f"{terminal_pose_error:.6e}"
    )
    print(
        "Original final residual: "
        f"{ee_result.iteration_results[-1].post_update_twist_errors_norm:.6e}"
    )
    print(
        "Target final residual:   "
        f"{target_result.iteration_results[-1].post_update_twist_errors_norm:.6e}"
    )

    print(
        "\nInterpretation: a fixed EE-to-target transform is absorbed by the "
        "terminal zero pose, while the calibrated joint axes remain unchanged."
    )
    print(
        "Do not export the target calibration as panda_link8 URDF geometry "
        "unless EE_T_Target is removed first or the target is represented as "
        "an explicit fixed child link."
    )
