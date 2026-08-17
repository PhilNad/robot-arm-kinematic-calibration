"""Generate a Markdown calibration report and plots from the minimal example."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import roboticstoolbox as rtb
from MinimalExample import (
    configurations,
    local_link_transforms,
    observed_ee_poses,
    pose_errors,
)

from robotkinecal import SerialRobotKineCal


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_report(output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    image_directory = output_directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)

    robot_model = rtb.models.URDF.Panda()
    cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    cal.set_observations(configurations, observed_ee_poses)
    result = cal.solve()
    screw_axes, zero_conf_ee_pose = cal.get_calibration(result)
    last_iteration = result.iteration_results[-1]

    nominal_cal = SerialRobotKineCal(robot_model, ee_name="panda_link8")
    nominal_screw_axes = nominal_cal.joint_screw_axis
    nominal_zero_pose = nominal_cal.zero_conf_EE_pose

    nominal_poses = [
        robot_model.fkine(q, end="panda_link8") for q in configurations
    ]
    calibrated_poses = [
        cal.forward_kinematics(cal.N_JOINTS + 1, q)
        for q in configurations
    ]
    nominal_position, nominal_orientation = pose_errors(
        nominal_poses, observed_ee_poses
    )
    calibrated_position, calibrated_orientation = pose_errors(
        calibrated_poses, observed_ee_poses
    )

    iterations = np.arange(1, result.nb_iterations_executed + 1)
    pre_errors = np.array(
        [item.pre_update_twist_errors_norm for item in result.iteration_results]
    )
    post_errors = np.array(
        [item.post_update_twist_errors_norm for item in result.iteration_results]
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.semilogy(iterations, pre_errors, "o-", label="Before update")
    axis.semilogy(iterations, post_errors, "s-", label="After update")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Twist error norm")
    axis.set_title("Calibration convergence")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    save_figure(figure, image_directory / "convergence.png")

    error_labels = ["Mean position\n[m]", "Max position\n[m]", "Mean orientation\n[rad]", "Max orientation\n[rad]"]
    nominal_errors = [
        nominal_position.mean(),
        nominal_position.max(),
        nominal_orientation.mean(),
        nominal_orientation.max(),
    ]
    calibrated_errors = [
        calibrated_position.mean(),
        calibrated_position.max(),
        max(calibrated_orientation.mean(), 1e-16),
        max(calibrated_orientation.max(), 1e-16),
    ]
    x_positions = np.arange(len(error_labels))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.36
    axis.bar(x_positions - width / 2, nominal_errors, width, label="Nominal")
    axis.bar(x_positions + width / 2, calibrated_errors, width, label="Calibrated")
    axis.set_yscale("log")
    axis.set_xticks(x_positions, error_labels)
    axis.set_ylabel("Error (log scale)")
    axis.set_title("Dataset error before and after calibration")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()
    save_figure(figure, image_directory / "error_comparison.png")

    axis_point_shifts = []
    axis_direction_changes = []
    screw_changes = []
    for nominal_axis, calibrated_axis in zip(nominal_screw_axes, screw_axes):
        nominal_direction = nominal_axis[3:] / np.linalg.norm(nominal_axis[3:])
        calibrated_direction = calibrated_axis[3:] / np.linalg.norm(
            calibrated_axis[3:]
        )
        nominal_point = np.cross(nominal_direction, nominal_axis[:3])
        calibrated_point = np.cross(calibrated_direction, calibrated_axis[:3])
        axis_point_shifts.append(1e3 * np.linalg.norm(calibrated_point - nominal_point))
        axis_direction_changes.append(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(nominal_direction, calibrated_direction), -1.0, 1.0
                    )
                )
            )
        )
        screw_changes.append(np.linalg.norm(calibrated_axis - nominal_axis))

    joint_labels = [f"J{index}" for index in range(1, cal.N_JOINTS + 1)]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(joint_labels, axis_point_shifts, color="#4472c4")
    axes[0].set_ylabel("Shift [mm]")
    axes[0].set_title("Joint-axis point shift")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(joint_labels, axis_direction_changes, color="#ed7d31")
    axes[1].set_ylabel("Change [deg]")
    axes[1].set_title("Joint-axis direction change")
    axes[1].grid(True, axis="y", alpha=0.3)
    save_figure(figure, image_directory / "joint_axis_changes.png")

    nominal_local = local_link_transforms(
        nominal_cal.joint_zero_conf_poses, nominal_zero_pose
    )
    calibrated_local = local_link_transforms(
        cal.joint_zero_conf_poses, zero_conf_ee_pose
    )
    urdf_definitions = cal.get_urdf_xyzrpy(result)
    local_translation_changes = []
    local_orientation_changes = []
    for nominal_transform, calibrated_transform in zip(
        nominal_local, calibrated_local
    ):
        difference = calibrated_transform @ nominal_transform.inv()
        local_translation_changes.append(1e3 * np.linalg.norm(difference.t))
        local_orientation_changes.append(np.degrees(difference.angvec()[0]))

    link_labels = [f"L{index}" for index in range(1, len(urdf_definitions) + 1)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(link_labels, local_translation_changes, color="#70ad47")
    axes[0].set_ylabel("Change [mm]")
    axes[0].set_title("URDF local translation change")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(link_labels, local_orientation_changes, color="#a64d79")
    axes[1].set_ylabel("Change [deg]")
    axes[1].set_title("URDF local orientation change")
    axes[1].grid(True, axis="y", alpha=0.3)
    save_figure(figure, image_directory / "urdf_transform_changes.png")

    zero_difference = zero_conf_ee_pose @ nominal_zero_pose.inv()
    joint_rows = "\n".join(
        f"| J{index} | {point_shift:.6f} | {direction_change:.6f} | {screw_change:.6e} |"
        for index, (point_shift, direction_change, screw_change) in enumerate(
            zip(axis_point_shifts, axis_direction_changes, screw_changes), start=1
        )
    )
    urdf_rows = "\n".join(
        f"| {definition['name']} | {translation:.6f} | {orientation:.6f} |"
        for definition, translation, orientation in zip(
            urdf_definitions, local_translation_changes, local_orientation_changes
        )
    )
    screw_rows = "\n".join(
        f"| J{index} | `[{', '.join(f'{value:.10e}' for value in axis)}]` |"
        for index, axis in enumerate(screw_axes, start=1)
    )
    zero_matrix = "\n".join(
        "    " + np.array2string(row, precision=10, suppress_small=False)
        for row in zero_conf_ee_pose.A
    )

    report = f"""# Panda 机械臂运动学标定报告

## 1. 标定概况

| 项目 | 结果 |
|---|---:|
| 机器人模型 | Franka Panda |
| 标定末端 | `panda_link8` |
| 关节数 | {cal.N_JOINTS} |
| 观测数 | {len(configurations)} |
| 观测类型 | 完整 SE(3) 位姿 |
| 待估参数 | {last_iteration.N_PARAMS} |
| 终止原因 | `{result.termination_reason}` |
| 迭代次数 | {result.nb_iterations_executed} |
| 最终 twist 误差范数 | {last_iteration.post_update_twist_errors_norm:.6e} |
| 回归矩阵秩 | {last_iteration.matrix_rank}/{last_iteration.N_PARAMS} |
| 条件数 | {last_iteration.condition_number:.6e} |

## 2. 收敛过程

![Calibration convergence](images/convergence.png)

每轮更新均降低了 twist 残差，最终误差达到数值精度量级。本数据为无噪声仿真数据，不能将该误差水平直接视为真实机器人的可达精度。

## 3. 标定前后末端误差

![Error comparison](images/error_comparison.png)

| 指标 | 名义模型 | 标定后模型 |
|---|---:|---:|
| 平均位置误差 [m] | {nominal_position.mean():.6e} | {calibrated_position.mean():.6e} |
| 最大位置误差 [m] | {nominal_position.max():.6e} | {calibrated_position.max():.6e} |
| 平均姿态误差 [rad] | {nominal_orientation.mean():.6e} | {calibrated_orientation.mean():.6e} |
| 最大姿态误差 [rad] | {nominal_orientation.max():.6e} | {calibrated_orientation.max():.6e} |

## 4. 关节轴参数变化

![Joint-axis changes](images/joint_axis_changes.png)

| 关节 | 轴点位移 [mm] | 轴方向变化 [deg] | `||ΔS||` |
|---|---:|---:|---:|
{joint_rows}

## 5. 零位末端姿态变化

| 指标 | 数值 |
|---|---:|
| 平移变化 [mm] | {1e3*np.linalg.norm(zero_difference.t):.6f} |
| 姿态变化 [deg] | {np.degrees(zero_difference.angvec()[0]):.6f} |

标定后的零位末端齐次变换：

```text
{zero_matrix}
```

## 6. 标定后的 POE 螺旋轴

| 关节 | `[v_x, v_y, v_z, w_x, w_y, w_z]` |
|---|---|
{screw_rows}

## 7. URDF 局部变换变化

![URDF transform changes](images/urdf_transform_changes.png)

图中 `L1`～`L8` 按下表顺序对应相邻 link 变换。

| 变换 | 平移变化 [mm] | 姿态变化 [deg] |
|---|---:|---:|
{urdf_rows}

## 8. 结论与注意事项

- 求解在 {result.nb_iterations_executed} 轮后正常收敛，回归矩阵满秩。
- 当前条件数为 {last_iteration.condition_number:.3f}，未表现出明显的数值病态。
- 标定后模型能够重现当前无噪声仿真观测。
- 本报告使用同一数据集进行求解和误差统计，因此展示的是拟合误差，不是独立泛化误差。
- 真实机器人应额外划分验证集，并检查相机外参、末端标定板外参、单位、时间同步、机械柔性和回差。
- 将结果写入 URDF 前应备份名义模型，并在独立数据上验证标定后的正运动学。
"""
    report_path = output_directory / "minimal_example_calibration_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/minimal_example"),
        help="Directory for the Markdown report and generated images.",
    )
    arguments = parser.parse_args()
    report_path = generate_report(arguments.output_directory)
    print(f"Generated calibration report: {report_path}")


if __name__ == "__main__":
    main()
