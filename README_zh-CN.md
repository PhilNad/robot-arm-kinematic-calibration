# 机器人机械臂运动学标定

[English README](README.md) | 简体中文

相关实施文档：

- [眼在手外数据用于手眼与运动学标定：理论汇总](docs/eye_to_hand_kinematic_calibration_theory_zh-CN.md)：数学推导、复合模型、可辨识性和 URDF 边界；
- [现场操作手册（SOP）](docs/field_calibration_sop_zh-CN.md)：按阶段执行、记录、验收和签字；
- [部署与实施流程](docs/calibration_workflow_zh-CN.md)：流程说明、数据规范和技术背景。

`robotkinecal` 是一个用于串联机械臂运动学标定的 Python 库。它实现了论文 [《POE-based robot kinematic calibration using axis configuration space and the adjoint error model》](https://doi.org/10.1109/TRO.2016.2593042) 中提出的方法，根据机器人的名义模型、关节配置以及实测末端执行器位姿，估计更准确的机器人运动学参数。

## 主要功能

- 标定仅包含旋转关节的串联机器人；
- 支持末端执行器三维位置观测或完整 `SE(3)` 位姿观测；
- 支持 Robotics Toolbox for Python 内置的多种机器人模型；
- 支持通过 URDF 加载自定义机器人；
- 使用每个关节 4 个参数的最小参数化，有利于提高求解效率和收敛速度；
- 可以输出适合写入 URDF 的关节 XYZ/RPY 定义；
- 可以输出适用于指数积（POE）正运动学公式的关节螺旋轴。

本项目基于 [Robotics Toolbox for Python](https://petercorke.github.io/robotics-toolbox-python/intro.html)，因此可以使用其中提供的 Puma 560、Franka、Kinova Jaco、UR3/UR5/UR10、KUKA LBR 和 Fetch 等机器人模型。

## 安装

从 PyPI 安装：

```bash
pip install robotkinecal
```

使用 `uv` 安装：

```bash
uv add robotkinecal
```

项目要求 Python 3.10～3.12。安装完成后可以这样导入：

```python
from robotkinecal import SerialRobotKineCal
```

如果要从源码运行测试或参与开发：

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## 标定数据

该库不负责采集标定数据。不同机器人系统的数据采集方法差异较大，调用者需要自行提供以下两组一一对应的数据：

1. 机器人到达的关节配置，每组包含所有待标定关节的角度，单位为弧度；
2. 对应配置下测得的末端执行器位置或完整位姿。

末端观测可以来自：

- 安装在末端执行器上的运动捕捉标记；
- 激光跟踪仪；
- 眼在手上的相机和已知标定板；
- 外部相机或其他能够测量末端位姿的设备。

完整位姿观测应使用 `spatialmath.SE3` 对象。仅位置观测应使用形状为 `(3,)` 的 NumPy 数组，例如：

```python
import numpy as np
from spatialmath import SE3

pose_observation = SE3.Rt(R=np.eye(3), t=[0.4, 0.1, 0.5])
position_observation = np.array([0.4, 0.1, 0.5])
```

完整位姿通常能提供更多约束；仅使用位置时，一般需要更多、分布更丰富的观测数据。此外，仅位置观测无法辨识末端执行器自身的方向。

## 快速开始

下面演示使用 Robotics Toolbox 中的 Franka Panda 模型进行标定：

```python
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from robotkinecal import SerialRobotKineCal

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

# 外部测量系统观测到的末端执行器位姿。
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

robot_model = rtb.models.URDF.Panda()

cal = SerialRobotKineCal(
    robot_model,
    ee_name="panda_link8",
    verbose=True,
)

cal.set_observations(configurations, observed_ee_poses)
result = cal.solve(max_iterations=100, step_size=1.0)

print("是否收敛：", result.has_converged)
print("是否发散：", result.is_diverging)
print("迭代次数：", result.nb_iterations_executed)

# 获取并打印实际标定参数。
screw_axes, zero_conf_ee_pose = cal.get_calibration(result)
print("终止原因：", result.termination_reason)
print("最终 twist 误差：", result.iteration_results[-1].post_update_twist_errors_norm)
cal.print_screw_axes(result)
cal.print_urdf_joint_definitions(result)
```

可直接运行的完整数据示例见 [Examples/MinimalExample.py](Examples/MinimalExample.py)：

```bash
python Examples/MinimalExample.py
```

脚本完成后会继续输出以下结果摘要：

```text
Calibration summary
-------------------
Termination reason: converged
Iterations: 4
Final twist error norm: 8.832049e-09
Regressor rank: 34/34
Regressor condition number: 3.064929e+01

Dataset error comparison
------------------------
Mean position error [m]: 5.656636e-02 -> 2.656028e-09
Max position error [m]:  1.169498e-01 -> 3.769683e-09
Mean orientation error [rad]: 6.321331e-02 -> 0.000000e+00
Max orientation error [rad]:  8.841659e-02 -> 0.000000e+00
```

随后还会完整打印 7 个标定后关节螺旋轴、标定后的零位末端齐次变换，以及全部 URDF 关节 XYZ/RPY。上述参数才是实际标定结果，迭代过程中的误差、秩和条件数属于求解诊断信息。

运行以下命令可以生成包含收敛曲线、误差对比、关节轴变化和 URDF 变化图片的 Markdown 标定报告：

```bash
python Examples/GenerateCalibrationReport.py
```

已生成的示例报告见 [reports/minimal_example/minimal_example_calibration_report.md](reports/minimal_example/minimal_example_calibration_report.md)。

## 使用自定义 URDF

可以用 Robotics Toolbox 加载自定义 URDF，然后将模型交给标定器：

```python
import roboticstoolbox as rtb

from robotkinecal import SerialRobotKineCal

robot_model = rtb.ERobot.URDF("Examples/gen3.urdf")
ee_name = robot_model.ee_links[0].parent.name

cal = SerialRobotKineCal(robot_model, ee_name=ee_name, verbose=True)
cal.set_observations(configurations, observed_ee_poses)
result = cal.solve()
```

完整示例见 [Examples/FromURDF.py](Examples/FromURDF.py)。

## 获取标定结果

### 获取 POE 参数

```python
screw_axes, zero_conf_ee_pose = cal.get_calibration(result)

for index, screw_axis in enumerate(screw_axes, start=1):
    print(f"关节 {index}: {screw_axis}")

print("零位末端执行器姿态：")
print(zero_conf_ee_pose)
```

也可以直接打印：

```python
cal.print_screw_axes(result)
```

根据标定结果创建 POE 机器人模型：

```python
calibrated_robot = rtb.PoERobot(
    [rtb.PoELink(axis) for axis in screw_axes],
    zero_conf_ee_pose,
)
```

### 获取 URDF 关节定义

对于 `ERobot` 或 `DHRobot`，可以获得相邻关节坐标系之间的 XYZ/RPY 参数：

```python
joint_definitions = cal.get_urdf_xyzrpy(result)

for joint in joint_definitions:
    print(joint["name"])
    print("  XYZ:", joint["xyz"])
    print("  RPY:", joint["rpy"])
```

也可以使用内置的格式化输出：

```python
cal.print_urdf_joint_definitions(result)
```

`PoERobot` 本身不描述各关节坐标系的局部位置，因此不能从它生成 URDF 关节定义。

## 如何理解迭代输出

启用 `verbose=True` 后，每轮会输出以下指标：

- `Norm of twist errors`：所有末端 twist 误差组成向量的范数；
- `Avg./Max. Position error`：末端位置误差的平均值和最大值，通常以米为单位；
- `Avg./Max. Orientation error`：末端方向误差的平均值和最大值，单位为弧度；
- `Joints uncertainty`：依据当前线性化问题和残差得到的关节参数不确定度估计。

对于无噪声仿真数据，误差可能降低到浮点数精度附近。这种结果只说明算法能够还原生成观测数据的模型，不能代表真实机械臂也能达到相同精度。

真实数据中的相机测量误差、手眼标定误差、关节编码器误差、机械柔性、负载和温度变化都会影响最终精度。建议保留一组未参与求解的数据，用来独立验证标定前后的末端误差。

## 项目示例

- [MinimalExample.py](Examples/MinimalExample.py)：最小完整示例；
- [FrankaSimulation.py](Examples/FrankaSimulation.py)：构造带运动学扰动的 Franka 仿真模型并验证标定效果；
- [FrankaReal.py](Examples/FrankaReal.py)：使用眼在手相机采集的真实 Franka 数据；
- [FromURDF.py](Examples/FromURDF.py)：从自定义 Kinova Gen3 URDF 创建并标定模型。

真实 Franka 示例使用安装在末端执行器上的 RealSense D405 相机观测标定板，共采集约 115 个姿态。求解后，可以把生成的关节参数写入 `franka_description` 的运动学配置，使 ROS 和 MoveIt 等组件使用更准确的机器人模型。

示例数据位于 [Examples/calib_data.pickle](Examples/calib_data.pickle)。请注意，反序列化 Pickle 文件可能执行任意代码，因此只应加载可信来源的文件。

## 常见问题

### 眼在手外的相机观测必须先做手眼标定吗？

不一定。固定相机观察刚性安装在末端的 Board、Marker 或 Tracker 时：

```text
Camera_T_Target(q)
= Camera_T_Base · Base_T_EE(q) · EE_T_Target
```

等式左右两侧的固定变换可以被复合 POE 模型吸收。因此，如果目标只是得到“关节角到固定相机坐标系下 Target 位姿”的映射，可以直接把 `Camera_T_Target` 作为完整位姿观测，不必提前单独完成手眼标定。

但这种方式不能分别辨识 `Camera_T_Base`、`EE_T_Target` 和机器人 Base/EE 坐标系下的本体几何。求解后的关节螺旋轴表达在 Camera 坐标系中，末端零位姿态对应 Target，而不一定对应传入的 `ee_name`。因此该复合结果不能直接回写成原机器人 URDF 几何参数。

如果目标是恢复机器人 Base/EE 物理坐标系下的参数或修改原 URDF，则仍需使用已知外参把观测转换为 `Base_T_EE`，或者在 URDF 中把 Target 建模为明确的固定子 link。数值对比和自动回归示例见 [FixedTargetFrameComparison.py](Examples/FixedTargetFrameComparison.py)。

### 如何调整含噪或病态的标定问题？

`solve()` 支持分别设置位置、姿态权重以及可选的 Tikhonov 阻尼，默认值仍与原有算法一致：

```python
result = cal.solve(
    step_size=0.5,
    position_weight=1.0,
    orientation_weight=0.5,
    damping=1e-8,
    backtracking=True,
)
```

每轮结果提供 `matrix_rank`、`condition_number` 和 `singular_values`，用于判断数据的可观测性和数值条件；`twist_corrections` 表示最小二乘原始解，`applied_twist_corrections` 表示经过 `step_size` 缩放后实际应用的修正量。

默认启用自动回溯。如果某次更新导致残差增大，求解器会恢复更新前模型并反复将步长减半。每轮会保存最终接受的 `step_size`、`pre_update_twist_errors_norm` 和 `post_update_twist_errors_norm`。如果在最小步长前仍找不到下降更新，求解会恢复原模型，并将 `termination_reason` 设置为 `"line_search_failed"`。

`solve()` 会更新标定器内部的运动学模型。如果需要从同一个名义模型独立比较不同求解参数，可在两次实验之间调用 `cal.reset()`；已经加载的观测数据会被保留。

### 标定不收敛怎么办？

可以依次检查以下内容：

1. 确认使用了正确的机器人模型、基坐标系和末端坐标系；
2. 确认每个末端观测与对应的关节角严格同步；
3. 增加观测数量；
4. 提高关节配置的多样性，避免所有样本集中在狭小工作区域；
5. 检查位姿变换方向，避免把 `base → tool` 与 `tool → base` 混用；
6. 检查长度单位是否统一为米，角度是否使用弧度；
7. 检查相机外参、手眼标定和标定板在机器人基座下的位姿；
8. 使用独立验证集判断问题来自过拟合还是数据系统误差。

可以修改 [Examples/FrankaSimulation.py](Examples/FrankaSimulation.py) 中的 `N_OBSERVATIONS`，观察样本数量对收敛效果的影响。

### 需要多少组采样数据？

仓库没有固定样本数要求。7 自由度机械臂使用完整位姿时，方程计数下限仅为 6 组，但该下限不保证满秩、抗噪声或泛化能力。工程上建议从 20～30 个覆盖充分的完整位姿开始，检查回归矩阵秩、条件数和独立验证误差，再每次补采 10～20 个互补姿态。高质量动捕/激光跟踪通常可在 20～40 组内工作，稳定工业视觉约 30～60 组，常规标定板视觉约 50～100 组，高噪声或遮挡场景约 80～150 组。另保留 15～30 组独立验证数据。

样本较少会产生两类不同后果：低于约束数量下限，或姿态激励不足导致回归矩阵秩亏时，部分参数不可辨识，通常无法得到唯一结果；样本刚好满秩时，求解器可能正常收敛，但结果对单帧噪声和数据划分非常敏感，表现为参数波动大、验证误差高。姿态充分且噪声近似独立时，增加样本主要改善精度和稳定性，其随机误差通常只近似按 `1 / sqrt(N)` 下降；重复采集大量相似姿态不会等比例提升效果，系统误差也不会因增加样本自动消失。

仅位置观测约束更少，通常需要更多数据。详细的计算依据、分级建议和停止准则见[部署、采集与验证流程](docs/calibration_workflow_zh-CN.md#6-姿态规划与数据量)。

### 为什么提示参数不可辨识？

这表示堆叠后的回归矩阵没有满列秩，部分运动学参数无法由现有数据唯一确定。常见原因包括：

- 观测数量不足；
- 机器人姿态变化不够丰富；
- 某些关节在采集过程中几乎没有运动；
- 仅测量位置时，测量点恰好位于最后一个旋转关节的轴线上。

通常应增加覆盖范围更广的采样姿态。仅位置观测时，还应让被测点适当偏离最后一个关节轴线。

### 支持移动关节吗？

当前不支持。该实现只接受旋转关节；模型包含移动关节时会抛出异常。

## 算法原理

指数积公式把串联机器人的正运动学表示为：

```text
T(q) = exp(S₁q₁) exp(S₂q₂) ... exp(Sₙqₙ) M
```

其中：

- `Sᵢ` 是第 `i` 个关节在空间坐标系下的螺旋轴；
- `qᵢ` 是对应关节角；
- `M` 是所有关节角为零时的末端执行器位姿。

算法比较模型计算的末端位姿与实际观测，通过伴随误差模型将末端误差线性化。每轮迭代执行以下步骤：

1. 根据当前螺旋轴计算所有观测配置的末端位姿；
2. 计算预测值和观测值之间的位置或位姿误差；
3. 构造每个关节的最小参数基矩阵 `B` 和观测回归矩阵 `A`；
4. 堆叠全部观测并求解最小二乘问题；
5. 更新各关节螺旋轴以及零位末端姿态；
6. 重复以上步骤，直到误差收敛、开始发散或达到最大迭代次数。

与直接优化每个关节的 6 维 twist 相比，该方法为每个旋转关节只使用 4 个独立参数，避免不必要的过参数化。

## 使用限制

- 目前仅支持旋转关节；
- 假设待标定部分是从机器人基座到指定末端的串联链；
- 标定结果高度依赖观测精度、坐标系定义和采样姿态的可观测性；
- 该库只估计运动学几何参数，不补偿柔性、回差、热变形或负载引起的动态误差；
- 在修改实际机器人 URDF 或 ROS 配置前，应使用独立数据验证结果并保留原始模型。

## 许可证

本项目使用 [MIT License](LICENSE)。
