# Robot Arm Kinematic Calibration
This simple Python library implements the method described in [Local POE model for robot kinematic calibration](https://doi.org/10.1016/S0094-114X(01)00048-9) to determine the kinematic parameters of a robot arm from the nominal robot model and a set of end-effector poses.

By virtue of relying on [the robotics toolbox](https://petercorke.github.io/robotics-toolbox-python/intro.html) (RTB), this library supports all robot models defined in RTB, such as:
- Puma 560
- Franka robots
- Kinova Gen 3
- UR 3/5/10 robots
- Kuka LBR
- Fetch
- etc.

It is also possible to define a custom robot model with RTB and use this library to calibrate it.

Since the method used to produce the calibration data varies greatly between different robot setups, this library does not provide any tools to collect the calibration data. Instead, it expects the user to provide the calibration data in the form of a list of joint configurations and measured end-effector poses.

For instance, end-effector poses can be obtained from a motion capture system with markers attached to the robot end-effector, from a laser tracker, from a camera attached to the robot end-effector and observing a known pattern, etc. In all cases, the resulting dataset should contain the joint configurations and the corresponding end-effector poses in SE(3).

To facilitate the management of the end-effector poses, the [with-respect-to](https://github.com/PhilNad/with-respect-to) library is used and the poses are assumed to be named *ee-0*, *ee-1*, ..., *ee-n*.

## Usage Examples
See [Examples/FrankaSimulation.py](Examples/FrankaSimulation.py) for an example on how to use this library to calibrate a Franka robot model using a simulated dataset, and [Examples/FrankaReal.py](Examples/FrankaReal.py) for an example on how to use this library to calibrate a real Franka robot using a dataset collected with a camera mounted on the end-effector of the robot (eye-in-hand).

### Minimal Example
```python
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np
import with_respect_to as WRT
from RobotKineCal.RobotKineCal import SerialRobotKineCal

#Joint configurations reached during data collection
configurations = [
    [0.11991574115863357, 0.49875949370033534, 0.710468885366869, 0.6302279697573877, 0.008085732035263304, 0.8373924410577887, 0.2743523221875973],
    [0.5747237008889052, 0.6750625204144085, 0.5578587064736473, 0.27663671141076185, 0.4055425202634496, 0.2562207606138225, 0.45095468295156205],
    [0.637120647672808, 0.4827699697361507, 0.7632007588615727, 0.9841267694363484, 0.6737738777747908, 0.9022722899806046, 0.17450161273521614],
    [0.17289957872676187, 0.3723378524812525, 0.7316570573211971, 0.549646660002465, 0.6754763125942058, 0.24581496295993133, 0.6909359923802936],
    [0.4766422613074568, 0.29019696585714183, 0.5000063082603127, 0.8761362129754574, 0.09279932695679771, 0.9922019408230374, 0.5246116534083605],
    [0.9878972331113948, 0.05005315444969349, 0.3454379850104754, 0.7644959423815235, 0.7082691704745409, 0.9185527251235208, 0.9591809950062664],
    [0.32864754244123096, 0.39254811070663953, 0.6960705613229867, 0.9554418923549504, 0.31818437564494073, 0.20367233258466566, 0.6154963350700174],
    [0.7113564638087916, 0.3783743233244643, 0.8752740544983237, 0.5155750883579142, 0.7889345138237629, 0.6613872517616122, 0.27764692619089315],
    [0.6144188594831896, 0.5706222325063981, 0.5423533313367412, 0.18435684086863646, 0.1645988155678333, 0.6349417431511627, 0.8311540126069151],
    [0.21246358814432875, 0.2741355327831245, 0.6943487715258115, 0.09489688382251737, 0.211083139993946, 0.4987954190139666, 0.5187312640040359],
    [0.5538035722034983, 0.22256191636640787, 0.6681278168914814, 0.7491796788515966, 0.8077078608544332, 0.27765157732547285, 0.3089077546570097],
    [0.6097760692583295, 0.7350728481791406, 0.14468891814385765, 0.7637365445731804, 0.8512900338916467, 0.117209465614729, 0.865309041908516],
    [0.23485040501024124, 0.8648859482378104, 0.7217691838330261, 0.8216100040321475, 0.4942192076288773, 0.5464968854628888, 0.5025988429660981],
    [0.7847314786653901, 0.37204538572241075, 0.34963592857772896, 0.9090619129905065, 0.7519433127853599, 0.5497647901017554, 0.5949492334003703],
    [0.06486321744536672, 0.2402494510992944, 0.9577574239310385, 0.41648826000128525, 0.6479766130275036, 0.9337076180616392, 0.390382595696696],
    [0.7749152642533456, 0.5254385465076856, 0.9412996543703323, 0.4845588942067488, 0.7231127649747242, 0.4996609438027224, 0.25065152885265574],
    [0.9830328694552237, 0.11512811456884076, 0.6802549815331893, 0.549951810711465, 0.47684729807831105, 0.4058036421565966, 0.17794590173366387],
    [0.18116863369675573, 0.19047486857150553, 0.6465446964333841, 0.8693345905511151, 0.8831798115066752, 0.4502682350342284, 0.7964624512848507],
    [0.8557302347770908, 0.7216316038306939, 0.5815153632919512, 0.9457434569627436, 0.4333761498204871, 0.4006547506599958, 0.531829484595734],
    [0.8966526493917544, 0.6466987905361836, 0.7201619661419558, 0.7056891618038801, 0.29890530608130306, 0.7511774648084326, 0.8686018503585171]]
#End-effector poses observed/measured by an external system
observed_ee_poses = [
    SE3([[-0.9990862631542131, -0.04271819465784187, 0.0013396349220970727, -0.0011607855830684955], [0.04273248974482775, -0.9989938892836435, 0.013606744436029035, 0.0196330454411938], [0.0007570315435681608, 0.013651557387856967, 0.9999065265734234, 0.08531900967093747], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.9985593644792383, 0.05088562775259403, 0.017024937563592253, 0.006309172598380061], [-0.050543320190718714, -0.9985222660589098, 0.019966395986225644, 0.014606588729703855], [0.018015781829225568, 0.019077134816442656, 0.9996556879907588, 0.08644854583535287], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.07852189267218572, 0.996233840033643, -0.03677564877742116, -0.01109925132793828], [-0.9969122243993787, -0.07844720192111726, 0.003471794036358112, -0.00692676515542119], [0.0005737719592244849, 0.03693470566513206, 0.9993175162595563, 0.08149256413764494], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.9566902214971863, 0.2910692207026845, 0.0047464566990736935, -0.005165016309235315], [-0.29040331009453185, -0.9553814890404037, 0.0539641351740271, 0.01478239757879653], [0.02024197563982521, 0.05024857369591272, 0.9985315935230707, 0.09081216686054806], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.8093568757492535, 0.5872792289877354, -0.00667494396811936, -0.010278685386671772], [-0.587245481653915, -0.809028274300459, 0.02481925984960822, 0.01884909698555157], [0.00917561738894433, 0.024007469295858962, 0.9996696701728729, 0.0836735919560034], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[0.019048879088305333, 0.99911472013999, -0.03750888168246812, -0.017373931418799886], [-0.9992291186410236, 0.020312263238540244, 0.033594351043167395, 0.003949138100054132], [0.0343265009192963, 0.03684003205371166, 0.9987314470731953, 0.08310752190598227], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.9314084532523962, 0.36397566783555124, -8.020999251702746e-05, -0.004685097752713135], [-0.3636845103793536, -0.9306545274544908, 0.04019611223225254, 0.014621395455895152], [0.014555759001443827, 0.03746816985285817, 0.9991918064754981, 0.08957185224354303], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.4812461931962341, 0.8761412621726856, -0.027903230146102038, -0.008729371116260687], [-0.8765847146985718, -0.4809598709483892, 0.016638524397717187, -0.00023761113246352765], [0.0011573637963945191, 0.03246677156359348, 0.999472145311354, 0.08305395147798122], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.921192577775066, 0.3888313793277475, 0.014642168630733723, 0.0034819643815528546], [-0.38835976949576934, -0.9211033354196007, 0.027300822626425282, 0.01968673544163361], [0.024102366882160206, 0.019462885936266626, 0.9995200208008391, 0.08219919026425176], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.8033060253295543, -0.5950489817463883, 0.024822147204062544, 0.009963953000977034], [0.5953126065807396, -0.8034826955493676, 0.004296323866722982, 0.014238836276486443], [0.017387642602697144, 0.018228200001887025, 0.9996826509494958, 0.08924105632000158], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.9030762343389528, 0.4293582202008553, -0.010238833826702947, -0.006689814526402868], [-0.4294358658865453, -0.9023833282687228, 0.035904957219527384, 0.008466252468044649], [0.006176735582033529, 0.036821836029950615, 0.999302757090629, 0.09068026551091267], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.5917922029999372, 0.8058558472644164, -0.019451012781615595, -0.004693328798916499], [-0.8053000750922454, -0.5899713551630228, 0.05852853273002878, 0.012640942612257714], [0.03569001996223746, 0.05030063137631296, 0.9980962222943435, 0.08934887317614483], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.4905163354361269, 0.8707537589708406, -0.034374640484872765, -0.010016740983291184], [-0.8713627221026697, -0.48959823062441915, 0.03194650370954243, 0.006370467509355047], [0.010987775031315742, 0.045623062233800755, 0.9988982956198654, 0.08366450143534676], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.38924206833296593, 0.9206761899681828, -0.029085485478926067, -0.01129237040159432], [-0.9208983116773134, -0.3882312828767432, 0.034968136149245696, 0.005804146719823478], [0.02090243501959986, 0.04039584411233919, 0.9989650964817052, 0.08637732560533404], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.8666571339050654, 0.49872933434888406, -0.013208456055755284, -0.015642889753003552], [-0.49880233683053504, -0.8656414463673486, 0.04314064328961197, 0.014706467736876984], [0.010081717306827265, 0.04397655501473008, 0.9989816923173226, 0.08595415813019075], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.490976136994303, 0.870801908580998, -0.02542575296510316, -0.003777724455914913], [-0.871161559845864, -0.49091023790383226, 0.009201900245230654, 0.00016278416529803017], [-0.0044687301408640545, 0.026667852048745177, 0.9996343612131561, 0.08215908773675072], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.970850720280775, 0.23948783875557592, 0.009718745725806548, -0.0001517108318873691], [-0.23938710823464604, -0.9708676484387982, 0.010479581383401266, 0.009561131554838936], [0.011945348105162368, 0.007847566699347469, 0.9998978569611726, 0.08666925810280439], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.6387620524931722, 0.769053495028154, -0.02323278028328653, -0.016815343070165904], [-0.7684332566178707, -0.6361483325453223, 0.06946674832939777, 0.013157591936897504], [0.03864415115335996, 0.06222556375628608, 0.9973136962845986, 0.09110642074715021], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[-0.23636419578747575, 0.9711882476639546, -0.030419640845292466, -0.005734722828834634], [-0.9716532206261924, -0.23609451005882315, 0.012222977003683353, 0.003275197863510298], [0.0046889014159093596, 0.03244641612722494, 0.9994624776768817, 0.08187995618644239], [0.0, 0.0, 0.0, 1.0]]),
    SE3([[0.17821486644325876, 0.9831117205848419, -0.041602959357758144, -0.00964437599486322], [-0.9839205954926931, 0.1785505576930099, 0.004467674206968081, 5.2257210194451686e-05], [0.01182045447163225, 0.040137802583436875, 0.9991242333463126, 0.07667738258637652], [0.0, 0.0, 0.0, 1.0]])
]
#Load the model of the robot
robot_model = rtb.models.URDF.Panda()

#Create the calibration object
ee_name = 'panda_link8'
cal = SerialRobotKineCal(robot_model, ee_name, verbose=True)

db = WRT.DbConnector(1)
for i in range(len(observed_ee_poses)):
    db.In('kine-cal').Set(f'ee-{i}').Wrt('world').Ei('world').As(observed_ee_poses[i].A)

#Set the data
cal.set_data('kine-cal', configurations)

#Solve the calibration problem
result = cal.solve()
```
produces
```
Iteration result:
        Norm of twist errors: 12.4316
        Avg. Position error: 0.9714
        Max. Position error: 1.0427
        Avg. Orientation error: 2.4339
        Max. Orientation error: 3.1195
        Joints uncertainty: [4.57e+13 4.46e+13 9.44e+15 2.64e+13 8.14e+15 2.38e+15 8.17e+15 7.76e+15]
Iteration result:
        Norm of twist errors: 10.0412
        Avg. Position error: 1.4229
        Max. Position error: 1.6114
        Avg. Orientation error: 1.5931
        Max. Orientation error: 1.9990
        Joints uncertainty: [5.78e+12 1.64e+12 3.47e+12 7.39e+12 9.71e+11 9.02e+12 2.02e+13 1.17e+12]
Iteration result:
        Norm of twist errors: 5.4568
        Avg. Position error: 1.1206
        Max. Position error: 1.3692
        Avg. Orientation error: 0.4360
        Max. Orientation error: 0.6104
        Joints uncertainty: [4.26e+12 2.18e+11 1.17e+12 1.08e+12 2.06e+12 3.77e+11 4.56e+12 3.97e+11]
Iteration result:
        Norm of twist errors: 1.5278
        Avg. Position error: 0.3373
        Max. Position error: 0.3523
        Avg. Orientation error: 0.0504
        Max. Orientation error: 0.0953
        Joints uncertainty: [6.50e+09 9.55e+09 1.93e+09 6.37e+09 4.06e+09 3.51e+09 1.08e+10 5.68e+09]
Iteration result:
        Norm of twist errors: 0.1195
        Avg. Position error: 0.0232
        Max. Position error: 0.0295
        Avg. Orientation error: 0.0107
        Max. Orientation error: 0.0258
        Joints uncertainty: [1.75e+09 1.44e+11 1.35e+10 1.23e+10 9.22e+08 1.94e+11 1.14e+10 1.26e+11]
Iteration result:
        Norm of twist errors: 0.0384
        Avg. Position error: 0.0032
        Max. Position error: 0.0087
        Avg. Orientation error: 0.0063
        Max. Orientation error: 0.0155
        Joints uncertainty: [1.65e+09 2.14e+09 2.68e+09 5.40e+08 7.71e+09 2.22e+09 5.58e+09 2.55e+09]
Iteration result:
        Norm of twist errors: 0.0384
        Avg. Position error: 0.0032
        Max. Position error: 0.0087
        Avg. Orientation error: 0.0062
        Max. Orientation error: 0.0155
        Joints uncertainty: [3.57e+09 3.46e+09 1.22e+09 2.58e+09 1.59e+09 2.11e+09 1.64e+09 3.39e+09]
The kinematic calibration has converged.
```

### Real Robot Example
This library was used to find the kinematic parameters of a real Franka Research 3 robot arm using a dataset collected with a RealSense D405 camera mounted on the robot end-effector. A calibration board was precisely mounted on the robot table, as shown in the following picture:
![FR3_Calibration_Setup](https://github.com/user-attachments/assets/ac481fa2-d099-4bc2-a510-666ff133a48e)

The dataset was collected by moving the robot to about 115 different poses, each having the camera (approximately) pointing at the centre of the calibration board. Assuming knowledge of the transform between the end-effector frame and the camera frame (a reasonable assumption since the camera mount was accurately 3D printed), end-effector poses were obtained from the camera images. The resulting dataset is [here](https://github.com/PhilNad/robot-arm-kinematic-calibration/blob/main/Examples/calib_data.pickle) and the code used to calibrate the robot is [here](https://github.com/PhilNad/robot-arm-kinematic-calibration/blob/main/Examples/FrankaReal.py).

Using `SerialRobotKineCal.print_urdf_joint_definitions()`, joint definitions that can directly be used in the [fr3.urdf.xacro](https://github.com/frankaemika/franka_description/blob/main/robots/fr3/fr3.urdf.xacro) URDF file from the [franka_description](https://github.com/frankaemika/franka_description) package were obtained:
```
Definition of the joints in RPY-XYZ format for use in a URDF:
        Joint panda_link0-panda_link1
                XYZ: [4.66940862e-03 1.33320085e-04 3.32135211e-01]
                RPY: [-7.87186237e-05  9.64086780e-03  9.43333437e-06]
        Joint panda_link1-panda_link2
                XYZ: [ 9.89596905e-05  3.21868027e-04 -8.41845003e-04]
                RPY: [-1.56837522e+00 -1.81292251e-04  8.95620323e-06]
        Joint panda_link2-panda_link3
                XYZ: [-0.0006072  -0.31487821  0.00031979]
                RPY: [ 1.56950118e+00 -1.32523377e-03  1.14910419e-05]
        Joint panda_link3-panda_link4
                XYZ: [ 0.08285224  0.00057119 -0.00112239]
                RPY: [1.56817607e+00 5.13115656e-04 1.27735192e-03]
        Joint panda_link4-panda_link5
                XYZ: [-0.0823229   0.38269415 -0.00057144]
                RPY: [-1.57377631e+00  1.09236370e-02 -5.66605662e-04]
        Joint panda_link5-panda_link6
                XYZ: [ 0.00130735  0.00031509 -0.0013069 ]
                RPY: [1.56590035 0.00205777 0.01091949]
        Joint panda_link6-panda_link7
                XYZ: [ 0.0871673   0.00141573 -0.00030438]
                RPY: [ 1.57225125  0.01596719 -0.00214542]
        Joint panda_link7-panda_link8
                XYZ: [-0.00139496  0.00218057  0.10560957]
                RPY: [ 0.01627551  0.00927677 -0.01589182]
```

After replacing the nominal kinematic parameters in the URDF file with the calibrated ones, any ROS node should be able to benefit from the improved accuracy of the robot model. This includes the [MoveIt](https://github.com/moveit/moveit) motion planner, whose collision avoidance capabilities depend on accurate kinematic parameters. In our experiments, the robot was a lot less likely to collide with the environment after calibration.

## Technical Details
The method described in [Local POE model for robot kinematic calibration](https://doi.org/10.1016/S0094-114X(01)00048-9) and used in this library is based on twists and on the product of exponentials (POE) formula for forward kinematics. Through an iterative least-squares optimization scheme, the twists representing perturbations to the pose of each link relative to the previous one are determined. Since the perturbations are relative to the previous link, the method is deemd *local*. This formulation greatly simplifies the implementation of the calibration algorithm. However, as pointed out in [this paper](https://doi.org/10.1109/TRO.2016.2593042), an equivalent formulation exists where less parameters are required (avoiding the introduction of redundant parameters and possibly slightly improving convergence speed). In practice, very few iterations are required to converge to a solution and the over-parametrization of the problem is not an issue. 
