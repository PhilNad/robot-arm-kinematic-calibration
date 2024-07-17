from spatialmath import SE3
import roboticstoolbox as rtb
import with_respect_to as WRT
from RobotKineCal.RobotKineCal import SerialRobotKineCal
import pickle

TEMPORARY_DATABASE = 1
db = WRT.DbConnector(TEMPORARY_DATABASE)

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

#Load the data
fin = open('calib_data.pickle', 'rb')
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
result = cal.solve()

#Print the URDF joint definitions
cal.print_urdf_joint_definitions(result)