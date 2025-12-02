# coding=utf-8
# -------------------------------------------------------------------
# 机器人手眼标定程序 (Eye-on-Hand, 相机安装在机械臂末端)
#
# 依赖库:
# 1. libpyauboi5 (遨博官方库)
# 2. robotcontrol.py (您提供的机器人控制封装)
# 3. camera_realsense.py (您自己的Realsense封装)
# 4. pip install numpy
# 5. pip install opencv-contrib-python (必须是contrib版, 包含aruco和calibrateHandEye)
# 6. pip install scipy (用于姿态转换)
# -------------------------------------------------------------------

import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation as R

# --- 关键修正 1: 导入和初始化 ---
try:
    # 确保 camera_realsense.py 文件在同一目录下
    from camera_realsense import RealSenseCamera
except ImportError:
    print("错误：无法导入 'camera_realsense.py'")
    print("请确保您已提供了 Realsense D435 的封装脚本。")
    exit()

try:
    # 修正导入：文件名是 robotcontrol, 不是 AUBOrobotcontrol
    from robotcontrol import Auboi5Robot, RobotErrorType
except ImportError:
    print("错误：无法导入 'robotcontrol.py'")
    print("请确保您提供的 'robotcontrol.py' 文件在同一目录下。")
    exit()

# --- 用户配置区域 ---
# 请根据您的实际情况修改以下参数

# 机器人IP地址
ROBOT_IP = "192.168.1.40"  # <<<--- 修改为您的 AUBO 机器人 IP

# ChArUco 标定板参数
SQUARES_X = 7              # 标定板 X 方向的方块数
SQUARES_Y = 5              # 标定板 Y 方向的方块数
SQUARE_LENGTH = 0.04       # 方块的边长 (单位: 米)
MARKER_LENGTH = 0.02       # ArUco 标记的边长 (单位: 米)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# --- 配置结束 ---


class HandEyeCalibrator:
    def __init__(self):
        # --- 关键修正 2: 严格遵循 robotcontrol.py 的初始化流程 ---
        print("正在初始化 AUBO 机器人库...")
        # 必须先调用静态的 initialize 方法
        if Auboi5Robot.initialize() != RobotErrorType.RobotError_SUCC:
            raise RuntimeError("无法初始化机器人库(libpyauboi5)。")

        print("正在连接 AUBO 机器人...")
        self.robot = Auboi5Robot()
        self.robot.create_context()
        if self.robot.connect(ROBOT_IP, 8899) != RobotErrorType.RobotError_SUCC:
            raise ConnectionError(f"无法连接到机器人 {ROBOT_IP}")
        
        # 必须上电才能获取姿态
        self.robot.robot_startup()
        print("机器人连接并上电成功。")

        # 初始化相机
        self.camera = RealSenseCamera()
        self.cam_matrix = np.array([
            [self.camera.intrinsics.fx, 0, self.camera.intrinsics.ppx],
            [0, self.camera.intrinsics.fy, self.camera.intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(self.camera.intrinsics.coeffs)

        # 创建 ChArUco 标定板
        self.board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)

        # 存储标定数据 (Eye-on-Hand)
        self.all_base_to_gripper_R = []  # 机器人基座 -> 末端 (旋转矩阵)
        self.all_base_to_gripper_t = []  # 机器人基座 -> 末端 (平移向量)
        self.all_cam_to_board_R = []     # 相机 -> 标定板 (旋转矩阵)
        self.all_cam_to_board_t = []     # 相机 -> 标定板 (平移向量)

    def capture_and_process(self):
        """
        (Eye-on-Hand 流程):
        1. 移动机器人(带着相机)到新位置。
        2. 获取机器人末端(Gripper)相对于基座(Base)的位姿。
        3. 拍摄固定标定板(Board)的照片。
        4. 计算标定板(Board)相对于相机(Cam)的位姿。
        """
        print("\n-------------------------------------------")
        input("请移动机器人(带着相机)到一个新的、能看清标定板的位置，然后按 Enter 键采集...")

        # 1. 获取机器人当前姿态 (Base -> Gripper)
        waypoint = self.robot.get_current_waypoint()
        if not waypoint:
            print("错误：无法获取机器人姿态。")
            return False
        
        pos_base_to_gripper = waypoint['pos'] # (x, y, z)
        ori_quat_base_to_gripper = waypoint['ori'] # (w, x, y, z)

        # 将四元数转换为旋转矩阵
        # 注意: scipy 使用 (x, y, z, w) 格式
        rot_matrix_base_to_gripper = R.from_quat([
            ori_quat_base_to_gripper[1], 
            ori_quat_base_to_gripper[2], 
            ori_quat_base_to_gripper[3], 
            ori_quat_base_to_gripper[0]
        ]).as_matrix()
        
        # 2. 获取并处理相机图像
        color_image, _ = self.camera.get_frames()
        if color_image is None:
            print("错误：无法获取相机图像。")
            return False

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR_GRAY)
        
        # 3. --- 关键修正 3: 使用 solvePnP/estimatePoseCharucoBoard 获取位姿 ---
        corners, ids, _, _ = self.charuco_detector.detectBoard(gray)

        if ids is not None and len(ids) > 6: # 至少需要6个点保证PnP稳定性
            # 获取检测到的角点的3D世界坐标
            obj_points = self.board.getChessboardCorners()[ids.flatten()]
            
            # 使用 solvePnP 计算标定板在相机坐标系下的位姿 (Cam -> Board)
            # (estimatePoseCharucoBoard 是更优的选择，但 solvePnP 也可以工作)
            retval, rvec, tvec = cv2.solvePnP(obj_points, corners, self.cam_matrix, self.dist_coeffs)
            
            if retval:
                print(f"成功检测到 {len(ids)} 个角点并解算位姿。")
                rot_matrix_cam_to_board, _ = cv2.Rodrigues(rvec)

                # 4. 存储数据
                self.all_base_to_gripper_R.append(rot_matrix_base_to_gripper)
                self.all_base_to_gripper_t.append(np.array(pos_base_to_gripper))
                self.all_cam_to_board_R.append(rot_matrix_cam_to_board)
                self.all_cam_to_board_t.append(tvec.flatten())
                
                # 在图像上绘制检测结果以供预览
                cv2.drawFrameAxes(color_image, self.cam_matrix, self.dist_coeffs, rvec, tvec, SQUARE_LENGTH * 0.5)
                cv2.imshow("Calibration View", color_image)
                cv2.waitKey(500)
                return True
        
        # 如果检测失败
        print("警告：未能在图像中检测到足够的角点，请调整机器人位置重试。")
        cv2.imshow("Calibration View", color_image)
        cv2.waitKey(500)
        return False

    def calibrate(self):
        """执行手眼标定计算 (Eye-on-Hand)"""
        print("\n-------------------------------------------")
        print(f"已收集 {len(self.all_base_to_gripper_R)} 组数据，开始计算...")

        if len(self.all_base_to_gripper_R) < 10:
            print(f"错误：数据点太少({len(self.all_base_to_gripper_R)})，至少需要10个有效数据点。")
            return

        # --- 关键修正 4: 使用 Eye-on-Hand (AX=XB) 的正确参数 ---
        # A: base_to_gripper (机械臂基座 -> 末端)
        # B: cam_to_board (相机 -> 标定板)
        # X: gripper_to_cam (机械臂末端 -> 相机) <--- 这是我们要的结果
        
        # 注意: cv2.calibrateHandEye 需要平移向量是 (N, 3, 1) 或 (N, 3)
        t_base2gripper = [t.reshape(3, 1) for t in self.all_base_to_gripper_t]
        t_cam2board = [t.reshape(3, 1) for t in self.all_cam_to_board_t]

        R_gripper2cam, t_gripper2cam = cv2.calibrateHandEye(
            self.all_base_to_gripper_R, t_base2gripper,
            self.all_cam_to_board_R, t_cam2board,
            method=cv2.CALIB_HAND_EYE_TSAI # TSAI 方法比较常用
        )

        # 组合成 4x4 变换矩阵
        gripper_to_cam_matrix = np.eye(4)
        gripper_to_cam_matrix[:3, :3] = R_gripper2cam
        gripper_to_cam_matrix[:3, 3] = t_gripper2cam.flatten()

        print("\n--- 标定成功! (Eye-on-Hand) ---")
        print("机械臂末端(Gripper)到相机(Cam)的变换矩阵 (gripper_to_cam_matrix):")
        print(np.round(gripper_to_cam_matrix, 5))

        # 保存结果
        np.save("hand_eye_gripper_to_cam_matrix.npy", gripper_to_cam_matrix)
        print("\n矩阵已保存到文件: hand_eye_gripper_to_cam_matrix.npy")

    def run(self):
        """运行标定主循环"""
        while True:
            self.capture_and_process()
            
            print(f"\n当前已采集 {len(self.all_base_to_gripper_R)} 个有效数据点。")
            cmd = input("继续采集请按 Enter, 输入 'c' 进行计算, 输入 'q' 退出: ").lower()

            if cmd == 'c':
                self.calibrate()
                break
            elif cmd == 'q':
                break
    
    def cleanup(self):
        print("\n正在关闭机器人和相机...")
        
        # --- 关键修正 5: 遵循 robotcontrol.py 的清理流程 ---
        if self.robot.connected:
            self.robot.robot_shutdown()
            self.robot.disconnect()
        
        # 必须最后调用静态的 uninitialize 方法
        Auboi5Robot.uninitialize()
        
        self.camera.stop()
        cv2.destroyAllWindows()
        print("清理完成。")


if __name__ == "__main__":
    calibrator = None
    try:
        calibrator = HandEyeCalibrator()
        calibrator.run()
    except Exception as e:
        print(f"\n--- 发生严重错误 ---")
        print(e)
        import traceback
        traceback.print_exc()
    finally:
        if calibrator:
            calibrator.cleanup()