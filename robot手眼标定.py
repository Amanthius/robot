# coding=utf-8
import numpy as np
import cv2
import time
from AUBOrobotcontrol import Auboi5Robot, RobotErrorType
from camera_realsense import RealSenseCamera

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
        # 初始化机器人
        print("正在初始化 AUBO 机器人...")
        self.robot = Auboi5Robot()
        self.robot.create_context()
        if self.robot.connect(ROBOT_IP, 8899) != RobotErrorType.RobotError_SUCC:
            raise ConnectionError(f"无法连接到机器人 {ROBOT_IP}")
        self.robot.robot_startup()
        print("机器人连接成功。")

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

        # 存储标定数据
        self.all_charuco_corners = []
        self.all_charuco_ids = []
        self.robot_poses_R = []  # 旋转矩阵
        self.robot_poses_t = []  # 平移向量

    def capture_and_process(self):
        """捕获一帧图像，检测标定板，并记录机器人姿态"""
        print("\n-------------------------------------------")
        input("请将机器人探针精确对准一个新的标定板角点，然后按 Enter 键采集...")

        # 1. 获取机器人当前姿态
        waypoint = self.robot.get_current_waypoint()
        if not waypoint:
            print("错误：无法获取机器人姿态。")
            return False
        
        pos = waypoint['pos']
        ori_quat = waypoint['ori'] # (w, x, y, z)

        # 将四元数转换为旋转矩阵
        # 注意: scipy 使用 (x, y, z, w) 格式
        from scipy.spatial.transform import Rotation as R
        rot_matrix = R.from_quat([ori_quat[1], ori_quat[2], ori_quat[3], ori_quat[0]]).as_matrix()
        
        # 2. 获取并处理相机图像
        color_image, _ = self.camera.get_frames()
        if color_image is None:
            print("错误：无法获取相机图像。")
            return False

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.charuco_detector.detectBoard(gray)

        if ids is not None and len(ids) > 4:
            print(f"成功检测到 {len(ids)} 个角点。")
            self.all_charuco_corners.append(corners)
            self.all_charuco_ids.append(ids)
            self.robot_poses_R.append(rot_matrix)
            self.robot_poses_t.append(np.array(pos))
            
            # 在图像上绘制检测结果以供预览
            cv2.aruco.drawDetectedCorners(color_image, corners, ids)
            cv2.imshow("Calibration View", color_image)
            cv2.waitKey(500)
            return True
        else:
            print("警告：未能在图像中检测到足够的角点，请调整机器人位置重试。")
            cv2.imshow("Calibration View", color_image)
            cv2.waitKey(500)
            return False

    def calibrate(self):
        """执行手眼标定计算"""
        print("\n-------------------------------------------")
        print(f"已收集 {len(self.robot_poses_R)} 组数据，开始计算...")

        if len(self.robot_poses_R) < 10:
            print("错误：数据点太少，至少需要10个有效数据点。")
            return

        # 估计每个视图中标定板的姿态
        obj_points = [self.board.getChessboardCorners()[ids.flatten()] for ids in self.all_charuco_ids]
        
        ret, cam_to_board_R_vecs, cam_to_board_t_vecs = cv2.calibrateCamera(
            obj_points, self.all_charuco_corners, (self.camera.width, self.camera.height), 
            self.cam_matrix, self.dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        # 将旋转向量转换为旋转矩阵
        cam_to_board_R_mats = [cv2.Rodrigues(rvec)[0] for rvec in cam_to_board_R_vecs]

        # 执行手眼标定 (Eye to Hand, camera is static)
        # 我们需要计算 base -> gripper 和 board -> camera 的变换
        # OpenCV 需要 gripper -> base 和 camera -> board
        
        # gripper -> base 的变换
        R_gripper2base = [np.linalg.inv(R) for R in self.robot_poses_R]
        t_gripper2base = [-np.linalg.inv(R) @ t for R, t in zip(self.robot_poses_R, self.robot_poses_t)]

        R_cam2board, t_cam2board = cam_to_board_R_mats, cam_to_board_t_vecs
        
        # 使用 Tsai-Lenz (AX=XB) 方法
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_cam2board, t_cam2board, 
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # 组合成 4x4 变换矩阵
        cam_to_base_matrix = np.eye(4)
        cam_to_base_matrix[:3, :3] = R_cam2base
        cam_to_base_matrix[:3, 3] = t_cam2base.flatten()

        print("\n--- 标定成功! ---")
        print("相机到机器人基座的变换矩阵 (cam_to_base_matrix):")
        print(np.round(cam_to_base_matrix, 4))

        # 保存结果
        np.save("hand_eye_matrix.npy", cam_to_base_matrix)
        print("\n矩阵已保存到文件: hand_eye_matrix.npy")

    def run(self):
        """运行标定主循环"""
        while True:
            self.capture_and_process()
            
            print(f"\n当前已采集 {len(self.robot_poses_R)} 个有效数据点。")
            cmd = input("继续采集请按 Enter, 输入 'c' 进行计算, 输入 'q' 退出: ").lower()

            if cmd == 'c':
                self.calibrate()
                break
            elif cmd == 'q':
                break
    
    def cleanup(self):
        print("\n正在关闭机器人和相机...")
        self.robot.robot_shutdown()
        self.robot.disconnect()
        self.camera.stop()
        cv2.destroyAllWindows()
        print("清理完成。")


if __name__ == "__main__":
    calibrator = HandEyeCalibrator()
    try:
        calibrator.run()
    finally:
        calibrator.cleanup()
