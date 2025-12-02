# coding=utf-8
import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation as R
from AUBOrobotcontrol import Auboi5Robot, RobotErrorType
from camera_realsense import RealSenseCamera

# --- 用户配置区域 ---
# 请根据您的实际情况修改以下参数

# 机器人IP地址
ROBOT_IP = "192.168.1.40"  # <<<--- 修改为您的 AUBO 机器人 IP

# --- 标定模式选择 ---
# 'BOARD': 使用 `generate_aruco_board.py` 生成的标定板 (推荐, 更精确)
# 'SINGLE': 使用单个 ArUco 标记 (精度较低)
CALIBRATION_MODE = 'BOARD' # <<<--- 选择您的标定模式

# ArUco 字典 (必须与 generate_aruco_board.py 保持一致)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# --- 'BOARD' 模式配置 (如果您使用标定板) ---
# 打印并**精确测量**后，填入以下值 (单位：米)
MARKERS_X = 5              # X 方向的标记数量
MARKERS_Y = 4              # Y 方向的标记数量
BOARD_MARKER_SIZE_METERS = 0.04  # <<<--- 打印出标定板后，请**精确测量**并修改
BOARD_MARKER_SEP_METERS = 0.01 # <<<--- 打印出标定板后，请**精确测量**并修改

# --- 'SINGLE' 模式配置 (如果您使用单个标记) ---
SINGLE_MARKER_SIZE_METERS = 0.05  # <<<--- 打印出标记后，请**精确测量**并修改

# --- 配置结束 ---

class EyeOnHandCalibrator:
    def __init__(self):
        print("正在初始化 AUBO 机器人...")
        self.robot = Auboi5Robot()
        self.robot.create_context()
        if self.robot.connect(ROBOT_IP, 8899) != RobotErrorType.RobotError_SUCC:
            raise ConnectionError(f"无法连接到机器人 {ROBOT_IP}")
        self.robot.robot_startup()
        print("机器人连接成功。")

        self.camera = RealSenseCamera()
        self.cam_matrix = np.array([
            [self.camera.intrinsics.fx, 0, self.camera.intrinsics.ppx],
            [0, self.camera.intrinsics.fy, self.camera.intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(self.camera.intrinsics.coeffs)
        self.aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

        # 存储标定数据
        self.R_base_to_end = []  # 机器人末端 -> 基座的旋转
        self.t_base_to_end = []  # 机器人末端 -> 基座的平移
        self.R_cam_to_target = [] # 标定物 -> 相机的旋转
        self.t_cam_to_target = [] # 标定物 -> 相机的平移
        
        if CALIBRATION_MODE == 'BOARD':
            print("使用 [标定板] 模式进行标定。")
            self.board = cv2.aruco.GridBoard(
                size=(MARKERS_X, MARKERS_Y),
                markerLength=BOARD_MARKER_SIZE_METERS,
                markerSeparation=BOARD_MARKER_SEP_METERS,
                dictionary=ARUCO_DICT
            )
        elif CALIBRATION_MODE == 'SINGLE':
            print("使用 [单个标记] 模式进行标定。")
            # 为 estimatePoseSingleMarkers 准备 objectPoints
            self.single_marker_obj_points = np.array([
                [-SINGLE_MARKER_SIZE_METERS / 2, SINGLE_MARKER_SIZE_METERS / 2, 0],
                [ SINGLE_MARKER_SIZE_METERS / 2, SINGLE_MARKER_SIZE_METERS / 2, 0],
                [ SINGLE_MARKER_SIZE_METERS / 2, -SINGLE_MARKER_SIZE_METERS / 2, 0],
                [-SINGLE_MARKER_SIZE_METERS / 2, -SINGLE_MARKER_SIZE_METERS / 2, 0]
            ], dtype=np.float32)
        else:
            raise ValueError("CALIBRATION_MODE 必须是 'BOARD' 或 'SINGLE'")

    def get_robot_pose_matrix(self):
        """获取机器人末端在基座坐标系下的 4x4 变换矩阵"""
        waypoint = self.robot.get_current_waypoint()
        if not waypoint:
            return None
        
        pos = waypoint['pos']
        ori_quat = waypoint['ori'] # (w, x, y, z)
        
        r = R.from_quat([ori_quat[1], ori_quat[2], ori_quat[3], ori_quat[0]])
        rot_matrix = r.as_matrix()
        
        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = pos
        return matrix

    def capture_and_process(self):
        """捕获一帧图像，检测标记，并记录机器人姿态"""
        print("\n-------------------------------------------")
        input(f"请手动遥控机器人，从新的姿态对准 {CALIBRATION_MODE}，然后按 Enter 键采集...")

        color_image, depth_image = self.camera.get_frames()
        if color_image is None:
            print("错误：无法获取相机图像。")
            return False

        (corners, ids, rejected) = self.aruco_detector.detectMarkers(color_image)
        
        R_cam_to_target = None
        t_cam_to_target = None
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            if CALIBRATION_MODE == 'BOARD':
                # 使用标定板模式
                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners, ids, self.board, self.cam_matrix, self.dist_coeffs, None, None
                )
                if retval > 0: # 至少有一个标记被用于姿态估计
                    R_cam_to_target, _ = cv2.Rodrigues(rvec)
                    t_cam_to_target = tvec.flatten()
                    cv2.drawFrameAxes(color_image, self.cam_matrix, self.dist_coeffs, rvec, tvec, BOARD_MARKER_SIZE_METERS)
                    print(f"成功检测到标定板 (使用了 {retval} 个标记)。")
                else:
                    print("检测到标记，但无法估计标定板姿态。")

            elif CALIBRATION_MODE == 'SINGLE':
                # 使用单个标记模式
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, SINGLE_MARKER_SIZE_METERS, self.cam_matrix, self.dist_coeffs
                )
                # 只使用第一个检测到的标记
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                R_cam_to_target, _ = cv2.Rodrigues(rvec)
                t_cam_to_target = tvec.flatten()
                cv2.drawFrameAxes(color_image, self.cam_matrix, self.dist_coeffs, rvec, tvec, SINGLE_MARKER_SIZE_METERS * 0.5)
                print(f"成功检测到单个标记 (ID: {ids[0][0]})。")
            
            cv2.imshow("Calibration View", color_image)
            cv2.waitKey(500)

            # 如果姿态估计成功
            if R_cam_to_target is not None:
                T_base_to_end = self.get_robot_pose_matrix()
                if T_base_to_end is None:
                    print("错误：无法获取机器人姿态。")
                    return False

                # 存储数据
                self.R_base_to_end.append(T_base_to_end[:3, :3])
                self.t_base_to_end.append(T_base_to_end[:3, 3])
                self.R_cam_to_target.append(R_cam_to_target)
                self.t_cam_to_target.append(t_cam_to_target)
                
                print(f"成功采集数据点 {len(self.R_base_to_end)}")
                return True
            else:
                return False
                
        else:
            print("警告：未能在图像中检测到任何 ArUco 标记，请调整机器人位置重试。")
            cv2.imshow("Calibration View", color_image)
            cv2.waitKey(500)
            return False

    def calibrate(self):
        """执行手眼标定 (Eye-on-Hand) 计算"""
        print("\n-------------------------------------------")
        print(f"已收集 {len(self.R_base_to_end)} 组数据，开始计算...")

        if len(self.R_base_to_end) < 10:
            print(f"错误：数据点太少 ({len(self.R_base_to_end)})，至少需要10个有效数据点。")
            return

        # 求解 AX=XB
        # A: T_base_to_end
        # B: T_cam_to_target (标定物)
        # X: T_end_to_cam (我们要求解的)
        
        # OpenCV 的 calibrateHandEye 需要 T_gripper -> T_base 和 T_target -> T_cam
        # 我们的数据是 T_base -> T_gripper 和 T_cam -> T_target
        # 我们需要进行转换：
        R_gripper2base = [R.T for R in self.R_base_to_end]
        t_gripper2base = [-R.T @ t for R, t in zip(self.R_base_to_end, self.t_base_to_end)]
        
        R_target2cam = [R.T for R in self.R_cam_to_target]
        t_target2cam = [-R.T @ t for R, t in zip(self.R_cam_to_target, self.t_cam_to_target)]
        
        R_cam_to_end, t_cam_to_end = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK  # 推荐使用 PARK 或 TSAI
        )

        # 我们得到的是 T_cam_to_end, 我们需要 T_end_to_cam
        R_end_to_cam = R_cam_to_end.T
        t_end_to_cam = -R_cam_to_end.T @ t_cam_to_end

        # 组合成 4x4 变换矩阵
        end_to_cam_matrix = np.eye(4)
        end_to_cam_matrix[:3, :3] = R_end_to_cam
        end_to_cam_matrix[:3, 3] = t_end_to_cam.flatten()

        print("\n--- 标定成功! ---")
        print("机器人末端(end)到相机(cam)的变换矩阵 (end_to_cam_matrix):")
        print(np.round(end_to_cam_matrix, 4))

        # 保存结果
        np.save("end_to_cam_matrix.npy", end_to_cam_matrix)
        print("\n矩阵已保存到文件: end_to_cam_matrix.npy")
        print("您现在可以关闭此脚本，并运行 'run_robot_eye_on_hand.py'。")

    def run(self):
        """运行标定主循环"""
        while True:
            # 尝试采集数据，如果失败 (e.g., 未检测到标记)，则循环重试
            self.capture_and_process()
            
            print(f"\n当前已采集 {len(self.R_base_to_end)} 个有效数据点。")
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
    calibrator = EyeOnHandCalibrator()
    try:
        calibrator.run()
    finally:
        calibrator.cleanup()
