# coding=utf-8
import time
import cv2
import numpy as np
import open3d as o3d
import requests
import base64
import re
from openai import OpenAI
from graspnet.graspnet import GraspBaseline
from scipy.spatial.transform import Rotation as R

from AUBOrobotcontrol import Auboi5Robot, RobotErrorType, RobotUserIoName, RobotIOType
from camera_realsense import RealSenseCamera

# --- 用户配置 ---
ROBOT_IP = "192.168.1.40" # <<<--- 确认这是您的 AUBO 机器人 IP

# !!! 新增：配置您的“拍照姿态”
# 请手动遥控机器人，让手上的相机能看清桌面上的物体
# 然后在示教器上读取这6个关节的角度（弧度），填入下方
SCAN_POSE_JOINTS = [0, -0.5, -1.57, 0.5, -1.57, 0] # <<<--- 示例姿态，请务必修改
# --- 配置结束 ---

class RealRobotEyeOnHandGrasp:
    def __init__(self, robot_ip=ROBOT_IP):
        print("正在加载 GraspNet 模型...")
        self.graspNet = GraspBaseline()
        print("GraspNet 模型加载完毕。")
        
        self.camera = RealSenseCamera()
        
        print(f"正在连接机器人 at {robot_ip}...")
        self.robot = Auboi5Robot()
        self.robot.create_context()
        if self.robot.connect(robot_ip, 8899) != RobotErrorType.RobotError_SUCC:
            raise ConnectionError(f"无法连接到机器人 {robot_ip}")
        self.robot.robot_startup()
        self.robot.init_profile()
        print("机器人连接并启动成功。")
        
        # 加载“眼在手上”的标定矩阵
        try:
            print("正在加载 '眼在手上' 标定矩阵 (end_to_cam_matrix.npy)...")
            self.T_end_to_cam = np.load('end_to_cam_matrix.npy')
            print("手眼标定矩阵加载成功。")
        except FileNotFoundError:
            print("\n!!! 错误: 未找到 end_to_cam_matrix.npy 文件。")
            print("请先运行 calibrate_eye_on_hand.py 以生成标定文件。\n")
            raise

        try:
            self.llm_client = OpenAI(api_key=open('keys.txt').readline().strip(), base_url="https://api.deepseek.com")
            print("LLM 客户端初始化成功。")
        except FileNotFoundError:
             print("\n!!! 警告: 未找到 keys.txt 文件。LLM 功能将不可用。\n")
             self.llm_client = None

    def get_world_point_cloud(self):
        """
        核心函数：移动到拍照点，获取点云，并将其转换到世界（基）坐标系
        """
        print(f"正在移动到拍照姿态: {SCAN_POSE_JOINTS}")
        self.robot.move_joint(SCAN_POSE_JOINTS)
        time.sleep(2) # 等待机器人稳定

        print("正在捕获图像和点云...")
        color_image, depth_image = self.camera.get_frames()
        if color_image is None:
            return None, None
        
        # 1. 获取相机坐标系下的点云
        pcd_cam = self.camera.get_point_cloud(color_image, depth_image)
        
        # 2. 获取当前机器人末端在基座标系下的姿态
        waypoint = self.robot.get_current_waypoint()
        pos = waypoint['pos']
        ori_quat = waypoint['ori'] # (w, x, y, z)
        r = R.from_quat([ori_quat[1], ori_quat[2], ori_quat[3], ori_quat[0]])
        
        T_base_to_end = np.eye(4)
        T_base_to_end[:3, :3] = r.as_matrix()
        T_base_to_end[:3, 3] = pos
        
        # 3. 核心转换：P_base = T_base_to_end * T_end_to_cam * P_cam
        #    (我们使用 Open3D 的 transform 方法)
        pcd_world = pcd_cam.transform(T_base_to_end @ self.T_end_to_cam)
        
        print("点云已成功转换到世界坐标系。")
        return pcd_world, color_image

    def detect_by_text(self, class_text, color_image):
        """使用 YOLO-World 服务进行开放词汇检测"""
        tmp_image_path = 'tmp.jpg'
        cv2.imwrite(tmp_image_path, color_image)

        with open(tmp_image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        data = {'image': image_base64, 'classes': [class_text]}
        try:
            response = requests.post('http://127.0.0.1:5000/detect', json=data, timeout=5)
            if response.status_code == 200:
                return response.json().get('detections', [])
            else:
                print(f"目标检测服务出错: {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"无法连接到目标检测服务 (http://127.0.0.1:5000/detect)。")
            return []

    def get_grasp_poses(self, pcd_world):
        """在世界坐标系点云中获取抓取姿态"""
        print("正在运行 GraspNet 推理...")
        # 因为点云已经是世界坐标系，GraspNet 的输出也将是世界坐标系
        gg = self.graspNet.run(pcd_world, vis=False)
        gg.nms()
        gg.sort_by_score()
        
        grasp_poses_world = {
            'scores': [g.score for g in gg],
            'matrices': [g.to_matrix() for g in gg] # 这些矩阵已经是世界坐标系！
        }
        return grasp_poses_world

    def execute_grasp(self, grasp_world_matrix):
        """执行单次抓取动作 (姿态已在世界坐标系)"""
        print("\n--- 开始执行抓取 ---")
        
        pos = grasp_world_matrix[:3, 3]
        rot_matrix = grasp_world_matrix[:3, :3]
        
        if pos[2] < 0.02: # 安全检查
            print(f"安全警告：目标抓取点 Z={pos[2]:.3f} 过低，取消抓取。")
            return

        r = R.from_matrix(rot_matrix)
        rpy_rad = r.as_euler('xyz', degrees=False)

        pre_grasp_pos = pos + np.array([0, 0, 0.15])  # 在抓取点上方 15cm
        
        try:
            print(f"正在打开夹爪...")
            self.open_gripper()
            time.sleep(1)

            print(f"正在移动到预抓取位置: {np.round(pre_grasp_pos, 3)}")
            self.robot.move_to_target_in_cartesian(list(pre_grasp_pos), list(np.rad2deg(rpy_rad)))
            time.sleep(2)

            print(f"正在下降到抓取位置: {np.round(pos, 3)}")
            self.robot.move_to_target_in_cartesian(list(pos), list(np.rad2deg(rpy_rad)))
            time.sleep(2)

            print("正在闭合夹爪...")
            self.close_gripper()
            time.sleep(1)

            print("正在抬升物体...")
            self.robot.move_to_target_in_cartesian(list(pre_grasp_pos), list(np.rad2deg(rpy_rad)))
            time.sleep(2)
            
            # 抓取完毕后，回到拍照位置
            print("抓取完成，正在返回拍照位置...")
            self.robot.move_joint(SCAN_POSE_JOINTS)
            
            print("--- 抓取流程结束 ---")

        except Exception as e:
            print(f"机械臂移动时发生错误: {e}")
            self.robot.move_stop()

    def open_gripper(self):
        """
        打开夹爪的函数 (使用末端工具I/O)。
        请根据您的夹爪电气接线和控制逻辑修改此函数。
        """
        print("正在通过末端I/O打开夹爪...")
        self.robot.set_tool_io_status("T_DI/O_00", 1)
        self.robot.set_tool_io_status("T_DI/O_01", 0)

    def close_gripper(self):
        """
        闭合夹爪的函数 (使用末端工具I/O)。
        请根据您的夹爪电气接线和控制逻辑修改此函数。
        """
        print("正在通过末端I/O闭合夹爪...")
        self.robot.set_tool_io_status("T_DI/O_00", 0)
        self.robot.set_tool_io_status("T_DI/O_01", 1)

    def plan_from_llm(self, task):
        """从大语言模型获取执行计划"""
        if not self.llm_client: return None
        template = """你是一个机器人，你拥有的技能API如下：
        1. get_grasp_by_name(name_text): 输入类别文本（注意是英文，要简短），返回检测候选抓取的List
        2. execute_grasp(grasp): 输入候选抓取，然后执行抓取
        现在需要你根据你所拥有的技能API，编写python代码完成给你的任务，只输出plan函数，不要输出其他代码以为的内容。你的任务是“KKKK”。
        """
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-coder",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": template.replace('KKKK', task)},
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用 LLM API 时出错: {e}")
            return None

    def cleanup(self):
        """程序退出时清理资源"""
        print("正在关闭系统...")
        self.robot.robot_shutdown()
        self.robot.disconnect()
        self.camera.stop()
        print("系统已关闭。")

# --- 全局 API 函数，供 LLM 生成的代码调用 ---
real_robot_system = None

def get_grasp_by_name(name_text):
    print(f"\nAPI 调用: get_grasp_by_name('{name_text}')")
    
    # 1. 移动相机, 拍照, 并获取世界坐标系下的点云
    pcd_world, color_image = real_robot_system.get_world_point_cloud()
    if pcd_world is None or pcd_world.is_empty():
        print("未能获取有效点云。")
        return []

    # 2. 获取所有可能的抓取姿态 (已在世界坐标系)
    grasp_poses_world = real_robot_system.get_grasp_poses(pcd_world)
    if not grasp_poses_world['matrices']:
        print("未能检测到任何抓取姿态。")
        return []

    # 3. 获取目标物体的 2D 检测框
    detections = real_robot_system.detect_by_text(name_text, color_image)
    if not detections:
        print(f"未能检测到物体: '{name_text}'")
        return []
    
    # 4. 筛选抓取姿态 (在世界坐标系中)
    x1, y1, x2, y2, _, _ = detections[0]
    
    # 我们需要将世界坐标系中的抓取点，投影回 2D 图像上
    intrinsics = real_robot_system.camera.o3d_intrinsics
    cam_matrix = intrinsics.intrinsic_matrix
    
    # 获取拍照时的 T_world_to_cam 变换
    waypoint = real_robot_system.robot.get_current_waypoint()
    pos = waypoint['pos']
    ori_quat = waypoint['ori'] # (w, x, y, z)
    r = R.from_quat([ori_quat[1], ori_quat[2], ori_quat[3], ori_quat[0]])
    T_base_to_end = np.eye(4)
    T_base_to_end[:3, :3] = r.as_matrix()
    T_base_to_end[:3, 3] = pos
    
    T_world_to_cam = np.linalg.inv(T_base_to_end @ real_robot_system.T_end_to_cam)

    valid_grasps = []
    for score, T_world_grasp in zip(grasp_poses_world['scores'], grasp_poses_world['matrices']):
        # 抓取点在世界坐标系的位置
        grasp_point_world = T_world_grasp[:3, 3]
        
        # 将抓取点转换到相机坐标系
        grasp_point_cam_h = T_world_to_cam @ np.append(grasp_point_world, 1)
        grasp_point_cam = grasp_point_cam_h[:3] / grasp_point_cam_h[3]
        
        # 投影到像素坐标
        if grasp_point_cam[2] > 0: # 必须在相机前方
            u = int(cam_matrix[0, 0] * grasp_point_cam[0] / grasp_point_cam[2] + cam_matrix[0, 2])
            v = int(cam_matrix[1, 1] * grasp_point_cam[1] / grasp_point_cam[2] + cam_matrix[1, 2])
            
            if x1 < u < x2 and y1 < v < y2:
                valid_grasps.append({'matrix': T_world_grasp, 'score': score})

    if not valid_grasps:
        print("没有位于检测框内的有效抓取。")
        return []

    best_grasp = sorted(valid_grasps, key=lambda x: x['score'], reverse=True)[0]
    print(f"找到 {len(valid_grasps)} 个有效抓取, 已选择最佳抓取。")
    return [best_grasp['matrix']]

def execute_grasp(grasp):
    print(f"\nAPI 调用: execute_grasp(...)")
    real_robot_system.execute_grasp(grasp)

def main():
    global real_robot_system
    try:
        real_robot_system = RealRobotEyeOnHandGrasp()
        
        task = input("\n请输入您的指令 (例如 '帮我拿一下香蕉吧'): ")
        if not real_robot_system.llm_client:
            print("LLM 未初始化，无法执行任务。")
            return

        print('好的！正在向 LLM 请求代码计划...')
        code = real_robot_system.plan_from_llm(task)
        if not code:
            return

        pattern = r"```python\n([\s\S]*?)```"
        match = re.search(pattern, code, re.DOTALL)

        if match:
            code_block = match.group(1).strip()
            print("--- LLM 生成的代码 ---")
            print(code_block)
            print("----------------------")
            
            exec_globals = globals()
            exec(code_block, exec_globals)
            exec_globals['plan']()
        else:
            print("错误：无法从 LLM 的响应中解析出 Python 代码。")
            print("LLM 原始响应:", code)

    except Exception as e:
        print(f"程序运行中发生未处理的异常: {e}")
    finally:
        if real_robot_system:
            real_robot_system.cleanup()

if __name__ == '__main__':
    main()