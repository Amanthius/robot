# -*- coding: utf-8 -*-
import time
import numpy as np
import cv2
import open3d as o3d
import requests
import base64
import re
import copy
from scipy.spatial.transform import Rotation as R
from openai import OpenAI
import matplotlib.pyplot as plt # For visualization if needed

# --- Import Your Modules ---
try:
    from graspnet.graspnet import GraspBaseline
    from graspnet.grasp import GraspGroup # Although not directly used, good practice
except ImportError as e:
    print(f"Error importing GraspNet modules: {e}")
    print("Please ensure 'graspnet' folder is in the Python path or current directory.")
    exit()

try:
    from camera_realsense import RealsenseCamera
except ImportError as e:
    print(f"Error importing RealSense camera module: {e}")
    print("Please ensure 'camera_realsense.py' is in the Python path or current directory.")
    exit()

try:
    from robotcontrol import Auboi5Robot, RobotErrorType, RobotIOType, RobotToolIoName, RobotUserIoName, logger_init
except ImportError as e:
    print(f"Error importing Robot control module: {e}")
    print("Please ensure 'robotcontrol.py' and 'libpyauboi5' are setup correctly.")
    exit()

# --- Configuration ---
ROBOT_IP = "192.168.1.40"  # <<<--- ADJUST: Your AUBO Robot IP
YOLO_API_URL = "http://127.0.0.1:5000/detect"
HAND_EYE_CALIBRATION_FILE = "hand_eye_gripper_to_cam_matrix.npy" # From your calibration script
LLM_API_KEY_FILE = "keys.txt"
LLM_BASE_URL = "https://api.deepseek.com" # Or your preferred endpoint
LLM_MODEL = "deepseek-coder" # Or your preferred model

# Gripper Control Configuration (<<<--- ADJUST THESE)
GRIPPER_IO_TYPE = RobotIOType.Tool_DO # Example: Tool Digital Output. Check robotcontrol.py for options (Tool_DO, User_DO etc.)
GRIPPER_OPEN_PIN = RobotToolIoName.tool_io_0 # Example: Tool IO Pin 0. Check robotcontrol.py for names
GRIPPER_CLOSE_PIN = RobotToolIoName.tool_io_0 # Example: Using the same pin, state determines open/close
GRIPPER_OPEN_VALUE = 1  # Example: Value 1 means open
GRIPPER_CLOSE_VALUE = 0 # Example: Value 0 means close
GRIPPER_WAIT_TIME = 1.0 # Seconds to wait after gripper action

# Initialize Logging (from robotcontrol.py)
logger_init()
logger = Auboi5Robot.logger # Use the logger from robotcontrol

class RealRobotGrasping:
    def __init__(self):
        logger.info("Initializing Real Robot Grasping System...")

        # 1. Initialize GraspNet
        logger.info("Loading GraspNet model...")
        self.graspNet = GraspBaseline()
        logger.info("GraspNet model loaded.")

        # 2. Initialize RealSense Camera
        logger.info("Initializing RealSense Camera...")
        self.camera = RealsenseCamera()
        if not self.camera.start():
            raise RuntimeError("Failed to start RealSense camera.")
        self.cam_matrix, self.dist_coeffs = self.camera.get_intrinsics()
        if self.cam_matrix is None:
             raise RuntimeError("Failed to get camera intrinsics.")
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.camera.width, self.camera.height,
            self.cam_matrix[0, 0], self.cam_matrix[1, 1],
            self.cam_matrix[0, 2], self.cam_matrix[1, 2]
        )
        logger.info("RealSense Camera initialized.")

        # 3. Initialize AUBO Robot
        logger.info("Initializing AUBO Robot library...")
        if Auboi5Robot.initialize() != RobotErrorType.RobotError_SUCC:
            raise RuntimeError("Failed to initialize AUBO robot library (libpyauboi5).")
        self.robot = Auboi5Robot()
        self.robot.create_context()
        logger.info(f"Connecting to AUBO robot at {ROBOT_IP}...")
        if self.robot.connect(ROBOT_IP, 8899) != RobotErrorType.RobotError_SUCC:
            raise ConnectionError(f"Failed to connect to AUBO robot at {ROBOT_IP}")
        # Enable event callback if needed (optional)
        # self.robot.enable_robot_event()
        logger.info("AUBO Robot connected. Starting up...")
        # Robot startup (power on, collision settings) - may take time
        # Using default collision and tool dynamics for now
        self.robot.robot_startup()
        self.robot.init_profile() # Initialize global move profile
        # Set reasonable speed/acceleration limits (<<<--- ADJUST if needed)
        self.robot.set_joint_maxvelc((1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.robot.set_joint_maxacc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))
        self.robot.set_end_max_line_velc(0.2)
        self.robot.set_end_max_line_acc(0.5)
        logger.info("AUBO Robot started up and profile initialized.")

        # 4. Load Hand-Eye Calibration Matrix (Gripper to Camera)
        logger.info(f"Loading Hand-Eye calibration matrix from {HAND_EYE_CALIBRATION_FILE}...")
        try:
            self.T_gripper_cam = np.load(HAND_EYE_CALIBRATION_FILE)
            logger.info("Hand-Eye matrix (T_gripper_cam):\n{}".format(np.round(self.T_gripper_cam, 4)))
        except FileNotFoundError:
            raise FileNotFoundError(f"Hand-Eye calibration file not found: {HAND_EYE_CALIBRATION_FILE}. Please run calibration first.")
        except Exception as e:
            raise RuntimeError(f"Error loading Hand-Eye matrix: {e}")

        # 5. Initialize LLM Client
        logger.info("Initializing LLM client...")
        try:
            api_key = open(LLM_API_KEY_FILE).readline().strip()
            self.llm_client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
            # Test connection (optional but recommended)
            self.llm_client.models.list()
            logger.info("LLM client initialized.")
        except FileNotFoundError:
             raise FileNotFoundError(f"LLM API key file not found: {LLM_API_KEY_FILE}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {e}")

        # Ensure gripper is open initially
        self.open_gripper()


    def get_rgbd_pointcloud(self):
        """Gets RGB, Depth, and generates Point Cloud in Camera Frame {C}."""
        depth_frame, color_frame, depth_image_raw, color_image_raw = self.camera.get_frames()
        if color_image_raw is None or depth_image_raw is None:
            logger.error("Failed to get frames from camera.")
            return None, None, None

        color_o3d = o3d.geometry.Image(color_image_raw)
        depth_o3d = o3d.geometry.Image(depth_image_raw)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.camera.depth_scale,
            depth_trunc=1.5, # Truncate depth beyond 1.5m (<<<--- ADJUST if needed)
            convert_rgb_to_intensity=False
        )

        pcd_c = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.o3d_intrinsics
        )
        # Optional: Downsample point cloud if too dense
        # pcd_c = pcd_c.voxel_down_sample(voxel_size=0.005)

        return color_image_raw, depth_image_raw, pcd_c

    def get_grasp_candidates(self, pcd_c, samples=100, vis_o3d=False):
        """Runs GraspNet and transforms grasps to Robot Base Frame {B}."""
        if pcd_c is None or len(pcd_c.points) == 0:
             logger.warning("Empty point cloud received, skipping GraspNet.")
             return {'score': [], 'grasp_b': []} # Return empty dict

        logger.info(f"Running GraspNet on point cloud with {len(pcd_c.points)} points...")
        # GraspNet expects point cloud relative to camera.
        # Note: GraspNet might require specific orientation (e.g., Z facing away from camera)
        # Add transformation here if needed before passing to graspNet.run()
        # Example: T_graspnet_cam = np.array([...]) # Transform if GraspNet needs different camera convention
        # gg = self.graspNet.run(copy.deepcopy(pcd_c).transform(T_graspnet_cam), vis=False)
        gg = self.graspNet.run(copy.deepcopy(pcd_c), vis=False) # Assuming GraspNet uses standard camera frame
        logger.info(f"GraspNet generated {len(gg)} initial grasps.")

        gg.nms()
        gg.sort_by_score()
        logger.info(f"{len(gg)} grasps remaining after NMS.")

        grasp_poses = {'score':[], 'grasp_b':[]} # Store grasps in base frame {B}

        # Get current robot pose T_bg (Base to Gripper)
        waypoint = self.robot.get_current_waypoint()
        if not waypoint:
            logger.error("Failed to get current robot waypoint. Cannot transform grasps.")
            return grasp_poses # Return empty

        pos_b_g = waypoint['pos']
        ori_quat_b_g = waypoint['ori'] # (w, x, y, z)
        # Convert to 4x4 matrix T_bg
        T_bg = np.eye(4)
        T_bg[:3, :3] = R.from_quat([ori_quat_b_g[1], ori_quat_b_g[2], ori_quat_b_g[3], ori_quat_b_g[0]]).as_matrix() # scipy uses x,y,z,w
        T_bg[:3, 3] = pos_b_g

        # Calculate T_bc = T_bg @ T_gc
        T_bc = T_bg @ self.T_gripper_cam

        if vis_o3d:
            geometries = []
            # Coordinate frame for camera relative to base
            cam_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            cam_frame_vis.transform(T_bc)
            geometries.append(cam_frame_vis)
            # World/Base frame
            world_frame_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            geometries.append(world_frame_vis)
             # Point cloud transformed to base frame for visualization
            pcd_vis = copy.deepcopy(pcd_c).transform(T_bc)
            geometries.append(pcd_vis)


        logger.info(f"Transforming top {min(samples, len(gg))} grasps to base frame...")
        for i, grasp in enumerate(gg[:samples]):
            # grasp definition: score, width, height, depth, rotation(9), translation(3), object_id
            # Get grasp matrix relative to camera {C}
            grasp_matrix_c = np.eye(4)
            grasp_matrix_c[:3, :3] = grasp.rotation_matrix
            grasp_matrix_c[:3, 3] = grasp.translation

            # Transform grasp to base frame {B}: T_bc * Grasp_c
            grasp_matrix_b = T_bc @ grasp_matrix_c

            grasp_poses['score'].append(grasp.score)
            grasp_poses['grasp_b'].append(grasp_matrix_b) # Store 4x4 matrix in base frame

            if vis_o3d and i < 20: # Visualize top 20 grasps
                # Create gripper geometry (reuse GraspNet's function if possible, or simple boxes)
                 # Note: grasp.to_open3d_geometry() is in camera frame. Transform it.
                gripper_vis = grasp.to_open3d_geometry() # This gives a list of geometries
                if isinstance(gripper_vis, list): # Check if it returned a list
                    for geom in gripper_vis:
                        geom.transform(T_bc) # Transform each part
                        geometries.append(geom)
                else: # Assume it's a single geometry
                     gripper_vis.transform(T_bc)
                     geometries.append(gripper_vis)


        if vis_o3d:
            logger.info("Displaying Open3D visualization (point cloud, grasps in base frame)... Close window to continue.")
            o3d.visualization.draw_geometries(geometries)

        logger.info(f"Prepared {len(grasp_poses['grasp_b'])} grasp candidates in base frame.")
        return grasp_poses

    def detect_by_text(self, color_image, class_text):
        """Calls the YOLO-World API service."""
        if color_image is None:
             logger.error("No color image provided for detection.")
             return []

        logger.info(f"Sending image to YOLO-World API for class: '{class_text}'...")
        try:
            # Convert to JPEG bytes, then Base64
            _, buffer = cv2.imencode('.jpg', color_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            data = {
                'image': image_base64,
                'classes': [class_text]
            }
            response = requests.post(YOLO_API_URL, json=data, timeout=10) # 10 second timeout

            if response.status_code == 200:
                detections = response.json().get('detections', [])
                logger.info(f"YOLO-World detected {len(detections)} instances.")
                # Detections format: [x1, y1, x2, y2, score, class_id_from_yolo (float)]
                # Convert class_id back to name if needed, but here we only have one class
                processed_detections = []
                for det in detections:
                     # Ensure values are valid before adding
                    if len(det) >= 5 and all(isinstance(x, (int, float)) for x in det[:5]):
                        processed_detections.append(det[:5] + [class_text]) # x1,y1,x2,y2,score,name
                    else:
                        logger.warning(f"Received invalid detection format: {det}")
                return processed_detections
            else:
                logger.error(f"YOLO-World API error: {response.status_code} - {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to YOLO-World API at {YOLO_API_URL}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error during YOLO-World detection call: {e}")
            return []


    def project_base_to_pixel(self, points_b):
        """Projects 3D points from Robot Base Frame {B} to 2D Pixel Coordinates."""
        if not isinstance(points_b, np.ndarray):
            points_b = np.array(points_b)
        if points_b.ndim == 1:
            points_b = points_b.reshape(1, 3) # Make it (1, 3) if single point

        # Need T_cb = inv(T_bc) = inv(T_bg @ T_gc)
        waypoint = self.robot.get_current_waypoint()
        if not waypoint:
            logger.error("Failed to get current robot waypoint for projection.")
            return np.array([])
        pos_b_g = waypoint['pos']; ori_quat_b_g = waypoint['ori']
        T_bg = np.eye(4)
        T_bg[:3, :3] = R.from_quat([ori_quat_b_g[1], ori_quat_b_g[2], ori_quat_b_g[3], ori_quat_b_g[0]]).as_matrix()
        T_bg[:3, 3] = pos_b_g
        T_bc = T_bg @ self.T_gripper_cam
        try:
            T_cb = np.linalg.inv(T_bc)
        except np.linalg.LinAlgError:
            logger.error("Failed to invert T_bc matrix for projection.")
            return np.array([])

        points_b_h = np.hstack((points_b, np.ones((points_b.shape[0], 1)))) # Homogeneous coords
        points_c_h = (T_cb @ points_b_h.T).T
        
        # Dehomogenize and filter points behind camera or too close
        valid_idx = (points_c_h[:, 2] > 1e-6) # Check if Z is positive
        points_c = points_c_h[valid_idx, :3] / points_c_h[valid_idx, 3:4]
        
        if points_c.shape[0] == 0:
            return np.array([]) # No valid points to project

        # Project points_c (X, Y, Z) to pixels (u, v) using intrinsics
        x_c = points_c[:, 0]
        y_c = points_c[:, 1]
        z_c = points_c[:, 2]

        fx = self.cam_matrix[0, 0]
        fy = self.cam_matrix[1, 1]
        cx = self.cam_matrix[0, 2]
        cy = self.cam_matrix[1, 2]

        u = fx * (x_c / z_c) + cx
        v = fy * (y_c / z_c) + cy

        pixels = np.vstack((u, v)).T

        # Create a result array matching original size, fill with NaN for invalid points
        result_pixels = np.full((points_b_h.shape[0], 2), np.nan)
        result_pixels[valid_idx] = pixels

        return result_pixels


    def filter_grasp_by_text(self, class_text, grasp_poses, color_image, vis=False):
        """Filters grasp_poses (in base frame) based on YOLO detection bbox."""
        if not grasp_poses or not grasp_poses['grasp_b']:
            logger.warning("No grasp candidates to filter.")
            return []

        detections = self.detect_by_text(color_image, class_text)
        if not detections:
            logger.warning(f"No objects of class '{class_text}' detected by YOLO.")
            return []

        # Assume the first detection is the target (or implement logic to choose)
        det = detections[0]
        x1, y1, x2, y2, yolo_score, _ = det
        logger.info(f"Using YOLO detection for '{class_text}' at bbox [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] with score {yolo_score:.2f}")

        # Extract 3D grasp points (translation part) from the base frame matrices
        grasp_points_b = np.array([grasp_matrix[:3, 3] for grasp_matrix in grasp_poses['grasp_b']])

        # Project these 3D points to 2D pixel coordinates
        grasp_pixels = self.project_base_to_pixel(grasp_points_b)

        filtered_indices = []
        valid_pixels = []
        for i, px in enumerate(grasp_pixels):
             # Check if projection was valid (not NaN) and inside bbox
            if not np.isnan(px).any() and x1 < px[0] < x2 and y1 < px[1] < y2:
                filtered_indices.append(i)
                valid_pixels.append(px)

        logger.info(f"Filtered {len(filtered_indices)} grasps within the YOLO bbox.")

        if not filtered_indices:
            return []

        # Select the best grasp from the filtered list
        # Strategy: Combine GraspNet score with a rule (e.g., prefer grasps closer to bbox center, or more top-down)

        best_score = -1.0
        best_grasp_idx = -1

        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2

        for idx, px in zip(filtered_indices, valid_pixels):
            graspnet_score = grasp_poses['score'][idx]

            # --- Simple Rule Example: Penalize distance from bbox center ---
            dist_to_center = np.sqrt((px[0] - bbox_center_x)**2 + (px[1] - bbox_center_y)**2)
            # Normalize distance roughly (max possible dist is diagonal of image)
            max_dist = np.sqrt(self.camera.width**2 + self.camera.height**2)
            center_penalty = (dist_to_center / max_dist) * 0.1 # Small penalty, max 0.1
            
            # --- Rule Example 2: Prefer more top-down grasps (Z-axis of grasp aligned with -Z of base) ---
            grasp_matrix_b = grasp_poses['grasp_b'][idx]
            grasp_z_axis_b = grasp_matrix_b[:3, 2] # Z-axis of gripper in base frame
            base_neg_z_axis = np.array([0, 0, -1])
            # Dot product gives alignment (1 for aligned, -1 for opposite, 0 for perpendicular)
            # We want alignment with -Z base, so dot product close to 1
            top_down_score_bonus = (np.dot(grasp_z_axis_b, base_neg_z_axis) + 1) / 2 * 0.1 # Bonus up to 0.1


            # Combine scores (<<<--- ADJUST weights/logic as needed)
            combined_score = graspnet_score - center_penalty + top_down_score_bonus

            if combined_score > best_score:
                best_score = combined_score
                best_grasp_idx = idx

        if best_grasp_idx == -1:
             return [] # Should not happen if filtered_indices is not empty

        best_grasp_matrix_b = grasp_poses['grasp_b'][best_grasp_idx]
        best_graspnet_score = grasp_poses['score'][best_grasp_idx]
        best_pixel = grasp_pixels[best_grasp_idx]
        logger.info(f"Selected best grasp (Index {best_grasp_idx}) with Combined Score: {best_score:.3f} (GraspNet: {best_graspnet_score:.3f})")


        if vis:
            vis_image = color_image.copy()
            # Draw YOLO box
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis_image, f"{class_text} ({yolo_score:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Draw filtered grasp points
            for i, px in enumerate(grasp_pixels):
                if i in filtered_indices and not np.isnan(px).any():
                     color = (0, 0, 255) # Red for filtered
                     if i == best_grasp_idx:
                         color = (255, 0, 255) # Magenta for best
                         cv2.circle(vis_image, (int(px[0]), int(px[1])), 8, color, -1)
                     else:
                          cv2.circle(vis_image, (int(px[0]), int(px[1])), 5, color, -1)

            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Best Grasp for '{class_text}' (Score: {best_score:.2f})")
            plt.show()

        return [best_grasp_matrix_b] # Return as a list containing the best 4x4 matrix


    def matrix_to_pose_components(self, matrix_b):
        """Converts 4x4 matrix in base frame to (pos, rpy/quat) for robot control."""
        pos = matrix_b[:3, 3].tolist() # [x, y, z]
        rot_matrix = matrix_b[:3, :3]
        
        # Convert rotation matrix to Euler angles (RPY - XYZ convention often used by robots)
        try:
             # <<<--- ADJUST 'xyz' if your robot uses a different Euler convention (e.g., 'zyx')
            rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False) # Use radians
            # Note: robotcontrol might expect degrees, check its API
            # If degrees: rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
        except Exception as e:
            logger.error(f"Error converting rotation matrix to RPY: {e}")
            return None, None
        
        # Alternatively, convert to Quaternion if robot API prefers it
        # quat_xyzw = R.from_matrix(rot_matrix).as_quat() # gives [x, y, z, w]
        # quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]] # Convert to [w, x, y, z] if needed
        
        return pos, rpy # Or return pos, quat_wxyz


    def open_gripper(self):
        logger.info("Opening gripper...")
        try:
            # <<<--- ADJUST IO_Type, Pin Name, Value based on your gripper wiring
            self.robot.set_tool_io_status(GRIPPER_OPEN_PIN, GRIPPER_OPEN_VALUE)
            # Alternative if using control box IO:
            # self.robot.set_board_io_status(GRIPPER_IO_TYPE, GRIPPER_OPEN_PIN, GRIPPER_OPEN_VALUE)
            time.sleep(GRIPPER_WAIT_TIME)
            logger.info("Gripper opened.")
            return True
        except Exception as e:
            logger.error(f"Failed to open gripper: {e}")
            return False

    def close_gripper(self):
        logger.info("Closing gripper...")
        try:
            # <<<--- ADJUST IO_Type, Pin Name, Value based on your gripper wiring
            self.robot.set_tool_io_status(GRIPPER_CLOSE_PIN, GRIPPER_CLOSE_VALUE)
            # Alternative if using control box IO:
            # self.robot.set_board_io_status(GRIPPER_IO_TYPE, GRIPPER_CLOSE_PIN, GRIPPER_CLOSE_VALUE)
            time.sleep(GRIPPER_WAIT_TIME)
            logger.info("Gripper closed.")
            return True
        except Exception as e:
            logger.error(f"Failed to close gripper: {e}")
            return False


    def execute_grasp(self, grasp_matrix_b):
        """Executes the grasp sequence using the real robot."""
        logger.info("Executing grasp sequence...")

        # --- Define Offsets (relative to grasp frame) ---
        # <<<--- ADJUST these offsets based on your gripper and environment
        pre_grasp_offset_z = -0.10 # Move 10cm back along gripper's Z-axis (approach axis)
        post_grasp_lift_z = 0.15   # Lift 15cm straight up in base frame Z-axis

        # Calculate pre-grasp pose
        pre_grasp_transform = np.eye(4)
        pre_grasp_transform[2, 3] = pre_grasp_offset_z # Offset along Z-axis of the grasp pose
        pre_grasp_matrix_b = grasp_matrix_b @ pre_grasp_transform

        # Convert matrices to robot command format (pos, rpy)
        pre_grasp_pos, pre_grasp_rpy = self.matrix_to_pose_components(pre_grasp_matrix_b)
        grasp_pos, grasp_rpy = self.matrix_to_pose_components(grasp_matrix_b)

        if pre_grasp_pos is None or grasp_pos is None:
             logger.error("Failed to convert grasp matrices to robot pose components.")
             return False

        try:
            # 1. Open gripper
            if not self.open_gripper(): return False

            # 2. Move to pre-grasp position (Axis move - typically safer)
            logger.info(f"Moving to pre-grasp joint configuration...")
            # Use inverse kinematics to find joint angles for pre-grasp cartesian pose
            current_waypoint = self.robot.get_current_waypoint()
            if not current_waypoint: raise RuntimeError("Failed to get current waypoint for IK")
            
            ik_result_pre = self.robot.inverse_kin(current_waypoint['joint'], pre_grasp_pos, self.robot.rpy_to_quaternion(pre_grasp_rpy))
            if not ik_result_pre or 'joint' not in ik_result_pre:
                logger.error(f"Inverse kinematics failed for pre-grasp pose: {pre_grasp_pos}, {pre_grasp_rpy}")
                return False
            self.robot.move_joint(ik_result_pre['joint'])
            logger.info("Reached pre-grasp pose.")
            time.sleep(0.5)

            # 3. Move linearly to grasp position (Line move)
            logger.info("Moving linearly to grasp pose...")
             # Use inverse kinematics for the final grasp pose as well
            ik_result_grasp = self.robot.inverse_kin(ik_result_pre['joint'], grasp_pos, self.robot.rpy_to_quaternion(grasp_rpy))
            if not ik_result_grasp or 'joint' not in ik_result_grasp:
                logger.error(f"Inverse kinematics failed for grasp pose: {grasp_pos}, {grasp_rpy}")
                # Attempt to move back safely
                self.robot.move_joint(ik_result_pre['joint'])
                return False
            # <<<--- Check if robotcontrol has move_line that accepts cartesian targets, otherwise use move_line with joints
            # If move_line takes joints:
            self.robot.move_line(ik_result_grasp['joint'])
            # If move_line takes cartesian (less common for safety):
            # self.robot.move_line_cartesian(grasp_pos, grasp_rpy) # Fictional function
            logger.info("Reached grasp pose.")
            time.sleep(0.5)

            # 4. Close gripper
            if not self.close_gripper():
                 # Attempt to move back safely even if close fails
                 self.robot.move_line(ik_result_pre['joint'])
                 return False

            # 5. Move linearly back to pre-grasp position
            logger.info("Moving linearly back to pre-grasp pose...")
            self.robot.move_line(ik_result_pre['joint'])
            logger.info("Returned to pre-grasp pose.")
            time.sleep(0.5)

            # 6. Lift the object straight up
            logger.info(f"Lifting object by {post_grasp_lift_z}m...")
            lift_pos = list(pre_grasp_pos) # Copy position
            lift_pos[2] += post_grasp_lift_z # Add Z offset in base frame
            
            # Use IK for lift position
            ik_result_lift = self.robot.inverse_kin(ik_result_pre['joint'], lift_pos, self.robot.rpy_to_quaternion(pre_grasp_rpy))
            if not ik_result_lift or 'joint' not in ik_result_lift:
                 logger.warning(f"Inverse kinematics failed for lift pose: {lift_pos}, {pre_grasp_rpy}. Skipping lift.")
            else:
                 # Move linearly up if possible, otherwise joint move
                 # self.robot.move_line(ik_result_lift['joint']) # Prefer line move if safe
                 self.robot.move_joint(ik_result_lift['joint']) # Safer default
                 logger.info("Lift complete.")
                 time.sleep(0.5)

            logger.info("Grasp sequence finished successfully.")
            return True

        except Exception as e:
            logger.error(f"Error during grasp execution: {e}")
            # Attempt a safe stop or move back
            try:
                self.robot.move_stop()
                # Optional: Try moving to a known safe joint config if IK available
                # safe_joints = (0, 0, np.pi/2, 0, np.pi/2, 0) # Example safe pose
                # self.robot.move_joint(safe_joints)
            except Exception as stop_e:
                 logger.error(f"Error trying to stop robot after failure: {stop_e}")
            return False

    def plan_from_llm(self, task):
        """Gets a Python plan from the LLM based on the task."""
        logger.info(f"Requesting plan from LLM for task: '{task}'")
        # Same template as before
        template = """你是一个机器人，你拥有的技能API如下：
        1. get_grasp_by_name(name_text): 输入物体类别文本（英文，简短），扫描场景，检测物体和抓取姿态，返回一个包含最佳抓取姿态(4x4矩阵)的列表，如果找不到则返回空列表。
        2. execute_grasp(grasp_matrix): 输入一个抓取姿态(4x4矩阵)，机器人将执行抓取动作序列（靠近、抓取、抬起）。
        现在需要你根据你所拥有的技能API，编写python代码完成给你的任务，只输出plan函数，代码需要包含在\`\`\`python ... \`\`\`中，不要输出其他代码以外的内容。你的任务是“KKKK”。
        """
        prompt = template.replace('KKKK', task)

        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful robot assistant that writes Python code to accomplish tasks using provided APIs."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1 # Low temperature for deterministic code generation
            )
            content = response.choices[0].message.content
            logger.info("LLM plan received.")
            return content
        except Exception as e:
            logger.error(f"Error communicating with LLM: {e}")
            return None


    def cleanup(self):
        """Shuts down robot and camera."""
        logger.info("Cleaning up resources...")
        try:
            if self.robot and self.robot.connected:
                logger.info("Shutting down robot...")
                # Optional: Move to a safe home position before shutdown
                # home_joints = (0, 0, np.pi/2, 0, np.pi/2, 0)
                # self.robot.move_joint(home_joints)
                self.robot.robot_shutdown()
                self.robot.disconnect()
                logger.info("Robot disconnected.")
            # Uninitialize the library
            Auboi5Robot.uninitialize()
            logger.info("Robot library uninitialized.")
        except Exception as e:
            logger.error(f"Error during robot cleanup: {e}")

        try:
            if self.camera:
                logger.info("Stopping camera...")
                self.camera.stop()
                logger.info("Camera stopped.")
        except Exception as e:
            logger.error(f"Error during camera cleanup: {e}")

        logger.info("Cleanup finished.")


# --- Global API functions for LLM ---
robot_grasping_instance = None # Global instance

def get_grasp_by_name(name_text):
    """API function: Scans scene, detects object & grasps, returns best grasp matrix."""
    if robot_grasping_instance is None:
         logger.error("Robot system not initialized.")
         return []
    logger.info(f"API CALL: get_grasp_by_name('{name_text}')")
    try:
        # 1. Get Point Cloud and Color Image
        color_image, _, pcd_c = robot_grasping_instance.get_rgbd_pointcloud()
        if pcd_c is None or color_image is None:
            return []

        # 2. Get Grasp Candidates (in base frame)
        # Increase samples for real world?
        grasp_candidates_b = robot_grasping_instance.get_grasp_candidates(pcd_c, samples=150, vis_o3d=False) # <<<--- Set vis_o3d=True for debugging grasps

        # 3. Filter Grasps using YOLO and Text
        best_grasp_list = robot_grasping_instance.filter_grasp_by_text(name_text, grasp_candidates_b, color_image, vis=True) # <<<--- Set vis=True for bbox/grasp visualization

        if best_grasp_list:
            logger.info(f"API CALL: Found best grasp for '{name_text}'.")
            return best_grasp_list # List containing the 4x4 matrix
        else:
            logger.warning(f"API CALL: No valid grasp found for '{name_text}' after filtering.")
            return []
    except Exception as e:
        logger.error(f"API CALL: Error in get_grasp_by_name: {e}")
        return []

def execute_grasp(grasp_matrix):
    """API function: Executes the grasp sequence for the given grasp matrix."""
    if robot_grasping_instance is None:
         logger.error("Robot system not initialized.")
         return False # Indicate failure
    logger.info("API CALL: execute_grasp(...) called.")
    if not isinstance(grasp_matrix, np.ndarray) or grasp_matrix.shape != (4, 4):
        logger.error("API CALL: Invalid grasp_matrix provided to execute_grasp.")
        return False
    try:
        success = robot_grasping_instance.execute_grasp(grasp_matrix)
        logger.info(f"API CALL: execute_grasp finished. Success: {success}")
        return success
    except Exception as e:
        logger.error(f"API CALL: Error in execute_grasp: {e}")
        return False

# --- Main Execution Logic ---
if __name__ == "__main__":
    try:
        robot_grasping_instance = RealRobotGrasping()

        # --- Interactive Loop ---
        while True:
            try:
                task = input("请输入您的指令 (例如 '帮我拿起香蕉', 输入 'q' 退出): ")
                if task.lower() == 'q':
                    break
                if not task:
                    continue

                # Get plan from LLM
                llm_response = robot_grasping_instance.plan_from_llm(task)
                if not llm_response:
                    logger.warning("未能从LLM获取计划。")
                    continue

                # Extract Python code
                code_block = None
                pattern = r"```python\n([\s\S]*?)```"
                match = re.search(pattern, llm_response, re.DOTALL)
                if match:
                    code_block = match.group(1).strip()
                    logger.info("从LLM响应中提取的代码:\n---\n{}\n---".format(code_block))
                else:
                    logger.warning("未能从LLM响应中提取有效的Python代码块。响应:\n{}".format(llm_response))
                    continue

                # Execute the code safely
                logger.info("准备执行LLM生成的代码...")
                # Define the scope for exec, including our API functions
                exec_scope = {
                    "get_grasp_by_name": get_grasp_by_name,
                    "execute_grasp": execute_grasp,
                    "np": np # Make numpy available if needed in plan
                }
                # Wrap the extracted code in a function call `plan()` if it defines one
                if "def plan():" in code_block:
                    exec_code = code_block + "\nplan()"
                else:
                    # If it's just direct calls, execute as is
                    logger.warning("LLM code does not define plan(). Executing directly.")
                    exec_code = code_block

                try:
                    exec(exec_code, exec_scope)
                    logger.info("LLM代码执行完毕。")
                except Exception as exec_e:
                    logger.error(f"执行LLM生成的代码时出错: {exec_e}")
                    import traceback
                    traceback.print_exc() # Print full traceback for debugging

            except KeyboardInterrupt:
                logger.info("接收到中断信号...")
                break
            except Exception as loop_e:
                 logger.error(f"主循环中发生错误: {loop_e}")
                 time.sleep(1) # Avoid tight loop on error

    except Exception as main_e:
        logger.error(f"系统初始化或运行时发生严重错误: {main_e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot_grasping_instance:
            robot_grasping_instance.cleanup()
        logger.info("Real Robot Grasping 程序结束。")
