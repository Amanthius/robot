# coding=utf-8
import cv2
import numpy as np

# --- 配置 ---
# ArUco 字典 (与 calibrate_eye_on_hand.py 保持一致)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 标定板参数
MARKERS_X = 5        # X 方向的标记数量
MARKERS_Y = 4        # Y 方向的标记数量
MARKER_LENGTH_PX = 200 # 每个标记的像素大小 (200px)
MARKER_SEP_PX = 50   # 标记之间的间隔 (50px)

# 输出文件名
FILENAME = "aruco_board.png"
# --- 结束配置 ---

def create_aruco_board():
    """
    创建并保存一个 ArUco 标定板图像。
    """
    print(f"正在生成 {MARKERS_X}x{MARKERS_Y} 的 ArUco 标定板...")
    
    # 创建标定板对象 (注意：OpenCV的board.generateImage单位是像素)
    # 我们将在标定时使用以米为单位的真实测量值
    board = cv2.aruco.GridBoard(
        size=(MARKERS_X, MARKERS_Y),
        markerLength=MARKER_LENGTH_PX,
        markerSeparation=MARKER_SEP_PX,
        dictionary=ARUCO_DICT
    )
    
    # 计算图像的总大小
    img_width = MARKERS_X * (MARKER_LENGTH_PX + MARKER_SEP_PX) - MARKER_SEP_PX + 2 * MARKER_SEP_PX
    img_height = MARKERS_Y * (MARKER_LENGTH_PX + MARKER_SEP_PX) - MARKER_SEP_PX + 2 * MARKER_SEP_PX
    
    # 生成图像
    board_image = board.generateImage((img_width, img_height))
    
    # 保存图像
    cv2.imwrite(FILENAME, board_image)
    
    print("\n--- 标定板生成成功! ---")
    print(f"文件名: {FILENAME}")
    print(f"图像尺寸: {board_image.shape[1]} x {board_image.shape[0]} 像素")
    
    print("\n--- 重要：打印和测量 ---")
    print(f"1. 请将 {FILENAME} 打印出来 (推荐使用A3或更大的纸张)。")
    print("2. 打印后，用尺子**精确测量**：")
    print(f"   a) 一个标记的实际边长 (单位：米) -> MARKER_SIZE_METERS")
    print(f"   b) 两个标记之间的间隔 (单位：米) -> MARKER_SEPARATION_METERS")
    print("3. 将这两个测量值填入 'calibrate_eye_on_hand.py' 脚本的配置中。")
    
    # 为了方便预览，显示图像
    # 注意: 在 WSL 中可能需要配置 X Server 才能显示
    try:
        cv2.imshow("ArUco Board", board_image)
        print("\n按任意键退出预览...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"\n无法打开预览窗口 (可能在 WSL 环境中)。请直接在文件夹中打开 {FILENAME}。")

if __name__ == "__main__":
    # 确保您已安装 opencv-contrib-python
    # pip install opencv-contrib-python
    create_aruco_board()