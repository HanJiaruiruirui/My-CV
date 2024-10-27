import cv2
import numpy as np

# 定义激光的颜色范围（假设是红色激光），需要调整范围以获得更好的效果
laser_color_lower = np.array([0, 100, 100])
laser_color_upper = np.array([10, 255, 255])

# 开启摄像头捕获
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 读取每一帧
    ret, frame = cap.read()
    if not ret:
        break
    
    # 转换为 HSV 色彩空间以便颜色分割
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 创建掩膜，仅保留激光的颜色
    mask = cv2.inRange(hsv_frame, laser_color_lower, laser_color_upper)
    
    # 对掩膜进行一些处理（如膨胀和腐蚀）以去除噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # 查找轮廓来识别激光点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # 计算轮廓的面积来过滤掉噪声
        area = cv2.contourArea(cnt)
        if area > 10:  # 根据实际效果调整面积阈值
            # 获取激光点的位置
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            # 绘制检测到的激光点
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.putText(frame, f"Laser detected at {center}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Laser Detection', frame)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
