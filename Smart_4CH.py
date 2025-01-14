import cv2
import numpy as np

# 读取图像
image = cv2.imread('./dataset/img_1.png')
if image is None:
    print("Error: Could not read the image.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Image', binary)

# 查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化面积
total_area = 0

# 遍历每个轮廓并计算面积
for contour in contours:
    area = cv2.contourArea(contour)
    total_area += area

# 最大的轮廓为左心室轮廓
left_ventricle_contour = max(contours, key=cv2.contourArea)

#最小的轮廓为左心室轮廓
#left_ventricle_contour = min(contours, key=cv2.contourArea)

# 计算轮廓的质心
M = cv2.moments(left_ventricle_contour)
if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
else:
    cx, cy = 0, 0

# 获取左心室轮廓的外接凸包
hull = cv2.convexHull(left_ventricle_contour)

# 计算最远点的横向和纵向长度
min_x = np.min(hull[:, :, 0])
max_x = np.max(hull[:, :, 0])
min_y = np.min(hull[:, :, 1])
max_y = np.max(hull[:, :, 1])

# 横向和纵向长度
ventricle_width = max_x - min_x
ventricle_height = max_y - min_y





# 绘制外接凸包
cv2.drawContours(image, [hull], -1, (0, 255, 0), 1)

# 绘制质心
cv2.circle(image, (cx, cy), 2, (255, 255, 0), -1)  # 黄色圆点表示质心

# 绘制横向的线（平行于x轴）
horizontal_line_start = (min_x, cy)
horizontal_line_end = (max_x, cy)
cv2.line(image, horizontal_line_start, horizontal_line_end, (255, 0, 0), 1)  # 红色线

# 绘制纵向的线（平行于y轴）
vertical_line_start = (cx, min_y)
vertical_line_end = (cx, max_y)
cv2.line(image, vertical_line_start, vertical_line_end, (0, 0, 255), 1)  # 蓝色线

# 计算交点的函数
def find_intersections(contour, line_func):
    intersections = []
    n = len(contour)
    for i in range(n):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % n][0]  # 处理闭合轮廓并正确解包
        x1, y1 = p1
        x2, y2 = p2
        x, y = line_func(x1, y1, x2, y2)
        if x is not None and y is not None:
            intersections.append((int(round(x)), int(round(y))))
    return intersections

# 计算横向线与轮廓的交点
def horizontal_line(x1, y1, x2, y2):
    if y1 == y2 == cy:
        return None, None
    if y1 == y2:
        return None, None
    t = (cy - y1) / (y2 - y1)
    if 0 <= t <= 1:
        x = x1 + t * (x2 - x1)
        return x, cy
    return None, None

horizontal_intersections = find_intersections(left_ventricle_contour, horizontal_line)

# 计算纵向线与轮廓的交点
def vertical_line(x1, y1, x2, y2):
    if x1 == x2 == cx:
        return None, None
    if x1 == x2:
        return None, None
    t = (cx - x1) / (x2 - x1)
    if 0 <= t <= 1:
        y = y1 + t * (y2 - y1)
        return cx, y
    return None, None

vertical_intersections = find_intersections(left_ventricle_contour, vertical_line)

# 检查并绘制交点
def draw_intersection(intersections, color):
    for point in intersections:
        dist = cv2.pointPolygonTest(left_ventricle_contour, point, False)
        if dist >= 0:
            cv2.circle(image, point, 3, color, -1)  # 绿色圆点表示交点在轮廓上
        else:
            # 找到最近的轮廓点
            distances = np.linalg.norm(left_ventricle_contour - point, axis=2).flatten()
            closest_point_index = np.argmin(distances)
            closest_point = tuple(left_ventricle_contour[closest_point_index][0])
            cv2.circle(image, closest_point, 3, color, -1)  # 绿色圆点表示最近的轮廓点

draw_intersection(horizontal_intersections, (0, 255, 0))  # 绿色圆点表示横向交点
draw_intersection(vertical_intersections, (0, 255, 0))    # 绿色圆点表示纵向交点

# 计算横向交点之间的距离
if len(horizontal_intersections) == 2:
    horizontal_distance = np.linalg.norm(np.array(horizontal_intersections[0]) - np.array(horizontal_intersections[1]))
    print(f"Horizontal Distance between intersections: {horizontal_distance} pixels")
else:
    print("Error: Expected 2 horizontal intersections, but found", len(horizontal_intersections))

# 计算纵向交点之间的距离
if len(vertical_intersections) == 2:
    vertical_distance = np.linalg.norm(np.array(vertical_intersections[0]) - np.array(vertical_intersections[1]))
    print(f"Vertical Distance between intersections: {vertical_distance} pixels")
else:
    print("Error: Expected 2 vertical intersections, but found", len(vertical_intersections))

# 计算LVSI
LVSI = vertical_distance / horizontal_distance
print(f"LVSI: {LVSI}")
# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
