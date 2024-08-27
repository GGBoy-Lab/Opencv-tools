import numpy as np
import cv2
import matplotlib.pyplot as plt

# 设置阈值
thresh = 1
inch_to_mm = 18
DPI = 200  # 使用默认DPI

def load_and_preprocess_image(image_path):
    """加载并预处理图像"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    return binary_img

def find_and_sort_contours(binary_img):
    """查找并按面积排序轮廓"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)

def calculate_centroid(contour):
    """计算轮廓的质心"""
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy
    return None

def draw_rectangle(image, center, width, height):
    """在图像上绘制矩形"""
    rect = plt.Rectangle((center[0] - width // 2, center[1] - height // 2), width, height,
                         edgecolor='r', facecolor='none', linewidth=2)
    plt.gca().add_patch(rect)

def extract_masked_region(binary_img, center, width, height):
    """提取矩形框内的区域"""
    rect_mask = np.zeros_like(binary_img)
    cv2.rectangle(rect_mask, (center[0] - width // 2, center[1] - height // 2),
                  (center[0] + width // 2, center[1] + height // 2), 255, -1)
    masked_img = cv2.bitwise_and(binary_img, rect_mask)
    return masked_img

def mDistance(x1, y1, x2, y2):
    """计算两个点之间的物理距离"""
    return ((((x1 - x2) / DPI) ** 2 + ((y1 - y2) / DPI) ** 2) ** 0.5) * inch_to_mm

def find_intersections(masked_img, centroid):
    """找到垂直线与白色区域的交点"""
    height, width = masked_img.shape
    intersections = []
    for y in range(height):
        if masked_img[y, centroid[0]] == 255:
            intersections.append((centroid[0], y))
    return intersections

def calculate_imt_left_right(masked_img, centroid, shift_range):
    """计算向左和向右移动shift_range个像素点后的IMT"""
    imts = []

    # 向左移动
    for shift in range(-shift_range, shift_range + 1):
        left_centroid = (centroid[0] + shift, centroid[1])
        left_intersections = find_intersections(masked_img, left_centroid)
        if len(left_intersections) >= 2:
            upper_left = left_intersections[0]
            lower_left = left_intersections[-1]
            imt_left = round(mDistance(upper_left[0], upper_left[1], lower_left[0], lower_left[1]), 3)
            imts.append((left_centroid, imt_left, upper_left, lower_left))

    return imts

def main():
    image_path = "XXX.png"
    binary_img = load_and_preprocess_image(image_path)

    # 寻找轮廓
    contours = find_and_sort_contours(binary_img)
    centroid = calculate_centroid(contours[0]) if contours else (0, 0)

    # 绘制不同尺寸的矩形
    rectangles = [(200, 200)]  # 宽和高的列表

    # 绘制图像
    plt.imshow(binary_img, cmap='gray')
    plt.title('B-mode of IMT')
    plt.xlabel('Lateral [mm]')
    plt.ylabel('Depth [mm]')

    # 绘制矩形
    for width, height in rectangles:
        draw_rectangle(binary_img, centroid, width, height)

    # 提取矩形框内的区域
    masked_img = extract_masked_region(binary_img, centroid, width, height)

    # 在矩形框内进行轮廓检测
    masked_contours = find_and_sort_contours(masked_img)

    # 计算矩形框内白色区域的质心
    centroid_masked = calculate_centroid(masked_contours[0]) if masked_contours else (0, 0)

    # 找到垂直线与白色区域的交点
    intersections = find_intersections(masked_img, centroid_masked)

    # 确保交点分别位于白色区域的上方和下方
    if len(intersections) >= 2:
        upper_intersection = intersections[0]
        lower_intersection = intersections[-1]
        plt.plot(upper_intersection[0], upper_intersection[1], 'bo', markersize=1)  # 小蓝圆点表示交点
        plt.plot(lower_intersection[0], lower_intersection[1], 'bo', markersize=1)  # 小蓝圆点表示交点

        # 计算IMT
        IMT = round(mDistance(upper_intersection[0], upper_intersection[1], lower_intersection[0], lower_intersection[1]), 3)
        print('IMT: ', IMT, 'mm')

        # 计算框内所有的IMT
        imts = calculate_imt_left_right(masked_img, centroid_masked, 100)
        for i, (centroid, imt, upper, lower) in enumerate(imts):
            print(f'IMT {i+1}: ', imt, 'mm')
            plt.plot(upper[0], upper[1], 'ro', markersize=0.2)  
            plt.plot(lower[0], lower[1], 'go', markersize=0.2)  

    # 显示图像
    plt.show()

if __name__ == "__main__":
    main()
