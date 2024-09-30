import cv2
import numpy as np
import time
start_time = time.time()
# 读取灰度图像
image = cv2.imread('./data/0-0.png', cv2.IMREAD_GRAYSCALE)

# 定义高斯模糊参数
kernel_size = (11, 11)
sigma = 0

# 定义 k 常数
k = 255


# 自定义方法进行平滑处理
def custom_gaussian_blur(image, kernel_size, sigma, iterations=100):
    smoothed_image = image.copy()
    kernel = np.ones(kernel_size) / (kernel_size[0] * kernel_size[1])

    for _ in range(iterations):
        smoothed_image = cv2.filter2D(smoothed_image, -1, kernel)
        smoothed_image = np.clip(smoothed_image, 0, 255).astype(np.uint8)

        # 计算每个像素周围邻域内的均值和标准差
        avg = cv2.boxFilter(smoothed_image, -1, kernel_size, normalize=True)
        std = cv2.boxFilter(smoothed_image ** 2, -1, kernel_size, normalize=True) - avg ** 2
        std = np.sqrt(std)

        # 计算 diff
        diff = smoothed_image - avg

        # 重新分配像素值
        rand = np.random.rand(*smoothed_image.shape)
        mask1 = (diff >= -k * std) & (diff <= k * std)
        mask2 = diff > k * std
        mask3 = diff < -k * std

        smoothed_image[mask1] = smoothed_image[mask1]
        smoothed_image[mask2] = avg[mask2] + rand[mask2] * std[mask2]
        smoothed_image[mask3] = avg[mask3] - rand[mask3] * std[mask3]

    return smoothed_image


# 平滑处理
smoothed_image = custom_gaussian_blur(image, kernel_size, sigma, iterations=1)

# 计算水平和垂直方向上的梯度
sobelx = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=11)
sobely = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=11)

# 计算梯度幅值
gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

# 计算梯度方向
gradient_direction = np.arctan2(sobely, sobelx) * (180 / np.pi)

# 设置阈值检测锯齿效应区域
threshold = 100
edge_mask = gradient_magnitude > threshold


# 根据梯度方向判断边缘像素类型
def classify_edge_pixels(gradient_magnitude, gradient_direction, edge_mask):
    vertical_threshold = 100
    horizontal_threshold = 100

    # 初始化边缘分类掩码
    vertical_edges = np.zeros_like(edge_mask, dtype=bool)
    horizontal_edges = np.zeros_like(edge_mask, dtype=bool)
    diagonal_edges = np.zeros_like(edge_mask, dtype=bool)

    # 分类边缘像素
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            if edge_mask[i, j]:
                direction = gradient_direction[i, j]
                if abs(direction) < vertical_threshold or abs(direction - 180) < vertical_threshold:
                    vertical_edges[i, j] = True
                elif abs(direction - 90) < horizontal_threshold:
                    horizontal_edges[i, j] = True
                else:
                    diagonal_edges[i, j] = True

    return vertical_edges, horizontal_edges, diagonal_edges


# 分类边缘像素
vertical_edges, horizontal_edges, diagonal_edges = classify_edge_pixels(gradient_magnitude, gradient_direction,
                                                                        edge_mask)


# 针对检测到的非直边边缘像素，根据其周围的像素值进行插值计算
def interpolate_edges(image, edge_mask, vertical_edges, horizontal_edges, diagonal_edges):
    interpolated_image = image.copy()
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float32) / 9

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if edge_mask[i, j] and not (vertical_edges[i, j] or horizontal_edges[i, j]):
                neighborhood = image[i - 1:i + 2, j - 1:j + 2]
                interpolated_value = np.sum(neighborhood * kernel)
                interpolated_image[i, j] = interpolated_value
    return interpolated_image


# 插值处理
interpolated_image = interpolate_edges(smoothed_image, edge_mask, vertical_edges, horizontal_edges, diagonal_edges)

# 对插值后的边缘区域进行高斯滤波
kernel_size = (5, 5)
sigma = 1.5
blurred_edges = cv2.GaussianBlur(interpolated_image, kernel_size, sigma)

# 融合处理后的边缘像素与非边缘像素
final_image = np.where(edge_mask, blurred_edges, interpolated_image)
end_time = time.time()

# 输出所需时间
elapsed_time = end_time - start_time
print(f"执行 1 次推理所需的时间: {elapsed_time:.2f} 秒")


# 将目标区域设置为不同强度的红色
def set_red_intensity(image, edge_mask, gradient_magnitude):
    red_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    max_gradient = np.max(gradient_magnitude)
    intensity = (gradient_magnitude / max_gradient) * 255

    # 确保 intensity 是二维数组
    intensity = intensity.astype(np.uint8)

    # 设置红色通道
    red_image[:, :, 2] = intensity

    # 在边缘区域设置红色
    # 将红色强度应用到红色通道
    red_image[edge_mask] = (0, 0, 255)  # 设置为纯红色
    red_image[edge_mask, 2] = intensity[edge_mask]  # 调整红色强度

    return red_image


# 设置红色强度
red_final_image = set_red_intensity(final_image, edge_mask, gradient_magnitude)

# 显示处理后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Image', smoothed_image.astype(np.uint8))
cv2.imshow('Gradient Magnitude', (gradient_magnitude * 255 / np.max(gradient_magnitude)).astype(np.uint8))
cv2.imshow('Edge Mask', (edge_mask * 255).astype(np.uint8))
cv2.imshow('Vertical Edges', (vertical_edges * 255).astype(np.uint8))
cv2.imshow('Horizontal Edges', (horizontal_edges * 255).astype(np.uint8))
cv2.imshow('Diagonal Edges', (diagonal_edges * 255).astype(np.uint8))
cv2.imshow('Interpolated Image', interpolated_image.astype(np.uint8))
cv2.imshow('Blurred Edges', blurred_edges.astype(np.uint8))
cv2.imshow('Final Image', final_image.astype(np.uint8))
cv2.imshow('Red Final Image', red_final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
