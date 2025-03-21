import numpy as np
import cv2


def validate_image(img):
    """校验输入图像是否为灰度图像"""
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("输入图像必须是灰度图像且类型为uint8")


def anisotropic_diffusion(img, iterations=30, k=100, lambda_=0.2):
    """
    各向异性扩散滤波实现
    :param img: 输入灰度图像（0-255）
    :param iterations: 迭代次数
    :param k: 扩散系数阈值，控制边缘敏感度
    :param lambda_: 时间步长（0.1-0.25）
    :return: 滤波后图像
    """
    validate_image(img)
    img = img.astype(np.float32)
    padded = np.pad(img, 1, mode='wrap')  # 边界对称填充

    for _ in range(iterations):
        grad_N = padded[:-2, 1:-1] - img
        grad_S = padded[2:, 1:-1] - img
        grad_E = padded[1:-1, 2:] - img
        grad_W = padded[1:-1, :-2] - img

        cN = np.exp(-(grad_N ** 2) / (k ** 2))
        cS = np.exp(-(grad_S ** 2) / (k ** 2))
        cE = np.exp(-(grad_E ** 2) / (k ** 2))
        cW = np.exp(-(grad_W ** 2) / (k ** 2))

        delta = lambda_ * (cN * grad_N + cS * grad_S + cE * grad_E + cW * grad_W)
        img += delta
        img = np.clip(img, 0, 255)

    return img.astype(np.uint8)
#
#
# #更高级的梯度计算方式
#
# def anisotropic_diffusion(img, iterations=10, k=5, lambda_=0.1):
#     validate_image(img)
#     img = img.astype(np.float32)
#     padded = np.pad(img, 1, mode='wrap')
#
#     # Sobel 算子计算梯度
#     sobel_x = cv2.Sobel(padded, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(padded, cv2.CV_64F, 0, 1, ksize=3)
#
#     for _ in range(iterations):
#         grad_N = sobel_y[:-2, 1:-1]
#         grad_S = sobel_y[2:, 1:-1]
#         grad_E = sobel_x[1:-1, 2:]
#         grad_W = sobel_x[1:-1, :-2]
#
#         cN = np.exp(-(grad_N ** 2) / (k ** 2))
#         cS = np.exp(-(grad_S ** 2) / (k ** 2))
#         cE = np.exp(-(grad_E ** 2) / (k ** 2))
#         cW = np.exp(-(grad_W ** 2) / (k ** 2))
#
#         delta = lambda_ * (cN * grad_N + cS * grad_S + cE * grad_E + cW * grad_W)
#         img += delta
#         img = np.clip(img, 0, 255)
#
#     return img.astype(np.uint8)


#中心差分法
# def anisotropic_diffusion(img, iterations=300, k=2, lambda_=0.2):
#     validate_image(img)
#     img = img.astype(np.float32)
#     padded = np.pad(img, 1, mode='wrap')
#
#     for _ in range(iterations):
#         grad_N = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
#         grad_S = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
#         grad_E = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
#         grad_W = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
#
#         cN = np.exp(-(grad_N ** 2) / (k ** 2))
#         cS = np.exp(-(grad_S ** 2) / (k ** 2))
#         cE = np.exp(-(grad_E ** 2) / (k ** 2))
#         cW = np.exp(-(grad_W ** 2) / (k ** 2))
#
#         delta = lambda_ * (cN * grad_N + cS * grad_S + cE * grad_E + cW * grad_W)
#         img += delta
#         img = np.clip(img, 0, 255)
#
#     return img.astype(np.uint8)

def clahe(img, tile_size=128, clip_limit=1.0):
    """
    CLAHE实现（分块直方图均衡化）
    :param img: 输入灰度图像（0-255）
    :param tile_size: 分块大小（建议8-16）
    :param clip_limit: 对比度限制阈值（1-4）
    :return: 增强后图像
    """
    validate_image(img)
    tiles_x = (img.shape[1] + tile_size - 1) // tile_size
    tiles_y = (img.shape[0] + tile_size - 1) // tile_size
    result = np.zeros_like(img, dtype=np.uint16)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x_start = tx * tile_size
            y_start = ty * tile_size
            x_end = min(x_start + tile_size, img.shape[1])
            y_end = min(y_start + tile_size, img.shape[0])
            tile = img[y_start:y_end, x_start:x_end]

            hist, _ = np.histogram(tile, bins=256, range=(0, 255))
            excess = np.sum(np.maximum(hist - clip_limit, 0))
            hist = np.clip(hist + excess // 256, 0, clip_limit)

            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            tile_eq = np.interp(tile.flatten(), np.arange(256), cdf).reshape(tile.shape)
            result[y_start:y_end, x_start:x_end] = tile_eq

    return np.clip(result, 0, 255).astype(np.uint8)


def morph_erode(img, kernel_size=3):
    """腐蚀操作"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel)


def morph_dilate(img, kernel_size=3):
    """膨胀操作"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel)


def top_hat(img, kernel_size=3):
    """顶帽变换提取亮边缘"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img - opened


def closing(img, kernel_size=3):
    """闭运算"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# 主流程
if __name__ == "__main__":
    img = cv2.imread('../images/827.jpg', 1)  # 读取彩色图像
    if img is None:
        raise FileNotFoundError("图像文件未找到")

    # 各向异性扩散去噪
    denoised_AD = anisotropic_diffusion(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cv2.imshow('denoised_AD', denoised_AD)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # CLAHE对比度增强
    enhanced = clahe(denoised_AD, tile_size=64, clip_limit=1)
    cv2.imshow('enhanced', enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 腐蚀操作
    edge = morph_erode(enhanced, kernel_size=7)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # 合并结果
    final = np.clip(0.8 * denoised_AD + 0.1 * enhanced + 0.3 * edge, 0, 255).astype(np.uint8)
    cv2.imshow('final', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


