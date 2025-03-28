import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import zoom, uniform_filter, gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte


# --------------------- 辅助函数 ---------------------
def validate_image(img):
    """校验输入图像是否为灰度图像"""
    if img is None or img.size == 0:
        raise ValueError("输入图像不能为空")
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("输入图像必须是灰度图像且类型为uint8")


def safe_uint8_conversion(img):
    """安全转换为uint8格式"""
    if img.dtype == np.uint8:
        return img.copy()
    # 归一化到0-255范围后转换
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    return img_as_ubyte(img_normalized)


# --------------------- 核心算法 ---------------------
def guided_filter(I, p, radius=20, eps=1e-3):
    """引导滤波实现"""
    I_uint = safe_uint8_conversion(I)
    p_uint = safe_uint8_conversion(p)

    I_norm = I_uint.astype(np.float32) / 255.0
    p_norm = p_uint.astype(np.float32) / 255.0

    # 计算局部统计量
    mean_I = uniform_filter(I_norm, size=radius)
    mean_p = uniform_filter(p_norm, size=radius)
    mean_Ip = uniform_filter(I_norm * p_norm, size=radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = uniform_filter(I_norm * I_norm, size=radius)
    var_I = mean_II - mean_I * mean_I

    # 计算线性系数
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 平均系数
    mean_a = uniform_filter(a, size=radius)
    mean_b = uniform_filter(b, size=radius)

    # 生成输出
    q = mean_a * I_norm + mean_b
    return np.clip(q * 255, 0, 255).astype(np.uint8)


def morphological_close(img, kernel_size=15):
    """形态学闭运算实现"""
    img_uint = safe_uint8_conversion(img)
    kernel = disk(kernel_size // 2)

    # 执行形态学操作
    dilated = rank.maximum(img_uint, kernel)
    closed = rank.minimum(dilated, kernel)

    # 保持浮点类型后续处理
    return closed.astype(np.float32) if img.dtype == np.float32 else closed


def dehaze_ultrasound(img, omega=0.4, t0=0.2, kernel_size=21):
    """改进的暗通道去雾算法"""
    # 预处理降噪并确保类型安全
    denoised = gaussian_filter(img, sigma=1)
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    # 优化暗通道计算
    dark_channel = morphological_close(denoised, kernel_size)

    # 大气光估计
    bright_thresh = np.percentile(denoised, 99)
    bright_mask = denoised >= bright_thresh
    if np.any(bright_mask):
        atmospheric = np.median(denoised[bright_mask])
    else:
        atmospheric = np.max(denoised)

    # 透射率计算
    transmission = 1 - omega * (dark_channel.astype(float) / (atmospheric + 1e-6))
    transmission = np.clip(transmission, t0, 1)

    # 引导滤波优化
    transmission_refined = guided_filter(
        denoised.astype(np.uint8),
        (transmission * 255).astype(np.uint8)
    )
    transmission_refined = transmission_refined.astype(float) / 255

    # 图像恢复
    recovered = (denoised.astype(float) - atmospheric) / (transmission_refined + 1e-6) + atmospheric
    return np.clip(recovered, 0, 255).astype(np.uint8)


def gaussian_kernel(size=9, sigma=1.0):
    """生成2D高斯核"""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / kernel.sum()


def build_gaussian_pyramid(img, levels=4, cached_kernel=None):
    """构建高斯金字塔"""
    cached_kernel = cached_kernel or gaussian_kernel()
    pyramid = [img]
    for _ in range(levels - 1):
        blurred = convolve2d(pyramid[-1], cached_kernel, mode='same')
        pyramid.append(blurred[::2, ::2])
    return pyramid


def consistency_diffusion(img, iterations=15, k=40, lambda_=0.15):
    """一致性扩散滤波"""
    img = img.astype(np.float32)
    padded = np.pad(img, 1, mode='reflect')

    for _ in range(iterations):
        grad_N = padded[:-2, 1:-1] - img
        grad_S = padded[2:, 1:-1] - img
        grad_E = padded[1:-1, 2:] - img
        grad_W = padded[1:-1, :-2] - img

        cN = np.exp(-np.abs(grad_N) / k)
        cS = np.exp(-np.abs(grad_S) / k)
        cE = np.exp(-np.abs(grad_E) / k)
        cW = np.exp(-np.abs(grad_W) / k)

        delta = lambda_ * (cN * grad_N + cS * grad_S + cE * grad_E + cW * grad_W)
        img += delta

    return np.clip(img, 0, 255).astype(np.uint8)


def black_hole_filling(img, ta=15, t1=15):
    """黑洞填充"""
    img_uint = safe_uint8_conversion(img)

    h = np.array([1, 0, 1]) / 2
    g = np.array([1, 0, 1]) / 2
    a = convolve2d(img_uint, h[np.newaxis, :], mode='same')
    b = convolve2d(img_uint, g[:, np.newaxis], mode='same')
    replacement = np.maximum(a - ta, b - t1)
    return np.where(img_uint < replacement, replacement, img_uint)


def boxfilter(img, radius):
    """高效盒式滤波器实现"""
    rows, cols = img.shape
    im_cum = np.cumsum(img, axis=0)

    # 垂直方向滤波
    dst = np.zeros_like(img)
    dst[0:radius + 1, :] = im_cum[radius:2 * radius + 1, :]
    dst[radius + 1:rows - radius, :] = im_cum[2 * radius + 1:rows, :] - im_cum[0:rows - 2 * radius - 1, :]
    dst[rows - radius:rows, :] = np.tile(im_cum[rows - 1:rows, :], (radius, 1)) - im_cum[
                                                                                  rows - 2 * radius - 1:rows - radius - 1,
                                                                                  :]

    # 水平方向滤波
    im_cum = np.cumsum(dst, axis=1)
    dst[:, 0:radius + 1] = im_cum[:, radius:2 * radius + 1]
    dst[:, radius + 1:cols - radius] = im_cum[:, 2 * radius + 1:cols] - im_cum[:, 0:cols - 2 * radius - 1]
    dst[:, cols - radius:cols] = np.tile(im_cum[:, cols - 1:cols], (1, radius)) - im_cum[:,
                                                                                  cols - 2 * radius - 1:cols - radius - 1]

    return dst


def guided_filter(I, p, radius=1, eps=1e-6):
    """
    I: 引导图像（灰度，0-255）
    p: 输入图像（需滤波图像）
    radius: 滤波半径
    eps: 正则化参数
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    # 计算局部统计量
    mean_I = boxfilter(I, radius) / (radius * 2 + 1) ** 2
    mean_p = boxfilter(p, radius) / (radius * 2 + 1) ** 2
    corr_I = boxfilter(I * I, radius) / (radius * 2 + 1) ** 2
    corr_Ip = boxfilter(I * p, radius) / (radius * 2 + 1) ** 2

    # 计算方差和协方差
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    # 计算线性系数
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 计算系数均值
    mean_a = boxfilter(a, radius) / (radius * 2 + 1) ** 2
    mean_b = boxfilter(b, radius) / (radius * 2 + 1) ** 2

    # 生成输出图像
    q = mean_a * I + mean_b
    return np.clip(q, 0, 255).astype(np.uint8)


def directional_filter(image, axis='x'):
    """方向滤波器"""
    # 定义基础核
    if axis == 'x':
        kernel = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
    elif axis == 'y':
        kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

    # 卷积运算
    filtered = np.zeros_like(image, dtype=np.float32)
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='reflect')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + 2 * pad + 1, j:j + 2 * pad + 1]
            filtered[i, j] = np.sum(region * kernel)

    return np.abs(filtered)


def reconstruct_pyramid(pyramid, cached_kernel=None):
    """金字塔重建"""
    cached_kernel = cached_kernel or gaussian_kernel()
    # 加权融合
    weights = np.linspace(1, len(pyramid), len(pyramid))
    weights /= weights.sum()

    for i in range(4):
        print(f"Layer {i + 1} before fusion: {pyramid[i].shape}")

    # 从金字塔的倒数第二层开始，向上遍历到第一层
    for i in range(len(pyramid) - 2, -1, -1):
        # 对当前层的下一层图像进行2倍放大，order=1表示使用线性插值
        expanded = zoom(pyramid[i + 1], 2, order=1)

        # 对放大的图像进行卷积操作，保持其尺寸不变
        blurred = convolve2d(expanded, cached_kernel, mode='same')

        # 根据权重合并当前层的原始图像和处理后的图像
        pyramid[i] = blurred * weights[i + 1] + pyramid[i] * weights[i]

    # 返回金字塔的顶层图像，即最终处理结果
    return pyramid[0]


# --------------------- 主流程 ---------------------
if __name__ == "__main__":
    try:
        # 1. 读取图像
        img = plt.imread('./data/10.png')

        # 2. 转换为灰度图并归一化
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        img = (img * 255).astype(np.uint8)  # 确保初始为uint8

        validate_image(img)

        # 3. 暗通道去雾
        dehazed = dehaze_ultrasound(img, omega=0.3, kernel_size=5)

        # 4. 一致性扩散降噪
        denoised = consistency_diffusion(dehazed, iterations=5, k=10)

        # 5. 黑洞填充
        filled = black_hole_filling(dehazed, ta=10)

        # 6. 高斯金字塔分解
        pyramid = build_gaussian_pyramid(filled, levels=3)

        plt.figure(figsize=(15, 10))
        for i, layer in enumerate(pyramid):
            print(f"Layer {i + 1} shape: {layer.shape}")
            plt.subplot(2, 3, i + 1)
            plt.imshow(layer, cmap='gray', aspect='auto')
            plt.title(f'Layer {i + 1}')
        plt.tight_layout()
        plt.show()

        # 7. 分层处理
        processed_pyramid = []
        for i, layer in enumerate(pyramid):
            if i == 0:
                # 第一层：先去雾，然后进行扩散滤波，然后再去雾
                layer = dehaze_ultrasound(layer, omega=0.3, t0=0.7)
                # layer = directional_filter(layer, axis='x')
                #layer = equalize_adapthist(layer, clip_limit=0.001)
                layer = consistency_diffusion(layer, iterations=10, k=20)
                layer = dehaze_ultrasound(layer, omega=0.2, t0=0.5)
            elif i == 1:
                # 第二层：先去雾，然后进行扩散滤波
                layer = dehaze_ultrasound(layer, omega=0.5, t0=0.5)
                layer = directional_filter(layer, axis='x')
                layer = consistency_diffusion(layer, iterations=15, k=40)
                layer = equalize_adapthist(layer, clip_limit=0.8)
                layer = guided_filter(layer, layer)
            elif i == 2:
                # layer = directional_filter(layer, axis='x')
                layer = consistency_diffusion(layer, iterations=20, k=60)
                layer = guided_filter(layer, layer)
            # elif i == 3:
            #     # layer = directional_filter(layer, axis='x')
            #     layer = consistency_diffusion(layer, iterations=25, k=80)

            processed_pyramid.append(layer)

        # 8. 金字塔重建
        reconstructed = reconstruct_pyramid(processed_pyramid)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # 9. 后处理增强
        reconstructed_normalized = reconstructed.astype(float) / 255.0
        final_result = equalize_adapthist(reconstructed_normalized, clip_limit=0.01)
        final_result = (final_result * 255).astype(np.uint8)

        # 10. 结果显示
        plt.figure(figsize=(18, 12))
        plt.subplot(231), plt.imshow(img, cmap='gray', aspect='auto'), plt.title('ori_image')
        plt.subplot(232), plt.imshow(dehazed, cmap='gray', aspect='auto'), plt.title('dehaze_image')
        plt.subplot(233), plt.imshow(denoised, cmap='gray', aspect='auto'), plt.title('denosie_image')
        plt.subplot(234), plt.imshow(filled, cmap='gray', aspect='auto'), plt.title('fill_image')
        plt.subplot(235), plt.imshow(reconstructed, cmap='gray', aspect='auto'), plt.title('Fusion_image')
        plt.subplot(236), plt.imshow(final_result, cmap='gray', aspect='auto'), plt.title('result')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"程序运行出错: {e}")
