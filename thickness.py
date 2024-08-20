from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 设置阈值
thresh = 1
# 原始图片路径
root_img_path = './dataset/mask_500.png'
# 预测掩模图片路径
mask_pred_path = './dataset/mask_500.png'

# 加载并处理预测掩模图片
img = Image.open(mask_pred_path).convert('RGB')
print(img.size)
# 确保图片大小为512x512
# if img.size != (600, 600):
#     img = img.resize((600, 600))
# 转换为numpy数组
# img = np.reshape(img, (600, 600, 3))
img = np.array(img)

# 转为灰度图
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化处理
ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)


# 绘制直线函数
def drawLine(img, Vx, Vy, Px, Py):
    _, cols = img.shape[:2]
    lefty = int((-Px * Vy / Vx) + Py)
    righty = int(((cols - Px) * Vy / Vx) + Py)
    cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 1)


# 查找并排序轮廓
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 拟合直线
fit_line_params = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
# 计算垂线参数
percentage_line_vx, percentage_line_vy = -fit_line_params[1], fit_line_params[0]

# 初始化掩模图像
img_contours_mask = np.zeros(img.shape, np.uint8)
contour_img = np.zeros(img.shape, np.uint8)

# 绘制轮廓
cv2.drawContours(img_contours_mask, contours, -1, (0, 255, 0), 1)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
# 显示图像
plt.imshow(img_contours_mask)
plt.show()

# 计算质心
centroid = []
for contour in contours:
    moments = cv2.moments(contour)
    centroid.append(int(moments['m10'] / moments['m00']))
    centroid.append(int(moments['m01'] / moments['m00']))
    x, y, w, h = cv2.boundingRect(contour)
    break
print(centroid)

# 在掩模上画出质心
cv2.circle(img_contours_mask, (centroid[0], centroid[1]), 2, (255, 0, 0), -1)
plt.imshow(img_contours_mask)
plt.show()

# 绘制多个图像比较
fig, axes = plt.subplots(1, 3, figsize=(15, 15))
axes[0].imshow(img)
axes[0].set_title('mask')
axes[1].imshow(contour_img)
axes[1].set_title('binary')
axes[2].imshow(img_contours_mask)
axes[2].set_title('center_line')

# 创建线掩模并画线
line_mask = np.zeros(img.shape, np.uint8)
drawLine(line_mask, percentage_line_vx, percentage_line_vy, centroid[0], centroid[1])
plt.imshow(line_mask)
plt.show()

# 转换类型以进行逻辑运算
img_contours_mask = np.array(img_contours_mask)
line_mask = np.array(line_mask)

# 找到交点
key_points = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img_contours_mask[i][j][1] == 255 and line_mask[i][j][1] == 255:
            key_points.append([i, j])

print('数量: ', len(key_points))
print('交点坐标: ', key_points)

# 交点可视化
if len(key_points) > 1:
    cv2.circle(img_contours_mask, (centroid[0], centroid[1]), 2, (255, 0, 0), -1)
    cv2.circle(img_contours_mask, (key_points[0][1], key_points[0][0]), 1, (0, 255, 0), -1)
    cv2.circle(img_contours_mask, (key_points[1][1], key_points[1][0]), 1, (0, 0, 255), -1)
else:
    print("未找到足够的交点。")
plt.imshow(img_contours_mask)
plt.show()

# 画线
drawLine(img_contours_mask, fit_line_params[0], fit_line_params[1], fit_line_params[2], fit_line_params[3])
plt.imshow(img_contours_mask)
plt.show()

# 画垂直线与质心
drawLine(img_contours_mask, percentage_line_vx, percentage_line_vy, centroid[0], centroid[1])
# 交点可视化
cv2.circle(img_contours_mask, (centroid[0], centroid[1]), 2, (255, 0, 0), -1)
cv2.circle(img_contours_mask, (key_points[0][1], key_points[0][0]), 1, (0, 255, 0), -1)
cv2.circle(img_contours_mask, (key_points[1][1], key_points[1][0]), 1, (0, 0, 255), -1)
plt.imshow(img_contours_mask)
plt.show()

# 提取感兴趣区域
ROI = img_contours_mask[y:y + h, x:x + w]
plt.imshow(ROI)
plt.show()

# 读取原始图片并处理
root_img = Image.open(root_img_path)
#root_img = root_img.resize((512, 512))
root_img = cv2.imread(root_img_path)
gt_mask = np.zeros(img.shape, np.uint8)

# 绘制mask上的信息
cv2.drawContours(gt_mask, contours, -1, (0, 255, 0), 1)
cv2.circle(gt_mask, (key_points[0][1], key_points[0][0]), 2, (255, 0, 0), -1)
cv2.circle(gt_mask, (key_points[1][1], key_points[1][0]), 2, (255, 0, 0), -1)
cv2.line(gt_mask, (key_points[0][1], key_points[0][0]), (key_points[1][1], key_points[1][0]), (0, 0, 255), 1)

# 绘制root_img上的信息
cv2.drawContours(root_img, contours, -1, (0, 255, 0), 1)
cv2.circle(root_img, (key_points[0][1], key_points[0][0]), 2, (128, 128, 0), -1)
cv2.circle(root_img, (key_points[1][1], key_points[1][0]), 2, (0, 128, 128), -1)
cv2.line(root_img, (key_points[0][1], key_points[0][0]), (key_points[1][1], key_points[1][0]), (0, 0, 255), 1)

# print(key_points)
# print('done!')

fig, arr = plt.subplots(1, 2, figsize=(15, 15))
arr[0].imshow(img_contours_mask)
arr[1].imshow(gt_mask)
# arr[2].imshow(root_img)

DPI = [200, 250, 300, 350]

inch_to_mm = 25.4


def mDistance(x1, y1, x2, y2):
    """
    计算两个点之间的物理距离。

    该函数通过考虑DPI（每英寸点数）和单位转换（英寸到毫米），来计算屏幕上两个点之间的实际物理距离。
    这对于需要对屏幕上的距离进行精确测量的情况非常有用，比如在图形用户界面设计中。

    参数:
    x1 (float): 第一个点的x坐标。
    y1 (float): 第一个点的y坐标。
    x2 (float): 第二个点的x坐标。
    y2 (float): 第二个点的y坐标。

    返回:
    float: 两个点之间的物理距离。
    """
    # 根据DPI对坐标进行缩放，并计算两点间的距离
    return ((((x1 - x2) / DPI[0]) ** 2 + ((y1 - y2) / DPI[0]) ** 2) ** 0.5) * inch_to_mm


(x1, y1, x2, y2) = (key_points[0][0], key_points[0][1], key_points[1][0], key_points[1][1])

nt1 = round(mDistance(x1, y1, x2, y2), 3)
print('nt1: ', nt1, 'mm')

x = round((x1 + x2) / 2)
y = round((y1 + y2) / 2)
image = cv2.putText(gt_mask, str(nt1) + ' mm', (y + 10, 256 - x),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2, cv2.LINE_AA)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
