import cv2
import numpy as np
from PIL import Image

def find_finder_accurate_corners(image_path):
    """精确查找定位点的角点"""
    image = Image.open(image_path)
    img_gray = np.array(image.convert('L'))
    
    # 预处理
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 查找轮廓
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    finder_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or area > (img_gray.shape[0] * img_gray.shape[1]) // 4:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        if len(approx) != 4:
            continue
        
        if not cv2.isContourConvex(approx):
            continue
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            continue
        
        finder_contours.append(contour)
    
    # 按面积排序
    finder_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    
    if len(finder_contours) < 3:
        print("没有找到足够的定位点")
        return None
    
    # 取最大的3个轮廓
    candidates = finder_contours[:3]
    
    # 找到每个轮廓的四个角点
    corner_points = []
    for contour in candidates:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        corner_points.append(box)
    
    # 现在我们需要找到三个定位点的外角点
    # 对于左上角定位点，取它的左上角
    # 对于右上角定位点，取它的右上角
    # 对于左下角定位点，取它的左下角
    
    # 先找到三个定位点的中心，区分位置
    centers = []
    for contour in candidates:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy, contour))
    
    if len(centers) < 3:
        return None
    
    # 根据位置分类
    img_h, img_w = img_gray.shape
    top_left_idx = None
    top_right_idx = None
    bottom_left_idx = None
    
    # 左上角：x和y都最小
    top_left_idx = min(range(3), key=lambda i: centers[i][0] + centers[i][1])
    # 右上角：x最大，y最小
    top_right_idx = min(range(3), key=lambda i: -centers[i][0] + centers[i][1])
    # 左下角：x最小，y最大
    bottom_left_idx = min(range(3), key=lambda i: centers[i][0] - centers[i][1])
    
    # 现在找到每个定位点的精确角点
    def get_extreme_corner(contour, corner_type):
        """获取轮廓的特定角点"""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        if corner_type == 'top_left':
            # 找x和y都最小的点
            return tuple(min(box, key=lambda p: p[0] + p[1]))
        elif corner_type == 'top_right':
            # 找x最大，y最小的点
            return tuple(min(box, key=lambda p: -p[0] + p[1]))
        elif corner_type == 'bottom_left':
            # 找x最小，y最大的点
            return tuple(min(box, key=lambda p: p[0] - p[1]))
        elif corner_type == 'bottom_right':
            # 找x和y都最大的点
            return tuple(max(box, key=lambda p: p[0] + p[1]))
    
    # 获取三个定位点的外角
    tl_corner = get_extreme_corner(centers[top_left_idx][2], 'top_left')
    tr_corner = get_extreme_corner(centers[top_right_idx][2], 'top_right')
    bl_corner = get_extreme_corner(centers[bottom_left_idx][2], 'bottom_left')
    
    # 画出来看看
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # 画定位点轮廓
    for contour in candidates:
        cv2.drawContours(img_color, [contour], -1, (0, 255, 0), 3)
    
    # 画角点
    cv2.circle(img_color, tl_corner, 10, (0, 0, 255), -1)  # 红色
    cv2.circle(img_color, tr_corner, 10, (255, 0, 0), -1)  # 蓝色
    cv2.circle(img_color, bl_corner, 10, (0, 255, 255), -1)  # 黄色
    
    # 保存结果
    output_path = image_path.replace('.png', '_corners.png').replace('.jpg', '_corners.jpg')
    cv2.imwrite(output_path, img_color)
    print(f"角点检测结果已保存到: {output_path}")
    
    return [tl_corner, tr_corner, bl_corner]

# 测试
print("=== 测试 screen.png ===")
corners1 = find_finder_accurate_corners('examples/screen.png')
print(f"找到角点: {corners1}")

print("\n=== 测试 photo.jpg ===")
corners2 = find_finder_accurate_corners('examples/photo.jpg')
print(f"找到角点: {corners2}")
