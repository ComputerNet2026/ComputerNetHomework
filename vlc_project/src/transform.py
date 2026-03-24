import cv2
import numpy as np
from PIL import Image
from .finder import find_finder_corners
from .config import MARGIN, FINDER_SIZE, SMALL_FINDER_SIZE, MATRIX_SIZE, MODULE_SIZE, TOTAL_SIZE

def detect_and_warp(image, module_size=MODULE_SIZE):
    img_gray = np.array(image.convert('L'))
    
    corners = find_finder_corners(img_gray)
    
    if corners is None:
        return None, None
    
    tl_finder, tr_finder, bl_finder, br_finder = corners
    
    src_pts = np.float32([tl_finder, tr_finder, bl_finder, br_finder])
    
    scale = 3
    module_pixel = module_size * scale
    
    dst_tl = ((MARGIN + FINDER_SIZE / 2) * module_pixel, 
              (MARGIN + FINDER_SIZE / 2) * module_pixel)
    
    dst_tr = ((MARGIN + MATRIX_SIZE - FINDER_SIZE / 2) * module_pixel, 
              (MARGIN + FINDER_SIZE / 2) * module_pixel)
    
    dst_bl = ((MARGIN + FINDER_SIZE / 2) * module_pixel, 
              (MARGIN + MATRIX_SIZE - FINDER_SIZE / 2) * module_pixel)
    
    dst_br = ((MARGIN + MATRIX_SIZE - SMALL_FINDER_SIZE / 2) * module_pixel, 
              (MARGIN + MATRIX_SIZE - SMALL_FINDER_SIZE / 2) * module_pixel)
    
    dst_pts = np.float32([dst_tl, dst_tr, dst_bl, dst_br])
    
    output_size = int(TOTAL_SIZE * module_pixel)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_gray, M, (output_size, output_size), flags=cv2.INTER_LANCZOS4)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    warped = cv2.filter2D(warped, -1, kernel)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    warped = clahe.apply(warped)
    
    warped = cv2.GaussianBlur(warped, (3, 3), 0.5)
    
    warped_flipped = 255 - warped
    
    return (Image.fromarray(warped), output_size, scale), (Image.fromarray(warped_flipped), output_size, scale)
