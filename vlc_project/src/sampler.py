import numpy as np
import cv2
from .config import MATRIX_SIZE, MARGIN, TOTAL_SIZE

def sample_modules(image, module_size=8, output_size=None, scale=1, sample_radius_factor=3, threshold_override=None):
    img_width, img_height = image.size
    if output_size is None:
        module_pixel_size = min(img_width, img_height) // TOTAL_SIZE
    else:
        module_pixel_size = module_size * scale
    
    matrix = [[0]*MATRIX_SIZE for _ in range(MATRIX_SIZE)]
    
    img_np = np.array(image.convert('L'))
    
    sample_radius = max(1, module_pixel_size // sample_radius_factor)
    
    if threshold_override is not None:
        threshold = threshold_override
    elif output_size is None:
        threshold = 128
    else:
        all_brightness = []
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                cx = int((MARGIN + j) * module_pixel_size + module_pixel_size // 2)
                cy = int((MARGIN + i) * module_pixel_size + module_pixel_size // 2)
                
                x1 = max(0, cx - sample_radius)
                x2 = min(img_width - 1, cx + sample_radius)
                y1 = max(0, cy - sample_radius)
                y2 = min(img_height - 1, cy + sample_radius)
                
                if x1 <= x2 and y1 <= y2:
                    region = img_np[y1:y2+1, x1:x2+1]
                    avg_brightness = np.median(region)
                    all_brightness.append(avg_brightness)
        
        if len(all_brightness) > 0:
            np_brightness = np.array(all_brightness, dtype=np.uint8)
            threshold, _ = cv2.threshold(np_brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            threshold = 128
    
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            cx = int((MARGIN + j) * module_pixel_size + module_pixel_size // 2)
            cy = int((MARGIN + i) * module_pixel_size + module_pixel_size // 2)
            
            x1 = max(0, cx - sample_radius)
            x2 = min(img_width - 1, cx + sample_radius)
            y1 = max(0, cy - sample_radius)
            y2 = min(img_height - 1, cy + sample_radius)
            
            if x1 <= x2 and y1 <= y2:
                region = img_np[y1:y2+1, x1:x2+1]
                median_brightness = np.median(region)
                matrix[i][j] = 1 if median_brightness < threshold else 0
    
    return matrix
