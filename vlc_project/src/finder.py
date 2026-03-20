import cv2
import numpy as np
from .config import MARGIN, FINDER_SIZE, SMALL_FINDER_SIZE, MATRIX_SIZE, MODULE_SIZE

def find_finder_corners(img):
    h, w = img.shape
    
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    all_candidates = []
    
    for invert in [False, True]:
        working_img = 255 - img_blur if invert else img_blur
        
        _, img_thresh = cv2.threshold(working_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 100 or area > (w * h) // 8:
                    continue
                
                has_child = hierarchy[0][i][2] != -1
                has_parent = hierarchy[0][i][3] != -1
                
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                rect_area = w_rect * h_rect
                if rect_area == 0:
                    continue
                
                solidity = area / rect_area
                aspect_ratio = float(w_rect) / h_rect
                
                if 0.5 < solidity < 1.5 and 0.5 < aspect_ratio < 2.0:
                    if has_child or has_parent:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            all_candidates.append((cx, cy, area))
                        
                        cx_bb = x + w_rect // 2
                        cy_bb = y + h_rect // 2
                        all_candidates.append((cx_bb, cy_bb, area))
    
    unique_candidates = []
    min_dist = min(w, h) // 40
    
    for cand in all_candidates:
        duplicate = False
        for uc in unique_candidates:
            dist = ((cand[0] - uc[0]) ** 2 + (cand[1] - uc[1]) ** 2) ** 0.5
            if dist < min_dist:
                duplicate = True
                break
        if not duplicate:
            unique_candidates.append(cand)
    
    if len(unique_candidates) < 4:
        return None
    
    unique_candidates.sort(key=lambda x: -x[2])
    
    candidate_points = [c[:2] for c in unique_candidates[:12]]
    
    best_four = None
    best_score = -1
    
    from itertools import combinations
    
    for four_points in combinations(candidate_points, 4):
        points = list(four_points)
        
        tl = min(points, key=lambda p: p[0] + p[1])
        tr = min(points, key=lambda p: -p[0] + p[1])
        bl = min(points, key=lambda p: p[0] - p[1])
        br = max(points, key=lambda p: p[0] + p[1])
        
        found = set()
        for p in [tl, tr, bl, br]:
            for cand_p in four_points:
                if abs(cand_p[0] - p[0]) < 5 and abs(cand_p[1] - p[1]) < 5:
                    found.add(cand_p)
                    break
        
        if len(found) < 4:
            continue
        
        def dist(p1, p2):
            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
        
        top_len = dist(tl, tr)
        bottom_len = dist(bl, br)
        left_len = dist(tl, bl)
        right_len = dist(tr, br)
        
        side_ratio1 = min(top_len, bottom_len) / max(top_len, bottom_len) if max(top_len, bottom_len) > 0 else 0
        side_ratio2 = min(left_len, right_len) / max(left_len, right_len) if max(left_len, right_len) > 0 else 0
        
        aspect = max(top_len, left_len) / min(top_len, left_len) if min(top_len, left_len) > 0 else 100
        
        score = side_ratio1 + side_ratio2
        
        if 0.7 < side_ratio1 < 1.3 and 0.7 < side_ratio2 < 1.3 and 0.5 < aspect < 2.0:
            if score > best_score:
                best_score = score
                best_four = (tl, tr, bl, br)
    
    if best_four is not None:
        return best_four
    
    points = [c[:2] for c in unique_candidates[:8]]
    tl = min(points, key=lambda p: p[0] + p[1])
    tr = min(points, key=lambda p: -p[0] + p[1])
    bl = min(points, key=lambda p: p[0] - p[1])
    br = max(points, key=lambda p: p[0] + p[1])
    
    return (tl, tr, bl, br)
