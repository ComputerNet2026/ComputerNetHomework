import cv2
import numpy as np


def detect_finder_patterns(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 轻度预处理（不过度）
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    for invert in [False, True]:
        working = 255 - gray if invert else gray

        _, binary = cv2.threshold(
            working, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            child = hierarchy[i][2]
            parent = hierarchy[i][3]

            if child != -1 and parent != -1:
                rect = cv2.boundingRect(contours[parent])
                x, y, w, h = rect

                if h == 0:
                    continue

                ratio = w / float(h)

                # 方形检测（放宽）
                if 0.6 < ratio < 1.4:
                    cx = x + w // 2
                    cy = y + h // 2

                    candidates.append((cx, cy, w * h))

    if len(candidates) < 3:
        return None

    #  去重（关键）
    unique = []
    for c in candidates:
        cx, cy, area = c
        duplicate = False

        for u in unique:
            dist = np.hypot(cx - u[0], cy - u[1])
            if dist < 20:
                duplicate = True
                break

        if not duplicate:
            unique.append(c)

    if len(unique) < 3:
        return None

    #  选面积最大的6个候选
    unique.sort(key=lambda x: -x[2])
    points = [u[:2] for u in unique[:6]]

    #  从中选最像“L形”的3个点（核心）
    best = None
    best_score = -1

    from itertools import combinations

    for combo in combinations(points, 3):
        A, B, C = combo

        def dist(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(p2))

        dAB = dist(A, B)
        dBC = dist(B, C)
        dCA = dist(C, A)

        dists = sorted([dAB, dBC, dCA])

        # L形特征：两短边 + 一长边
        score = dists[0] + dists[1] - dists[2]

        if score > best_score:
            best_score = score
            best = combo

    if best is None:
        return None

    return _sort_points(best)


def _sort_points(pts):
    A, B, C = pts

    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    dAB = dist(A, B)
    dBC = dist(B, C)
    dCA = dist(C, A)

    if dAB > dBC and dAB > dCA:
        top_left = C
        p1, p2 = A, B
    elif dBC > dAB and dBC > dCA:
        top_left = A
        p1, p2 = B, C
    else:
        top_left = B
        p1, p2 = A, C

    if p1[0] < p2[0]:
        bottom_left = p1
        top_right = p2
    else:
        bottom_left = p2
        top_right = p1

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left
    }