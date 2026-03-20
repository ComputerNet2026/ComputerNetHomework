import cv2
import numpy as np
from PIL import Image
import reedsolo
import os
from typing import Optional, List, Tuple, Dict, Any, Union

def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    poly = 0xA001
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return crc

class OptTransDecoderPillow:
    def __init__(self, blocks: int = 12, data_per_block: int = 215, rs_ecc_symbols: int = 40,
                 module_size: int = 6, matrix_size: int = 166):
        self.blocks = blocks
        self.data_per_block = data_per_block
        self.rs_ecc_symbols = rs_ecc_symbols
        self.matrix_size = 166
        self.margin = 4
        self.total_size = 174
        self.finder_size = 7

        # 帧间跟踪：上一次成功检测的定位点 (三个点，格式 [(x1,y1), (x2,y2), (x3,y3)])
        self.last_points = None

        try:
            self.rs = reedsolo.RSCodec(rs_ecc_symbols)
        except Exception as e:
            print(f"⚠️ ReedSolomon 初始化警告：{e}")
            self.rs = None

    @staticmethod
    def _adaptive_binarize(image: np.ndarray) -> np.ndarray:
        """将彩色/灰度图转换为二值图 (0/255)，使用自适应阈值"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # 自适应阈值，块大小51，常数C=10（可根据实际调整）
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 51, 10)
        return binary

    def decode_image(self, image_source: Union[str, Image.Image, np.ndarray]) -> Optional[bytes]:
        """
        解码图像，返回原始数据（兼容旧接口，仅返回数据）
        """
        result = self.decode_image_with_info(image_source)
        return result[0] if result else None

    def decode_image_with_info(self, image_source: Union[str, Image.Image, np.ndarray]) -> Optional[Tuple[bytes, int, int]]:
        """
        解码图像，返回 (数据, 当前帧序号, 总帧数)
        """
        try:
            # 1. 输入统一转换为 numpy 二值图 (0/255)
            if isinstance(image_source, str):
                if not os.path.exists(image_source):
                    return None
                # 用 OpenCV 读取以支持自适应二值化
                img_cv = cv2.imread(image_source)
                if img_cv is None:
                    return None
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                binary = self._adaptive_binarize(img_cv)
            elif isinstance(image_source, Image.Image):
                # PIL 图像转 numpy
                img_np = np.array(image_source.convert('RGB'))
                binary = self._adaptive_binarize(img_np)
            elif isinstance(image_source, np.ndarray):
                binary = self._adaptive_binarize(image_source)
            else:
                raise TypeError("不支持的输入类型")

            # 2. 透视校正
            warped_matrix, points_used = self._perspective_warp_robust(binary)
            matrix = None
            use_warped = False

            if warped_matrix is not None:
                print("✅ 检测到定位点，已应用动态透视矫正！")
                matrix = 1 - (warped_matrix > 127).astype(int)  # 转为 0/1，1=黑
                matrix = matrix[self.margin:self.margin+self.matrix_size,
                                self.margin:self.margin+self.matrix_size]
                use_warped = True
                self.last_points = points_used  # 保存本次定位点
                print("  已裁剪边距，获得 166x166 数据矩阵")
            else:
                # 透视校正失败，尝试使用上一次定位点（如果有）
                if self.last_points is not None:
                    print("⚠️ 当前帧定位失败，尝试使用上一帧定位点进行校正...")
                    warped_matrix = self._warp_with_prior_points(binary, self.last_points)
                    if warped_matrix is not None:
                        matrix = 1 - (warped_matrix > 127).astype(int)
                        matrix = matrix[self.margin:self.margin+self.matrix_size,
                                        self.margin:self.margin+self.matrix_size]
                        use_warped = True
                        print("  使用上一帧定位点校正成功")
                    else:
                        print("  使用上一帧定位点校正失败")
                if not use_warped:
                    print("⚠️ 未检测到定位点，回退到传统尺寸匹配模式...")
                    # 原有回退逻辑（略作调整，直接 resize 到 174x174）
                    h, w = binary.shape
                    if w == 1044:
                        current_margin = 4
                        target_w = 174
                    elif w == 996:
                        current_margin = 0
                        target_w = 166
                    else:
                        print(f"⚠️ 未知尺寸 {w}x{h}，尝试按 174 模块处理")
                        # 缩放至 174x174（注意最近邻插值保持二值性）
                        binary = cv2.resize(binary, (174, 174), interpolation=cv2.INTER_NEAREST)
                        current_margin = 4
                        target_w = 174
                        w, h = 174, 174

                    if w != target_w:
                        binary = cv2.resize(binary, (target_w, target_w), interpolation=cv2.INTER_NEAREST)
                        w, h = target_w, target_w

                    matrix = 1 - (binary > 127).astype(int)

                    if current_margin > 0:
                        m = current_margin
                        if matrix.shape[0] >= m+166 and matrix.shape[1] >= m+166:
                            matrix = matrix[m:m+166, m:m+166]
                        else:
                            print("⚠️ 裁剪越界，使用全图")

            # 3. 解析控制区，获取掩码和数据长度
            ctrl_info = self._parse_control_robust(matrix)
            best_mask = None
            data_len = None
            frame_num = 0
            total_frames = 1

            if ctrl_info:
                best_mask = ctrl_info['mask_pattern']
                data_len = ctrl_info['data_len']
                frame_num = ctrl_info.get('frame_num', 0)
                total_frames = ctrl_info.get('total_frames', 1)
                print(f"🔑 控制区解析成功：掩码={best_mask}, 长度={data_len}, 帧={frame_num}/{total_frames}")
            else:
                print("⚠️ 控制区解析失败，启动智能盲测...")
                best_mask, data_len, frame_num, total_frames = self._blind_test_mask_full(matrix)

                if best_mask is None and use_warped:
                    print("🔄 透视模式（已裁剪）失败，尝试使用原始 warped 矩阵（含边距）并重新裁剪为 166x166...")
                    matrix_retry = 1 - (warped_matrix > 127).astype(int)
                    matrix_retry = matrix_retry[self.margin:self.margin+self.matrix_size,
                                                self.margin:self.margin+self.matrix_size]
                    best_mask, data_len, frame_num, total_frames = self._blind_test_mask_full(matrix_retry)
                    if best_mask is not None:
                        matrix = matrix_retry
                        print(f"💡 重试成功，掩码={best_mask}, 长度={data_len}")

                if best_mask is None:
                    print("❌ 盲测也失败，尝试强制解码所有掩码并检查头部...")
                    best_mask = self._try_all_masks_find_best(matrix)
                    if best_mask is not None:
                        data_len = 2580
                        frame_num = 0
                        total_frames = 1
                        print(f"🔨 强制使用掩码 {best_mask}，长度暂用默认值 {data_len}")
                    else:
                        print("❌ 所有掩码均无效，解码失败。")
                        return None

            print(f"🎭 应用掩码模式：{best_mask}")
            unmasked_matrix = self._remove_mask(matrix, best_mask)
            data_bits = self._snake_extract(unmasked_matrix)
            data_bytes_list = self._bits_to_bytes(data_bits)
            raw_data = bytes(data_bytes_list)
            print(f"📦 原始提取数据长度：{len(raw_data)} 字节")

            corrected_data = self._apply_reed_solomon(raw_data, data_len)

            if corrected_data:
                print(f"✅ 纠错后数据长度：{len(corrected_data)} 字节")
                return corrected_data, frame_num, total_frames
            else:
                print("⚠️ 纠错失败，返回原始数据（可能有损坏）")
                return raw_data[:data_len] if data_len else raw_data, frame_num, total_frames

        except Exception as e:
            print(f"❌ 解码过程发生异常：{e}")
            import traceback
            traceback.print_exc()
            return None

    def _perspective_warp_robust(self, binary_np: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int]]]]:
        """
        增强版透视校正，基于轮廓层级检测定位点
        返回 (矫正后图像, 检测到的三个点列表)
        """
        h, w = binary_np.shape
        # 查找轮廓，使用 RETR_TREE 获取层级关系
        contours, hierarchy = cv2.findContours(binary_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None, None

        candidates = []  # 存储候选定位点中心坐标

        # 遍历所有轮廓，寻找具有三层嵌套的轮廓（定位点特征）
        for i, (cnt, hr) in enumerate(zip(contours, hierarchy[0])):
            # 最外层轮廓（无父轮廓）
            if hr[3] != -1:
                continue
            area = cv2.contourArea(cnt)
            if area < 50 or area > h * w * 0.05:
                continue
            # 检查是否有子轮廓
            child = hr[2]
            if child == -1:
                continue
            # 检查子轮廓是否有自己的子轮廓（即三层嵌套）
            child_hierarchy = hierarchy[0][child]
            if child_hierarchy[2] == -1:
                continue

            # 计算中心点
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            candidates.append((cx, cy))

        if len(candidates) < 3:
            return None, None

        # 取面积最大的三个（通常定位点面积相近，但以防干扰）
        # 可根据坐标关系进一步筛选，但简单取前三个
        # 这里我们使用距离排序，选取互相距离较远的三个点（真正的定位点应位于三个角）
        # 简单起见，先取所有候选，然后尝试构建三角形
        points = np.array(candidates)
        # 寻找三个点构成面积最大的三角形（大概率是三个定位点）
        best_triangle = None
        max_area = -1
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    tri_pts = points[[i, j, k]]
                    area_tri = 0.5 * abs(np.cross(tri_pts[1]-tri_pts[0], tri_pts[2]-tri_pts[0]))
                    if area_tri > max_area:
                        max_area = area_tri
                        best_triangle = tri_pts

        if best_triangle is None:
            return None, None

        pts = [tuple(p) for p in best_triangle]

        # 确定哪个点是左上角（离原点最近）
        tl = min(pts, key=lambda p: p[0] + p[1])
        others = [p for p in pts if p != tl]

        # 根据与 tl 的向量区分右上和左下
        v1 = (others[0][0] - tl[0], others[0][1] - tl[1])
        v2 = (others[1][0] - tl[0], others[1][1] - tl[1])

        # 判断哪个向量的x更大/更小（假设图像没有严重旋转）
        if abs(v1[0]) > abs(v1[1]):
            tr = others[0]
            bl = others[1]
        else:
            tr = others[1]
            bl = others[0]

        # 计算第四个点（平行四边形法则）
        br_x = bl[0] + (tr[0] - tl[0])
        br_y = bl[1] + (tr[1] - tl[1])
        br = (br_x, br_y)

        src_pts = np.float32([tl, tr, bl, br])
        target_size = self.total_size  # 174
        dst_pts = np.float32([[0, 0], [target_size, 0], [0, target_size], [target_size, target_size]])

        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(binary_np, M, (target_size, target_size), flags=cv2.INTER_NEAREST)
            return warped, pts
        except Exception:
            return None, None

    def _warp_with_prior_points(self, binary_np: np.ndarray, prior_points: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """
        使用上一次检测到的三个点（假设它们代表三个定位点）进行透视校正
        注意：需要将 prior_points 调整为当前帧中的对应位置（可通过模板匹配或光流微调）
        这里简化处理：假设 prior_points 在当前帧中位置不变（仅适用于连续帧微小运动）
        """
        if len(prior_points) != 3:
            return None
        pts = prior_points
        # 重新排序，找出左上角
        tl = min(pts, key=lambda p: p[0] + p[1])
        others = [p for p in pts if p != tl]
        v1 = (others[0][0] - tl[0], others[0][1] - tl[1])
        v2 = (others[1][0] - tl[0], others[1][1] - tl[1])
        if abs(v1[0]) > abs(v1[1]):
            tr = others[0]
            bl = others[1]
        else:
            tr = others[1]
            bl = others[0]
        br_x = bl[0] + (tr[0] - tl[0])
        br_y = bl[1] + (tr[1] - tl[1])
        br = (br_x, br_y)

        src_pts = np.float32([tl, tr, bl, br])
        target_size = self.total_size
        dst_pts = np.float32([[0, 0], [target_size, 0], [0, target_size], [target_size, target_size]])
        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(binary_np, M, (target_size, target_size), flags=cv2.INTER_NEAREST)
            return warped
        except Exception:
            return None

    def _parse_control_robust(self, matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        if matrix.shape[0] < 166 or matrix.shape[1] < 166:
            return None

        def read_copy_with_offset(copy_type, row_offset=0, col_offset=0):
            bits = []
            if copy_type == 'top_left':
                rows_1 = [4 + row_offset, 5 + row_offset]
                cols_1 = list(range(12 + col_offset, 44 + col_offset))
                cols_2 = [4 + col_offset, 5 + col_offset]
                rows_2 = list(range(12 + row_offset, 44 + row_offset))
            else:
                rows_1 = [161 + row_offset, 162 + row_offset]
                cols_1 = list(range(12 + col_offset, 44 + col_offset))
                cols_2 = [161 + col_offset, 162 + col_offset]
                rows_2 = list(range(12 + row_offset, 44 + row_offset))

            if any(r < 0 or r >= 166 for r in rows_1 + rows_2) or \
               any(c < 0 or c >= 166 for c in cols_1 + cols_2):
                return None

            for row in rows_1:
                for col in cols_1:
                    bits.append(matrix[row, col])
            for col in cols_2:
                for row in rows_2:
                    bits.append(matrix[row, col])
            return bits

        for copy_type in ['top_left', 'bottom_right']:
            for row_offset in range(-2, 3):
                for col_offset in range(-2, 3):
                    bits = read_copy_with_offset(copy_type, row_offset, col_offset)
                    if bits is None:
                        continue
                    bytes_data = self._bits_to_bytes(bits)
                    if len(bytes_data) >= 16:
                        crc_calc = _crc16(bytes(bytes_data[:14]))
                        crc_read = (bytes_data[14] << 8) | bytes_data[15]
                        if crc_calc == crc_read:
                            mask_pattern = bytes_data[8] & 0x07
                            data_len = (bytes_data[5] << 16) | (bytes_data[6] << 8) | bytes_data[7]
                            frame_num = (bytes_data[1] << 8) | bytes_data[2]
                            total_frames = (bytes_data[3] << 8) | bytes_data[4]
                            return {
                                'mask_pattern': mask_pattern,
                                'data_len': data_len,
                                'frame_num': frame_num,
                                'total_frames': total_frames
                            }
        return None

    def _blind_test_mask_full(self, matrix: np.ndarray) -> Tuple[Optional[int], Optional[int], int, int]:
        for m in range(8):
            try:
                unmasked = self._remove_mask(matrix, m)
                ctrl_info = self._parse_control_robust(unmasked)
                if ctrl_info:
                    return m, ctrl_info['data_len'], ctrl_info.get('frame_num', 0), ctrl_info.get('total_frames', 1)
            except:
                continue
        return None, None, 0, 1

    def _try_all_masks_find_best(self, matrix: np.ndarray) -> Optional[int]:
        for test_mask in range(8):
            unmasked = self._remove_mask(matrix, test_mask)
            data_bits = self._snake_extract(unmasked)
            data_bytes_list = self._bits_to_bytes(data_bits)
            raw_data = bytes(data_bytes_list)
            if len(raw_data) >= 4:
                if raw_data.startswith(b'\x89PNG') or raw_data.startswith(b'\xff\xd8') or raw_data.startswith(b'%PDF'):
                    return test_mask
        return None

    def _remove_mask(self, matrix: np.ndarray, mask_type: int) -> np.ndarray:
        h, w = matrix.shape
        masked = np.copy(matrix)
        for i in range(h):
            for j in range(w):
                if (i < 7 and j < 7) or (i < 7 and j >= w - 7) or (i >= w - 7 and j < 7):
                    continue
                mask_bit = 0
                if mask_type == 0:
                    mask_bit = 1 if (i + j) % 2 == 0 else 0
                elif mask_type == 1:
                    mask_bit = 1 if i % 2 == 0 else 0
                elif mask_type == 2:
                    mask_bit = 1 if j % 3 == 0 else 0
                elif mask_type == 3:
                    mask_bit = 1 if (i + j) % 3 == 0 else 0
                elif mask_type == 4:
                    mask_bit = 1 if (i // 2 + j // 3) % 2 == 0 else 0
                elif mask_type == 5:
                    mask_bit = 1 if ((i * j) % 2) + ((i * j) % 3) == 0 else 0
                elif mask_type == 6:
                    mask_bit = 1 if (((i * j) % 2) + ((i * j) % 3)) % 2 == 0 else 0
                elif mask_type == 7:
                    mask_bit = 1 if (((i + j) % 2) + ((i * j) % 3)) % 2 == 0 else 0
                if mask_bit:
                    masked[i, j] = 1 - masked[i, j]
        return masked

    def _snake_extract(self, matrix: np.ndarray) -> List[int]:
        bits = []
        h, w = matrix.shape
        for row in range(h - 1, -1, -1):
            if row % 2 == 0:
                cols = range(w - 1, -1, -1)
            else:
                cols = range(w)
            for col in cols:
                if not (self._is_finder_module(row, col, w) or self._is_control_module(row, col)):
                    bits.append(matrix[row, col])
        return bits

    def _is_finder_module(self, i, j, w):
        if (i < 7 and j < 7): return True
        if (i < 7 and j >= w - 7): return True
        if (i >= w - 7 and j < 7): return True
        return False

    def _is_control_module(self, i, j):
        CONTROL_START = 12
        BLOCK_SIZE = 32
        if i == 4 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if i == 5 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if j == 4 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        if j == 5 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        if i == 161 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if i == 162 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE: return True
        if j == 161 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        if j == 162 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE: return True
        return False

    def _bits_to_bytes(self, bits: List[int]) -> List[int]:
        bytes_list = []
        for i in range(0, len(bits), 8):
            chunk = bits[i:i+8]
            if len(chunk) < 8:
                chunk += [0] * (8 - len(chunk))
            byte_val = 0
            for bit in chunk:
                byte_val = (byte_val << 1) | bit
            bytes_list.append(byte_val)
        return bytes_list

    def _apply_reed_solomon(self, raw_data: bytes, expected_len: Optional[int]) -> Optional[bytes]:
        if self.rs is None:
            return raw_data[:expected_len] if expected_len else raw_data

        block_size = self.data_per_block + self.rs_ecc_symbols
        corrected_blocks = []
        total_blocks = len(raw_data) // block_size

        if total_blocks == 0:
            return None

        for i in range(total_blocks):
            start = i * block_size
            end = start + block_size
            block = raw_data[start:end]
            try:
                decoded_result = self.rs.decode(block)
                if isinstance(decoded_result, tuple):
                    decoded_block = decoded_result[0]
                else:
                    decoded_block = decoded_result
                corrected_blocks.append(decoded_block)
            except Exception as e:
                print(f"⚠️ 第 {i} 块 RS 纠错失败：{e}，直接取数据部分")
                corrected_blocks.append(block[:self.data_per_block])

        result = b''.join(corrected_blocks)
        if expected_len:
            return result[:expected_len]
        return result

def detect_file_extension(data: bytes) -> str:
    if data.startswith(b'\x89PNG\r\n\x1a\n'): return '.png'
    elif data.startswith(b'\xff\xd8\xff'): return '.jpg'
    elif data.startswith(b'%PDF'): return '.pdf'
    elif data.startswith(b'GIF8'): return '.gif'
    else: return '.bin'

def decode_to_file(image_path: str, output_path: Optional[str] = None) -> Optional[str]:
    decoder = OptTransDecoderPillow()
    data = decoder.decode_image(image_path)
    if not data:
        print("❌ 解码失败，未生成文件")
        return None
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = detect_file_extension(data)
        output_path = f"restored_{base_name}{ext}"
    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"💾 文件已保存至：{output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        decode_to_file(sys.argv[1])
    else:
        print("用法：python decoder_pillow.py <image_path>")
