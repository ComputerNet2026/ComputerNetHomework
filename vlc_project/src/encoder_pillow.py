from PIL import Image
from reedsolo import RSCodec
import struct

class OptTransEncoderPillow:
    def __init__(self, version=1):
        self.version = version
        self.module_size = 6  # 6 像素每模块
        self.matrix_size = 166  # 166×166 模块规格
        self.margin = 4  # 4 模块边距
        self.total_size = self.matrix_size + 2 * self.margin  # 包含边距的总模块数
        self.image_size = self.total_size * self.module_size  # 最终生成 996×996 像素图像
        
        # 数据存储参数
        self.data_per_frame = 2580  # 每帧可存储 2580 字节有效二进制数据
        self.blocks = 12  # 数据区做了 12 块分割设计
        self.data_per_block = 215  # 每块包含 215 字节数据
        self.ecc_per_block = 40  # 每块包含 40 字节纠错码
        self.total_ecc = 480  # 整体搭配 480 字节 RS 纠错码
        self.block_size = self.data_per_block + self.ecc_per_block  # 每块 255 字节
        self.total_data_area = self.blocks * self.block_size  # 12块 × 255 = 3,060 字节
        
        # RS 编码器 - 按块编码
        # 每块独立编码：215字节数据 + 40字节纠错码
        self.rs = RSCodec(self.ecc_per_block)
    
    def _generate_finder_pattern(self):
        pattern = [[0]*7 for _ in range(7)]
        for i in range(7):
            for j in range(7):
                if i == 0 or i == 6 or j == 0 or j == 6:
                    pattern[i][j] = 1  # 黑色边框
                elif 2 <= i <= 4 and 2 <= j <= 4:
                    pattern[i][j] = 1  # 黑色中心
                else:
                    pattern[i][j] = 0  # 白色中间层
        return pattern
    
    def _generate_timing_pattern(self):
        length = self.matrix_size - 14
        timing = [i % 2 for i in range(length)]
        return timing
    
    def _is_control_module(self, i, j):
        MARGIN = 4
        FINDER_SIZE = 7
        CONTROL_START = MARGIN + FINDER_SIZE + 1  # 4+7+1=12
        BLOCK_SIZE = 32
        
        # 副本一的控制块位置
        # 块1: 水平第0行 (12,4) 到 (43,4) → 32模块
        if i == 4 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
            return True
        # 块2: 水平第1行 (12,5) 到 (43,5) → 32模块
        if i == 5 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
            return True
        # 块3: 垂直第0列 (4,12) 到 (4,43) → 32模块
        if j == 4 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
            return True
        # 块4: 垂直第1列 (5,12) 到 (5,43) → 32模块
        if j == 5 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
            return True
        
        # 副本二的控制块位置（右下角定位点）
        # 块1: 水平第161行 (12,161) 到 (43,161)
        if i == 161 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
            return True
        # 块2: 水平第162行 (12,162) 到 (43,162)
        if i == 162 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
            return True
        # 块3: 垂直第161列 (161,12) 到 (161,43)
        if j == 161 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
            return True
        # 块4: 垂直第162列 (162,12) 到 (162,43)
        if j == 162 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
            return True
        
        return False
    
    def _is_data_module(self, i, j):
        # 排除定位点
        if (i < 7 and j < 7) or (i < 7 and j >= (self.matrix_size - 7)) or (i >= (self.matrix_size - 7) and j < 7):
            return False
        # 排除控制区
        if self._is_control_module(i, j):
            return False
        return True
    
    def _encode_control_area(self, data_len, mask_pattern=0, frame_num=0, total_frames=1):
        version = self.version
        
        # 16 字节基础信息
        control_data = bytearray(16)
        control_data[0] = version & 0xFF  # 版本号
        control_data[1] = (frame_num >> 8) & 0xFF  # 帧序号（大端序）
        control_data[2] = frame_num & 0xFF
        control_data[3] = (total_frames >> 8) & 0xFF  # 总帧数（大端序）
        control_data[4] = total_frames & 0xFF
        control_data[5] = (data_len >> 16) & 0xFF  # 本帧实际数据长度（大端序）
        control_data[6] = (data_len >> 8) & 0xFF
        control_data[7] = data_len & 0xFF
        control_data[8] = mask_pattern & 0xFF  # 掩码编号
        control_data[9:14] = b'\x00' * 5  # 预留扩展字段
        
        # CRC16 校验值（大端序）
        crc = 0xFFFF
        for byte in control_data[:14]:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        control_data[14] = (crc >> 8) & 0xFF
        control_data[15] = crc & 0xFF
        
        # 双副本冗余设计
        return control_data * 2
    
    def _write_control_area(self, matrix, control_bits):
        bit_idx = 0
        MARGIN = 4
        FINDER_SIZE = 7
        CONTROL_START = MARGIN + FINDER_SIZE + 1  # 4+7+1=12
        BLOCK_SIZE = 32
        
        # 副本一的控制块
        # 块1: 水平第0行 (12,4) 到 (43,4) → 存储字节0-3
        for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[4][j] = control_bits[bit_idx]
                bit_idx += 1
        
        # 块2: 水平第1行 (12,5) 到 (43,5) → 存储字节4-7
        for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[5][j] = control_bits[bit_idx]
                bit_idx += 1
        
        # 块3: 垂直第0列 (4,12) 到 (4,43) → 存储字节8-11
        for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[i][4] = control_bits[bit_idx]
                bit_idx += 1
        
        # 块4: 垂直第1列 (5,12) 到 (5,43) → 存储字节12-15
        for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[i][5] = control_bits[bit_idx]
                bit_idx += 1
        
        # 重置位索引，开始写入副本二
        bit_idx = 0
        
        # 副本二的控制块（右下角定位点）
        # 块1: 水平第161行 (12,161) 到 (43,161)
        for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[161][j] = control_bits[bit_idx]
                bit_idx += 1
        
        # 块2: 水平第162行 (12,162) 到 (43,162)
        for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[162][j] = control_bits[bit_idx]
                bit_idx += 1
        
        # 块3: 垂直第161列 (161,12) 到 (161,43)
        for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[i][161] = control_bits[bit_idx]
                bit_idx += 1
        
        # 块4: 垂直第162列 (162,12) 到 (162,43)
        for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
            if bit_idx < len(control_bits):
                matrix[i][162] = control_bits[bit_idx]
                bit_idx += 1
    
    def _snake_fill(self, data_bits):
        matrix = [[0]*self.matrix_size for _ in range(self.matrix_size)]
        bit_idx = 0
        # 从图像右下角开始填充，蛇形路径
        for row in range(self.matrix_size - 1, -1, -1):
            if row % 2 == 0:
                # 偶数行从右向左
                cols = range(self.matrix_size - 1, -1, -1)
            else:
                # 奇数行从左向右
                cols = range(self.matrix_size)
            for col in cols:
                if self._is_data_module(row, col) and bit_idx < len(data_bits):
                    matrix[row][col] = data_bits[bit_idx]
                    bit_idx += 1
        return matrix
    
    def _mask_pattern_0(self, i, j):
        return (i + j) % 2 == 0
    
    def _mask_pattern_1(self, i, j):
        return i % 2 == 0
    
    def _mask_pattern_2(self, i, j):
        return j % 3 == 0
    
    def _mask_pattern_3(self, i, j):
        return (i + j) % 3 == 0
    
    def _mask_pattern_4(self, i, j):
        return (i // 2 + j // 3) % 2 == 0
    
    def _mask_pattern_5(self, i, j):
        return ((i * j) % 2) + ((i * j) % 3) == 0
    
    def _mask_pattern_6(self, i, j):
        return (((i * j) % 2) + ((i * j) % 3)) % 2 == 0
    
    def _mask_pattern_7(self, i, j):
        return (((i + j) % 2) + ((i * j) % 3)) % 2 == 0
    
    def _get_mask_func(self, mask_pattern):
        mask_funcs = [
            self._mask_pattern_0,
            self._mask_pattern_1,
            self._mask_pattern_2,
            self._mask_pattern_3,
            self._mask_pattern_4,
            self._mask_pattern_5,
            self._mask_pattern_6,
            self._mask_pattern_7
        ]
        return mask_funcs[mask_pattern % 8]
    
    def _apply_mask(self, matrix, mask_pattern):
        masked = [row[:] for row in matrix]
        mask_func = self._get_mask_func(mask_pattern)
        for i in range(len(masked)):
            for j in range(len(masked[0])):
                # 定位区不参与掩码
                if (i < 7 and j < 7) or (i < 7 and j >= (self.matrix_size - 7)) or (i >= (self.matrix_size - 7) and j < 7):
                    continue
                # 控制区和数据区参与掩码
                if self._is_data_module(i, j) or self._is_control_module(i, j):
                    if mask_func(i, j):
                        masked[i][j] = 1 - masked[i][j]
        return masked
    
    def _calculate_mask_penalty(self, matrix):
        penalty = 0
        
        # 规则1：连续同色模块
        for i in range(self.matrix_size):
            run_length = 1
            for j in range(1, self.matrix_size):
                if matrix[i][j] == matrix[i][j-1]:
                    run_length += 1
                else:
                    if run_length >= 5:
                        penalty += run_length - 2
                    run_length = 1
            if run_length >= 5:
                penalty += run_length - 2
        
        for j in range(self.matrix_size):
            run_length = 1
            for i in range(1, self.matrix_size):
                if matrix[i][j] == matrix[i-1][j]:
                    run_length += 1
                else:
                    if run_length >= 5:
                        penalty += run_length - 2
                    run_length = 1
            if run_length >= 5:
                penalty += run_length - 2
        
        # 规则2：同色方块
        for i in range(self.matrix_size - 1):
            for j in range(self.matrix_size - 1):
                if (matrix[i][j] == matrix[i][j+1] == 
                    matrix[i+1][j] == matrix[i+1][j+1]):
                    penalty += 3
        
        # 规则3：特定模式
        # 这里可以添加更多特定模式的检测
        
        # 规则4：黑白比例
        dark_modules = 0
        total_modules = 0
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if self._is_data_module(i, j) or self._is_control_module(i, j):
                    total_modules += 1
                    if matrix[i][j] == 1:  # 1 为黑色块
                        dark_modules += 1
        
        if total_modules > 0:
            ratio = dark_modules / total_modules
            deviation = abs(ratio - 0.5)
            penalty += int(deviation * 100)
        
        return penalty
    
    def _encode_data(self, data):
        data_len = len(data)
        # 分帧处理，按 2580 字节每帧分割
        if data_len > self.data_per_frame:
            raise ValueError(f"数据长度超过每帧最大容量 {self.data_per_frame} 字节")
        
        # 不足部分补 0
        padded_data = data + b'\x00' * (self.data_per_frame - data_len)
        
        # 分块编码：每块 215 字节数据 + 40 字节纠错码
        encoded_data = bytearray()
        for i in range(self.blocks):
            start = i * self.data_per_block
            end = start + self.data_per_block
            block_data = padded_data[start:end]
            # 对每块数据进行 RS 编码
            encoded_block = self.rs.encode(block_data)
            encoded_data.extend(encoded_block)
        
        return encoded_data
    
    def encode_data(self, data, output_image, frame_num=0, total_frames=1):
        # 1. 创建空白图像框架
        matrix = [[0]*self.matrix_size for _ in range(self.matrix_size)]
        
        # 2. 放置定位区
        finder = self._generate_finder_pattern()
        # 左上角定位点
        for i in range(7):
            for j in range(7):
                matrix[i][j] = finder[i][j]
        # 右上角定位点
        for i in range(7):
            for j in range(7):
                matrix[i][self.matrix_size-7+j] = finder[i][j]
        # 左下角定位点
        for i in range(7):
            for j in range(7):
                matrix[self.matrix_size-7+i][j] = finder[i][j]
        
        # 3. 编码并写入控制区信息
        best_mask = 0
        min_penalty = float('inf')
        
        # 4. 将输入二进制数据分块后做 RS 编码
        data_bytes = self._encode_data(data)
        
        data_bits = []
        for byte in data_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]  # 高位在前
            data_bits.extend(bits)
        
        # 5. 通过蛇形路径将编码后的数据填充至数据区
        data_matrix = self._snake_fill(data_bits)
        
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if self._is_data_module(i, j):
                    matrix[i][j] = data_matrix[i][j]
        
        # 6. 自动选择并应用最优掩码
        for mask_pattern in range(8):
            test_matrix = [row[:] for row in matrix]
            
            control_bytes_test = self._encode_control_area(len(data), mask_pattern, frame_num, total_frames)
            control_bits_test = []
            for byte in control_bytes_test:
                bits = [(byte >> (7 - i)) & 1 for i in range(8)]
                control_bits_test.extend(bits)
            
            test_matrix_control = [row[:] for row in test_matrix]
            self._write_control_area(test_matrix_control, control_bits_test)
            
            masked_test = self._apply_mask(test_matrix_control, mask_pattern)
            penalty = self._calculate_mask_penalty(masked_test)
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_mask = mask_pattern
        
        # 写入最优掩码的控制区
        control_bytes = self._encode_control_area(len(data), mask_pattern=best_mask, frame_num=frame_num, total_frames=total_frames)
        control_bits = []
        for byte in control_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            control_bits.extend(bits)
        self._write_control_area(matrix, control_bits)
        
        # 应用最优掩码
        matrix = self._apply_mask(matrix, best_mask)
        
        # 7. 根据模块大小渲染生成最终图像
        # 添加边距
        padded = [[1]*self.total_size for _ in range(self.total_size)]
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                padded[self.margin+i][self.margin+j] = matrix[i][j]
        
        # 创建图像
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        pixels = img.load()
        
        for i in range(self.total_size):
            for j in range(self.total_size):
                # 0 为白色块、1 为黑色块
                color = (0, 0, 0) if padded[i][j] == 1 else (255, 255, 255)
                for y in range(i * self.module_size, (i + 1) * self.module_size):
                    for x in range(j * self.module_size, (j + 1) * self.module_size):
                        pixels[x, y] = color
        
        img.save(output_image)
        return img
    
    def encode_file(self, input_file, output_image):
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # 分帧处理
        total_frames = (len(data) + self.data_per_frame - 1) // self.data_per_frame
        
        if total_frames == 1:
            # 单帧处理
            return self.encode_data(data, output_image, frame_num=0, total_frames=1)
        else:
            # 多帧处理
            for i in range(total_frames):
                start = i * self.data_per_frame
                end = min((i + 1) * self.data_per_frame, len(data))
                frame_data = data[start:end]
                frame_output = f"{output_image.rsplit('.', 1)[0]}_frame{i}.{output_image.rsplit('.', 1)[1]}"
                self.encode_data(frame_data, frame_output, frame_num=i, total_frames=total_frames)
            return total_frames
