"""
OptTrans编码器模块
"""

class OptTransEncoder:
    """
    优化的编码器类
    """
    def __init__(self, version=1):
        # 在初始化时导入配置并存储为实例变量
        from .config import (
            MODULE_SIZE, MATRIX_SIZE, MARGIN, TOTAL_SIZE,
            DATA_PER_FRAME, BLOCKS, DATA_PER_BLOCK, ECC_PER_BLOCK, BLOCK_SIZE,
            FINDER_SIZE, SMALL_FINDER_SIZE
        )
        from reedsolo import RSCodec
        
        self.version = version
        self.module_size = MODULE_SIZE
        self.matrix_size = MATRIX_SIZE
        self.margin = MARGIN
        self.total_size = TOTAL_SIZE
        self.image_size = self.total_size * self.module_size
        
        self.data_per_frame = DATA_PER_FRAME
        self.blocks = BLOCKS
        self.data_per_block = DATA_PER_BLOCK
        self.ecc_per_block = ECC_PER_BLOCK
        self.block_size = self.data_per_block + self.ecc_per_block
        
        # 存储配置参数为实例变量，避免Pylint未定义变量错误
        self.FINDER_SIZE = FINDER_SIZE
        self.SMALL_FINDER_SIZE = SMALL_FINDER_SIZE
        
        self.rs = RSCodec(self.ecc_per_block)

    def _create_base_matrix(self):
        """
        创建带定位点的基础矩阵
        """
        from .patterns import generate_finder_pattern, generate_small_finder_pattern
        
        matrix = [[0]*self.matrix_size for _ in range(self.matrix_size)]
        
        # 生成定位点图案
        finder = generate_finder_pattern(size=self.FINDER_SIZE)
        small_finder = generate_small_finder_pattern()
        
        # 添加定位点
        # 左上角
        for i in range(self.FINDER_SIZE):
            for j in range(self.FINDER_SIZE):
                matrix[i][j] = finder[i][j]
        # 右上角
        for i in range(self.FINDER_SIZE):
            for j in range(self.FINDER_SIZE):
                matrix[i][self.matrix_size-self.FINDER_SIZE+j] = finder[i][j]
        # 左下角
        for i in range(self.FINDER_SIZE):
            for j in range(self.FINDER_SIZE):
                matrix[self.matrix_size-self.FINDER_SIZE+i][j] = finder[i][j]
        # 右下角（小型定位点）
        for i in range(self.SMALL_FINDER_SIZE):
            for j in range(self.SMALL_FINDER_SIZE):
                matrix[self.matrix_size-self.SMALL_FINDER_SIZE+i][self.matrix_size-self.SMALL_FINDER_SIZE+j] = small_finder[i][j]
        
        return matrix

    def _select_best_mask(self, base_matrix, data_bits, control_bits_test):
        """
        优化的掩码选择算法
        """
        from .control_area import write_control_area
        from .masking import apply_mask, calculate_mask_penalty
        
        best_mask = 0
        min_penalty = float('inf')
        
        # 尝试所有8种掩码模式
        for mask_pattern in range(8):
            test_matrix = [row[:] for row in base_matrix]
            
            # 写入测试控制区域
            test_matrix_control = [row[:] for row in test_matrix]
            write_control_area(test_matrix_control, control_bits_test)
            
            # 应用掩码
            masked_test = apply_mask(test_matrix_control, mask_pattern, self.matrix_size)
            penalty = calculate_mask_penalty(masked_test, self.matrix_size)
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_mask = mask_pattern
        
        return best_mask, min_penalty

    def encode_data(self, data, output_image, frame_num=0, total_frames=1):
        """
        优化的数据编码方法
        """
        from .control_area import encode_control_area, write_control_area
        from .masking import apply_mask
        from .data_codec import DataCodec
        
        # 创建基础矩阵（带定位点）
        base_matrix = self._create_base_matrix()
        
        # 编码数据
        data_bytes = DataCodec.encode_data(data, self.rs)
        
        # 转换为比特流
        data_bits = []
        for byte in data_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            data_bits.extend(bits)
        
        # 填充到矩阵
        data_matrix = DataCodec.snake_fill(data_bits, self.matrix_size)
        
        # 将数据写入基础矩阵的数据模块
        matrix = [row[:] for row in base_matrix]  # 复制基础矩阵
        data_positions = DataCodec.get_data_positions(self.matrix_size)
        
        bit_idx = 0
        for i, j in data_positions:
            if bit_idx < len(data_bits):
                matrix[i][j] = data_bits[bit_idx]
                bit_idx += 1
        
        # 编码控制信息用于掩码选择
        control_bytes_test = encode_control_area(
            self.version, len(data), 0, frame_num, total_frames
        )
        control_bits_test = []
        for byte in control_bytes_test:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            control_bits_test.extend(bits)
        
        # 选择最佳掩码
        best_mask, _ = self._select_best_mask(matrix, data_bits, control_bits_test)
        
        # 写入最终控制信息
        control_bytes = encode_control_area(
            self.version, len(data), best_mask, frame_num, total_frames
        )
        control_bits = []
        for byte in control_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            control_bits.extend(bits)
        write_control_area(matrix, control_bits)
        
        # 应用最佳掩码
        final_matrix = apply_mask(matrix, best_mask, self.matrix_size)
        
        # 创建图像
        padded = [[0]*self.total_size for _ in range(self.total_size)]
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                padded[self.margin+i][self.margin+j] = final_matrix[i][j]
        
        # 导入PIL模块
        from PIL import Image
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        pixels = img.load()
        
        # 绘制图像
        for i in range(self.total_size):
            for j in range(self.total_size):
                color = (0, 0, 0) if padded[i][j] == 1 else (255, 255, 255)
                for y in range(i * self.module_size, (i + 1) * self.module_size):
                    for x in range(j * self.module_size, (j + 1) * self.module_size):
                        pixels[x, y] = color
        
        img.save(output_image)
        return img

    def encode_file(self, input_file, output_image):
        """
        优化的文件编码方法
        """
        with open(input_file, 'rb') as f:
            data = f.read()
        
        total_frames = (len(data) + self.data_per_frame - 1) // self.data_per_frame
        
        if total_frames == 1:
            return self.encode_data(data, output_image, frame_num=0, total_frames=1)
        else:
            results = []
            for i in range(total_frames):
                start = i * self.data_per_frame
                end = min((i + 1) * self.data_per_frame, len(data))
                frame_data = data[start:end]
                frame_output = f"{output_image.rsplit('.', 1)[0]}_frame{i}.{output_image.rsplit('.', 1)[1]}"
                result = self.encode_data(frame_data, frame_output, frame_num=i, total_frames=total_frames)
                results.append(result)
            return results