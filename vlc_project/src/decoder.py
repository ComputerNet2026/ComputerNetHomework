"""
OptTrans解码器模块
"""

class OptTransDecoder:
    """
    优化的解码器类
    """
    def __init__(self):
        # 在初始化时导入配置
        from .config import (
            MODULE_SIZE, MATRIX_SIZE, MARGIN, TOTAL_SIZE,
            DATA_PER_FRAME, BLOCKS, DATA_PER_BLOCK, ECC_PER_BLOCK, BLOCK_SIZE
        )
        from reedsolo import RSCodec
        
        self.module_size = MODULE_SIZE
        self.matrix_size = MATRIX_SIZE
        self.margin = MARGIN
        self.total_size = TOTAL_SIZE
        self.data_per_frame = DATA_PER_FRAME
        self.blocks = BLOCKS
        self.data_per_block = DATA_PER_BLOCK
        self.ecc_per_block = ECC_PER_BLOCK
        self.block_size = BLOCK_SIZE
        self.rs = RSCodec(self.ecc_per_block)

    def _try_decode_with_params(self, image, output_size=None, scale=1, 
                               sample_radius_factor=3, threshold_override=None):
        """
        使用指定参数尝试解码
        """
        from .sampler import sample_modules
        from .control_area import read_control_area
        from .masking import apply_mask
        from .data_codec import DataCodec
        
        matrix = sample_modules(image, self.module_size, output_size, scale, 
                               sample_radius_factor, threshold_override)
        
        control_bytes = read_control_area(matrix)
        
        if len(control_bytes) < 16:
            return None, 0
        
        # CRC校验
        crc = (control_bytes[14] << 8) | control_bytes[15]
        calculated_crc = 0xFFFF
        for byte in control_bytes[:14]:
            calculated_crc ^= byte
            for _ in range(8):
                if calculated_crc & 1:
                    calculated_crc = (calculated_crc >> 1) ^ 0xA001
                else:
                    calculated_crc >>= 1
        
        crc_ok = (crc == calculated_crc)
        calculated_data_len = (control_bytes[5] << 16) | (control_bytes[6] << 8) | control_bytes[7]
        data_len_ok = (0 < calculated_data_len <= self.data_per_frame)
        mask_pattern = control_bytes[8]
        
        # 应用掩码
        matrix = apply_mask(matrix, mask_pattern, self.matrix_size)
        
        # 读取数据
        data_bits = DataCodec.snake_read(matrix, self.matrix_size)
        data_bytes = []
        for i in range(0, len(data_bits), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                byte = 0
                for bit in byte_bits:
                    byte = (byte << 1) | bit
                data_bytes.append(byte)
        
        # 解码数据
        decoded_data, success_count = DataCodec.decode_blocks(data_bytes, self.rs)
        
        # 计算评分
        score = 0
        if crc_ok: score += 20
        if data_len_ok: score += 10
        score += success_count * 5  # RS解码成功块数
        
        if crc_ok and data_len_ok:
            return decoded_data[:calculated_data_len], score
        
        use_data_len = calculated_data_len if data_len_ok else self.data_per_frame
        if len(decoded_data) >= 100:
            return decoded_data[:use_data_len], score
        
        return None, score

    def _try_decode_multiple_params(self, image, output_size=None, scale=1):
        """
        使用多种参数组合尝试解码
        """
        import numpy as np
        # 导入PIL模块
        from PIL import Image
        
        # 参数组合：(采样半径因子, 阈值覆盖, 是否反转)
        params_list = [
            (3, None, False),  # 默认参数
            (3, None, True),   # 反转
            (2, None, False),  # 更小半径
            (4, None, False),  # 更大半径
            (3, 128, False),   # 固定阈值
            (3, 128, True),    # 固定阈值+反转
            (2, None, True),   # 小半径+反转
            (4, None, True),   # 大半径+反转
        ]
        
        best_result = None
        best_score = -1
        best_desc = None
        
        for sample_radius_factor, threshold_override, invert in params_list:
            working_img = image
            if invert:
                img_array = np.array(working_img)
                working_img = Image.fromarray(255 - img_array)
            
            result, score = self._try_decode_with_params(
                working_img, output_size, scale, 
                sample_radius_factor, threshold_override
            )
            
            if result is not None and score > best_score:
                desc = f"半径{sample_radius_factor}"
                if threshold_override:
                    desc += f" 阈值{threshold_override}"
                if invert:
                    desc += " 反转"
                
                best_score = score
                best_result = result
                best_desc = desc
        
        if best_result is not None and best_score >= 15:  # 最低接受分数
            return best_result, best_desc
        
        return None, "结果质量太低或全部失败"

    def decode_data(self, input_image):
        """
        优化的解码数据方法
        """
        # 导入PIL模块
        from PIL import Image
        from .transform import detect_and_warp
        
        image = Image.open(input_image)
        
        print("优先尝试直接采样解码（适合原始生成图）...")
        result, method = self._try_decode_multiple_params(image)
        if result is not None:
            print(f"直接采样{method}成功")
            return result
        
        # 尝试透视变换
        use_warp = True
        warp_results = None
        try:
            warp_results = detect_and_warp(image, self.module_size)
            if warp_results[0] is not None:
                print("透视变换成功")
        except Exception as e:
            print(f"透视变换检查失败: {e}")
            use_warp = False
        
        if use_warp and warp_results is not None:
            warp_normal, warp_flipped = warp_results
            
            # 尝试原始透视变换
            if warp_normal is not None:
                print("尝试原始透视变换解码...")
                processed_image, output_size, scale = warp_normal
                result, method = self._try_decode_multiple_params(processed_image, output_size, scale)
                if result is not None:
                    print(f"原始透视变换{method}解码成功")
                    return result
            
            # 尝试翻转透视变换
            if warp_flipped is not None:
                print("尝试翻转透视变换解码...")
                processed_image, output_size, scale = warp_flipped
                result, method = self._try_decode_multiple_params(processed_image, output_size, scale)
                if result is not None:
                    print(f"翻转透视变换{method}解码成功")
                    return result
        
        # 所有智能解码失败，使用基础解码
        print("所有智能解码失败，使用基础解码")
        return self._basic_decode(image)

    def _basic_decode(self, image):
        """
        基础解码方法
        """
        from .sampler import sample_modules
        from .control_area import read_control_area
        from .data_codec import DataCodec
        from .masking import apply_mask
        
        matrix = sample_modules(image, self.module_size)
        
        control_bytes = read_control_area(matrix)
        print(f"控制区字节数量: {len(control_bytes)}")
        if len(control_bytes) < 16:
            raise ValueError("控制区数据不足")
        
        version = control_bytes[0]
        frame_num = (control_bytes[1] << 8) | control_bytes[2]
        total_frames = (control_bytes[3] << 8) | control_bytes[4]
        calculated_data_len = (control_bytes[5] << 16) | (control_bytes[6] << 8) | control_bytes[7]
        mask_pattern = control_bytes[8]
        
        # CRC校验
        crc = (control_bytes[14] << 8) | control_bytes[15]
        calculated_crc = 0xFFFF
        for byte in control_bytes[:14]:
            calculated_crc ^= byte
            for _ in range(8):
                if calculated_crc & 1:
                    calculated_crc = (calculated_crc >> 1) ^ 0xA001
                else:
                    calculated_crc >>= 1
        
        use_calculated_data_len = False
        final_data_len = None
        if crc == calculated_crc and calculated_data_len > 0 and calculated_data_len <= self.data_per_frame:
            use_calculated_data_len = True
            final_data_len = calculated_data_len
            print(f"CRC校验成功，使用数据长度: {final_data_len}")
        else:
            print("警告：控制区校验失败或数据长度不合理")
            print("解码后将去除末尾空字节")
            mask_pattern = 0  # 使用默认掩码
        
        matrix = apply_mask(matrix, mask_pattern, self.matrix_size)
        
        data_bits = DataCodec.snake_read(matrix, self.matrix_size)
        data_bytes = []
        for i in range(0, len(data_bits), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                byte = 0
                for bit in byte_bits:
                    byte = (byte << 1) | bit
                data_bytes.append(byte)
        
        decoded_data, _ = DataCodec.decode_blocks(data_bytes, self.rs)
        
        if use_calculated_data_len and final_data_len is not None:
            return decoded_data[:final_data_len]
        else:
            # 去除末尾空字节
            last_non_zero = len(decoded_data)
            for i in range(len(decoded_data) - 1, -1, -1):
                if decoded_data[i] != 0:
                    last_non_zero = i + 1
                    break
            return decoded_data[:last_non_zero]

    def decode_file(self, input_image, output_file):
        """
        解码文件方法
        """
        data = self.decode_data(input_image)
        with open(output_file, 'wb') as f:
            f.write(data)
        return len(data)