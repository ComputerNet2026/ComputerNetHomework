from PIL import Image
from reedsolo import RSCodec
import numpy as np
from .config import (
    MODULE_SIZE, MATRIX_SIZE, MARGIN, TOTAL_SIZE,
    DATA_PER_FRAME, BLOCKS, DATA_PER_BLOCK, ECC_PER_BLOCK, BLOCK_SIZE
)
from .sampler import sample_modules
from .transform import detect_and_warp
from .control_area import read_control_area
from .data_codec import snake_read, decode_blocks, is_data_module
from .masking import apply_mask

class OptTransDecoder:
    def __init__(self):
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
    
    def _try_decode_from_image(self, image, output_size=None, scale=1, sample_radius_factor=3, threshold_override=None):
        matrix = sample_modules(image, self.module_size, output_size, scale, sample_radius_factor, threshold_override)
        
        control_bytes = read_control_area(matrix)
        
        if len(control_bytes) < 16:
            return None
        
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
        mask_pattern = control_bytes[8]
        
        data_len_ok = (0 < calculated_data_len <= self.data_per_frame)
        
        matrix = apply_mask(matrix, mask_pattern, self.matrix_size)
        
        data_bits = snake_read(matrix, self.matrix_size)
        data_bytes = []
        for i in range(0, len(data_bits), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                byte = 0
                for bit in byte_bits:
                    byte = (byte << 1) | bit
                data_bytes.append(byte)
        
        decoded_data, _ = decode_blocks(data_bytes, self.rs)
        
        if crc_ok and data_len_ok:
            return decoded_data[:calculated_data_len]
        
        use_data_len = calculated_data_len if data_len_ok else self.data_per_frame
        if len(decoded_data) >= 100:
            return decoded_data[:use_data_len]
        
        return None
    
    def _try_decode_with_thresholds(self, image, output_size=None, scale=1):
        params_list = [
            (3, None, False),
            (3, None, True),
            (2, None, False),
            (4, None, False),
            (3, 128, False),
            (3, 128, True),
            (2, None, True),
            (4, None, True),
        ]
        
        best_result = None
        best_score = -1
        best_desc = None
        
        for sample_radius_factor, threshold_override, invert in params_list:
            working_img = image
            if invert:
                img_array = np.array(working_img)
                working_img = Image.fromarray(255 - img_array)
            
            matrix = sample_modules(
                working_img, self.module_size, output_size, scale, 
                sample_radius_factor=sample_radius_factor, 
                threshold_override=threshold_override
            )
            
            control_bytes = read_control_area(matrix)
            
            score = 0
            if len(control_bytes) >= 16 and control_bytes[0] == 1:
                score += 10
                
                crc = (control_bytes[14] << 8) | control_bytes[15]
                calculated_crc = 0xFFFF
                for byte in control_bytes[:14]:
                    calculated_crc ^= byte
                    for _ in range(8):
                        if calculated_crc & 1:
                            calculated_crc = (calculated_crc >> 1) ^ 0xA001
                        else:
                            calculated_crc >>= 1
                
                if crc == calculated_crc:
                    score += 20
                
                calculated_data_len = (control_bytes[5] << 16) | (control_bytes[6] << 8) | control_bytes[7]
                if 900 <= calculated_data_len <= 960:
                    score += 15
            
            result = self._try_decode_from_image(
                working_img, output_size, scale, 
                sample_radius_factor=sample_radius_factor, 
                threshold_override=threshold_override
            )
            
            if result is not None:
                desc = f"半径{sample_radius_factor}"
                if threshold_override:
                    desc += f" 阈值{threshold_override}"
                if invert:
                    desc += " 翻转"
                
                if 900 <= len(result) <= 960:
                    score += 10
                
                if len(control_bytes) >= 16 and control_bytes[0] == 1:
                    mask_pattern = control_bytes[8]
                    matrix_masked = apply_mask(matrix, mask_pattern, self.matrix_size)
                    data_bits = snake_read(matrix_masked, self.matrix_size)
                    data_bytes = []
                    for i in range(0, len(data_bits), 8):
                        byte_bits = data_bits[i:i+8]
                        if len(byte_bits) == 8:
                            byte = 0
                            for bit in byte_bits:
                                byte = (byte << 1) | bit
                            data_bytes.append(byte)
                    
                    success_count = 0
                    for i in range(self.blocks):
                        start = i * self.block_size
                        end = start + self.block_size
                        if end <= len(data_bytes):
                            block_data = bytes(data_bytes[start:end])
                            try:
                                self.rs.decode(block_data)[0]
                                success_count += 1
                            except:
                                pass
                    
                    score += success_count * 5
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_desc = desc
        
        MIN_ACCEPTABLE_SCORE = 20
        if best_result is not None and best_score >= MIN_ACCEPTABLE_SCORE:
            return best_result, best_desc
        
        return None, "结果质量太低或全部失败"
    
    def decode_data(self, input_image):
        image = Image.open(input_image)
        
        print("优先尝试直接采样解码（适合原始生成图）...")
        result, method = self._try_decode_with_thresholds(image)
        if result is not None:
            print(f"直接采样{method}成功")
            return result
        
        use_warp = True
        warp_normal = None
        warp_flipped = None
        try:
            warp_normal, warp_flipped = detect_and_warp(image, self.module_size)
            if warp_normal is not None:
                print("透视变换成功")
        except Exception as e:
            print(f"透视变换检查失败: {e}")
            use_warp = False
        
        if use_warp and warp_normal is not None:
            print("尝试原始透视变换解码...")
            processed_image, output_size, scale = warp_normal
            result, method = self._try_decode_with_thresholds(processed_image, output_size, scale)
            if result is not None:
                print(f"原始透视变换{method}解码成功")
                return result
        
        if use_warp and warp_flipped is not None:
            print("尝试翻转透视变换解码...")
            processed_image, output_size, scale = warp_flipped
            result, method = self._try_decode_with_thresholds(processed_image, output_size, scale)
            if result is not None:
                print(f"翻转透视变换{method}解码成功")
                return result
        
        print("所有智能解码失败，使用基础解码")
        processed_image = image
        output_size = None
        scale = 1
        
        matrix = sample_modules(processed_image, self.module_size, output_size, scale)
        
        control_bytes = read_control_area(matrix)
        print(f"控制区字节数量: {len(control_bytes)}")
        if len(control_bytes) < 16:
            raise ValueError("控制区数据不足")
        
        version = control_bytes[0]
        frame_num = (control_bytes[1] << 8) | control_bytes[2]
        total_frames = (control_bytes[3] << 8) | control_bytes[4]
        calculated_data_len = (control_bytes[5] << 16) | (control_bytes[6] << 8) | control_bytes[7]
        mask_pattern = control_bytes[8]
        
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
            mask_pattern = 0
        
        matrix = apply_mask(matrix, mask_pattern, self.matrix_size)
        
        data_bits = snake_read(matrix, self.matrix_size)
        data_bytes = []
        for i in range(0, len(data_bits), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                byte = 0
                for bit in byte_bits:
                    byte = (byte << 1) | bit
                data_bytes.append(byte)
        
        decoded_data, _ = decode_blocks(data_bytes, self.rs)
        
        if use_calculated_data_len and final_data_len is not None:
            return decoded_data[:final_data_len]
        else:
            last_non_zero = len(decoded_data)
            for i in range(len(decoded_data) - 1, -1, -1):
                if decoded_data[i] != 0:
                    last_non_zero = i + 1
                    break
            return decoded_data[:last_non_zero]
    
    def decode_file(self, input_image, output_file):
        data = self.decode_data(input_image)
        with open(output_file, 'wb') as f:
            f.write(data)
        return len(data)
