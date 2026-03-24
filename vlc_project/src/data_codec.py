"""
OptTrans数据编解码模块
"""

class DataCodec:
    """
    统一的数据编解码类
    """
    
    @staticmethod
    def is_data_module(i, j, matrix_size):
        """
        判断是否为数据模块
        """
        from .control_area import is_control_module
        
        # 检查是否在定位点区域
        if (i < 11 and j < 11) or (i < 11 and j >= (matrix_size - 11)) or (i >= (matrix_size - 11) and j < 11):
            return False
        # 检查是否在小型定位点区域
        if i >= (matrix_size - 7) and j >= (matrix_size - 7):
            return False
        # 检查是否在控制区域
        if is_control_module(i, j):
            return False
        return True

    @staticmethod
    def get_data_positions(matrix_size):
        """
        预计算所有数据模块的位置，提升性能
        """
        positions = []
        for i in range(matrix_size):
            for j in range(matrix_size):
                if DataCodec.is_data_module(i, j, matrix_size):
                    positions.append((i, j))
        return positions

    @staticmethod
    def encode_data(data, rs):
        """
        数据编码 - 修复容量计算问题
        """
        from .config import BLOCKS, DATA_PER_BLOCK, DATA_PER_FRAME
        
        # 计算实际可用容量
        actual_capacity = BLOCKS * DATA_PER_BLOCK
        data_len = len(data)
        
        if data_len > DATA_PER_FRAME:
            raise ValueError(f"数据长度超过每帧最大容量 {DATA_PER_FRAME} 字节")
        
        if data_len > actual_capacity:
            raise ValueError(f"数据长度超过实际容量 {actual_capacity} 字节")
        
        # 填充数据到指定长度
        padded_data = data + b'\x00' * (actual_capacity - data_len)
        
        encoded_data = bytearray()
        for i in range(BLOCKS):
            start = i * DATA_PER_BLOCK
            end = start + DATA_PER_BLOCK
            block_data = padded_data[start:end]
            encoded_block = rs.encode(block_data)
            encoded_data.extend(encoded_block)
        
        return encoded_data

    @staticmethod
    def decode_blocks(data_bytes, rs):
        """
        数据解码 - 优化错误处理
        """
        from .config import BLOCKS, BLOCK_SIZE, DATA_PER_BLOCK
        
        decoded_data = bytearray()
        success_count = 0
        
        for i in range(BLOCKS):
            start = i * BLOCK_SIZE
            end = start + BLOCK_SIZE
            if end <= len(data_bytes):
                block_data = bytes(data_bytes[start:end])
                try:
                    decoded_block = rs.decode(block_data)[0]
                    decoded_data.extend(decoded_block)
                    success_count += 1
                except Exception:
                    # 解码失败时使用原始数据
                    decoded_data.extend(data_bytes[start:start+DATA_PER_BLOCK])
        
        return decoded_data, success_count

    @classmethod
    def snake_fill(cls, data_bits, matrix_size):
        """
        优化的蛇形路径填充
        """
        # 预计算数据位置
        data_positions = cls.get_data_positions(matrix_size)
        
        matrix = [[0]*matrix_size for _ in range(matrix_size)]
        bit_idx = 0
        
        for i, j in data_positions:
            if bit_idx < len(data_bits):
                matrix[i][j] = data_bits[bit_idx]
                bit_idx += 1
        
        return matrix

    @classmethod
    def snake_read(cls, matrix, matrix_size):
        """
        优化的蛇形路径读取
        """
        # 预计算数据位置
        data_positions = cls.get_data_positions(matrix_size)
        
        bits = []
        for i, j in data_positions:
            bits.append(matrix[i][j])
        
        return bits

# 原有的函数接口
def is_data_module(i, j, matrix_size):
    return DataCodec.is_data_module(i, j, matrix_size)

def encode_data(data, rs):
    return DataCodec.encode_data(data, rs)

def decode_blocks(data_bytes, rs):
    return DataCodec.decode_blocks(data_bytes, rs)

def snake_fill(data_bits, matrix_size):
    return DataCodec.snake_fill(data_bits, matrix_size)

def snake_read(matrix, matrix_size):
    return DataCodec.snake_read(matrix, matrix_size)