from reedsolo import RSCodec
from .config import (
    DATA_PER_FRAME, BLOCKS, DATA_PER_BLOCK, ECC_PER_BLOCK, BLOCK_SIZE,
    MATRIX_SIZE, FINDER_SIZE, SMALL_FINDER_SIZE
)
from .control_area import is_control_module

def is_data_module(i, j, matrix_size):
    if (i < 11 and j < 11) or (i < 11 and j >= (matrix_size - 11)) or (i >= (matrix_size - 11) and j < 11):
        return False
    if i >= (matrix_size - 7) and j >= (matrix_size - 7):
        return False
    if is_control_module(i, j):
        return False
    return True

def encode_data(data, rs):
    data_len = len(data)
    if data_len > DATA_PER_FRAME:
        raise ValueError(f"数据长度超过每帧最大容量 {DATA_PER_FRAME} 字节")
    
    padded_data = data + b'\x00' * (DATA_PER_FRAME - data_len)
    
    encoded_data = bytearray()
    for i in range(BLOCKS):
        start = i * DATA_PER_BLOCK
        end = start + DATA_PER_BLOCK
        block_data = padded_data[start:end]
        encoded_block = rs.encode(block_data)
        encoded_data.extend(encoded_block)
    
    return encoded_data

def decode_blocks(data_bytes, rs):
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
                decoded_data.extend(data_bytes[start:start+DATA_PER_BLOCK])
    return decoded_data, success_count

def snake_fill(data_bits, matrix_size):
    matrix = [[0]*matrix_size for _ in range(matrix_size)]
    bit_idx = 0
    for row in range(matrix_size - 1, -1, -1):
        if row % 2 == 0:
            cols = range(matrix_size - 1, -1, -1)
        else:
            cols = range(matrix_size)
        for col in cols:
            if is_data_module(row, col, matrix_size) and bit_idx < len(data_bits):
                matrix[row][col] = data_bits[bit_idx]
                bit_idx += 1
    return matrix

def snake_read(matrix, matrix_size):
    bits = []
    for row in range(matrix_size - 1, -1, -1):
        if row % 2 == 0:
            cols = range(matrix_size - 1, -1, -1)
        else:
            cols = range(matrix_size)
        for col in cols:
            if is_data_module(row, col, matrix_size):
                bits.append(matrix[row][col])
    return bits
