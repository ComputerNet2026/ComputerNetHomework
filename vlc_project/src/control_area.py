from .config import MARGIN, FINDER_SIZE, MATRIX_SIZE

def is_control_module(i, j):
    CONTROL_START = MARGIN + FINDER_SIZE + 1
    BLOCK_SIZE = 32
    
    if i == 5 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
        return True
    if i == 6 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
        return True
    if j == 5 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
        return True
    if j == 6 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
        return True
    
    if i == 117 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
        return True
    if i == 118 and CONTROL_START <= j < CONTROL_START + BLOCK_SIZE:
        return True
    if j == 117 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
        return True
    if j == 118 and CONTROL_START <= i < CONTROL_START + BLOCK_SIZE:
        return True
    
    return False

def encode_control_area(version, data_len, mask_pattern=0, frame_num=0, total_frames=1):
    control_data = bytearray(16)
    control_data[0] = version & 0xFF
    control_data[1] = (frame_num >> 8) & 0xFF
    control_data[2] = frame_num & 0xFF
    control_data[3] = (total_frames >> 8) & 0xFF
    control_data[4] = total_frames & 0xFF
    control_data[5] = (data_len >> 16) & 0xFF
    control_data[6] = (data_len >> 8) & 0xFF
    control_data[7] = data_len & 0xFF
    control_data[8] = mask_pattern & 0xFF
    control_data[9:14] = b'\x00' * 5
    
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
    
    return control_data * 2

def write_control_area(matrix, control_bits):
    bit_idx = 0
    CONTROL_START = MARGIN + FINDER_SIZE + 1
    BLOCK_SIZE = 32
    
    for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[5][j] = control_bits[bit_idx]
            bit_idx += 1
    
    for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[6][j] = control_bits[bit_idx]
            bit_idx += 1
    
    for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[i][5] = control_bits[bit_idx]
            bit_idx += 1
    
    for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[i][6] = control_bits[bit_idx]
            bit_idx += 1
    
    bit_idx = 0
    
    for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[117][j] = control_bits[bit_idx]
            bit_idx += 1
    
    for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[118][j] = control_bits[bit_idx]
            bit_idx += 1
    
    for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[i][117] = control_bits[bit_idx]
            bit_idx += 1
    
    for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if bit_idx < len(control_bits):
            matrix[i][118] = control_bits[bit_idx]
            bit_idx += 1

def read_control_area(matrix):
    bits = []
    CONTROL_START = MARGIN + FINDER_SIZE + 1
    BLOCK_SIZE = 32
    
    for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if j < MATRIX_SIZE:
            bits.append(matrix[5][j])
    for j in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if j < MATRIX_SIZE:
            bits.append(matrix[6][j])
    for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if i < MATRIX_SIZE:
            bits.append(matrix[i][5])
    for i in range(CONTROL_START, CONTROL_START + BLOCK_SIZE):
        if i < MATRIX_SIZE:
            bits.append(matrix[i][6])
    
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) == 8:
            byte = 0
            for bit in byte_bits:
                byte = (byte << 1) | bit
            bytes_list.append(byte)
    
    return bytes_list
