from PIL import Image
from reedsolo import RSCodec
from .config import (
    MODULE_SIZE, MATRIX_SIZE, MARGIN, TOTAL_SIZE,
    DATA_PER_FRAME, BLOCKS, DATA_PER_BLOCK, ECC_PER_BLOCK,
    FINDER_SIZE, SMALL_FINDER_SIZE
)
from .patterns import generate_finder_pattern, generate_small_finder_pattern
from .control_area import encode_control_area, write_control_area
from .data_codec import encode_data, snake_fill, is_data_module
from .masking import apply_mask, calculate_mask_penalty

class OptTransEncoder:
    def __init__(self, version=1):
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
        
        self.rs = RSCodec(self.ecc_per_block)
    
    def encode_data(self, data, output_image, frame_num=0, total_frames=1):
        matrix = [[0]*self.matrix_size for _ in range(self.matrix_size)]
        
        finder = generate_finder_pattern(size=FINDER_SIZE)
        small_finder = generate_small_finder_pattern()
        
        for i in range(11):
            for j in range(11):
                matrix[i][j] = finder[i][j]
        for i in range(11):
            for j in range(11):
                matrix[i][self.matrix_size-11+j] = finder[i][j]
        for i in range(11):
            for j in range(11):
                matrix[self.matrix_size-11+i][j] = finder[i][j]
        for i in range(7):
            for j in range(7):
                matrix[self.matrix_size-7+i][self.matrix_size-7+j] = small_finder[i][j]
        
        best_mask = 0
        min_penalty = float('inf')
        
        data_bytes = encode_data(data, self.rs)
        
        data_bits = []
        for byte in data_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            data_bits.extend(bits)
        
        data_matrix = snake_fill(data_bits, self.matrix_size)
        
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if is_data_module(i, j, self.matrix_size):
                    matrix[i][j] = data_matrix[i][j]
        
        for mask_pattern in range(8):
            test_matrix = [row[:] for row in matrix]
            
            control_bytes_test = encode_control_area(self.version, len(data), mask_pattern, frame_num, total_frames)
            control_bits_test = []
            for byte in control_bytes_test:
                bits = [(byte >> (7 - i)) & 1 for i in range(8)]
                control_bits_test.extend(bits)
            
            test_matrix_control = [row[:] for row in test_matrix]
            write_control_area(test_matrix_control, control_bits_test)
            
            masked_test = apply_mask(test_matrix_control, mask_pattern, self.matrix_size)
            penalty = calculate_mask_penalty(masked_test, self.matrix_size)
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_mask = mask_pattern
        
        control_bytes = encode_control_area(self.version, len(data), mask_pattern=best_mask, frame_num=frame_num, total_frames=total_frames)
        control_bits = []
        for byte in control_bytes:
            bits = [(byte >> (7 - i)) & 1 for i in range(8)]
            control_bits.extend(bits)
        write_control_area(matrix, control_bits)
        
        matrix = apply_mask(matrix, best_mask, self.matrix_size)
        
        padded = [[0]*self.total_size for _ in range(self.total_size)]
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                padded[self.margin+i][self.margin+j] = matrix[i][j]
        
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        pixels = img.load()
        
        for i in range(self.total_size):
            for j in range(self.total_size):
                color = (0, 0, 0) if padded[i][j] == 1 else (255, 255, 255)
                for y in range(i * self.module_size, (i + 1) * self.module_size):
                    for x in range(j * self.module_size, (j + 1) * self.module_size):
                        pixels[x, y] = color
        
        img.save(output_image)
        return img
    
    def encode_file(self, input_file, output_image):
        with open(input_file, 'rb') as f:
            data = f.read()
        
        total_frames = (len(data) + self.data_per_frame - 1) // self.data_per_frame
        
        if total_frames == 1:
            return self.encode_data(data, output_image, frame_num=0, total_frames=1)
        else:
            for i in range(total_frames):
                start = i * self.data_per_frame
                end = min((i + 1) * self.data_per_frame, len(data))
                frame_data = data[start:end]
                frame_output = f"{output_image.rsplit('.', 1)[0]}_frame{i}.{output_image.rsplit('.', 1)[1]}"
                self.encode_data(frame_data, frame_output, frame_num=i, total_frames=total_frames)
            return total_frames
