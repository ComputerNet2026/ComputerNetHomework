from .config import MATRIX_SIZE, FINDER_SIZE, SMALL_FINDER_SIZE
from .control_area import is_control_module
from .data_codec import is_data_module

def mask_func_0(i, j): return (i + j) % 2 == 0
def mask_func_1(i, j): return i % 2 == 0
def mask_func_2(i, j): return j % 3 == 0
def mask_func_3(i, j): return (i + j) % 3 == 0
def mask_func_4(i, j): return (i // 2 + j // 3) % 2 == 0
def mask_func_5(i, j): return (((i * j) % 2) + ((i * j) % 3)) == 0
def mask_func_6(i, j): return (((i * j) % 2) + ((i * j) % 3)) % 2 == 0
def mask_func_7(i, j): return (((i + j) % 2) + ((i * j) % 3)) % 2 == 0

MASK_FUNCS = [
    mask_func_0, mask_func_1, mask_func_2, mask_func_3,
    mask_func_4, mask_func_5, mask_func_6, mask_func_7
]

def apply_mask(matrix, mask_pattern, matrix_size):
    masked = [row[:] for row in matrix]
    mask_func = MASK_FUNCS[mask_pattern % 8]
    
    for i in range(len(masked)):
        for j in range(len(masked[0])):
            if (i < 11 and j < 11) or (i < 11 and j >= (matrix_size - 11)) or (i >= (matrix_size - 11) and j < 11):
                continue
            if i >= (matrix_size - 7) and j >= (matrix_size - 7):
                continue
            if is_data_module(i, j, matrix_size):
                if mask_func(i, j):
                    masked[i][j] = 1 - masked[i][j]
    return masked

def calculate_mask_penalty(matrix, matrix_size):
    penalty = 0
    
    for i in range(matrix_size):
        run_length = 1
        for j in range(1, matrix_size):
            if matrix[i][j] == matrix[i][j-1]:
                run_length += 1
            else:
                if run_length >= 5:
                    penalty += run_length - 2
                run_length = 1
        if run_length >= 5:
            penalty += run_length - 2
    
    for j in range(matrix_size):
        run_length = 1
        for i in range(1, matrix_size):
            if matrix[i][j] == matrix[i-1][j]:
                run_length += 1
            else:
                if run_length >= 5:
                    penalty += run_length - 2
                run_length = 1
        if run_length >= 5:
            penalty += run_length - 2
    
    for i in range(matrix_size - 1):
        for j in range(matrix_size - 1):
            if (matrix[i][j] == matrix[i][j+1] == 
                matrix[i+1][j] == matrix[i+1][j+1]):
                penalty += 3
    
    dark_modules = 0
    total_modules = 0
    for i in range(matrix_size):
        for j in range(matrix_size):
            if is_data_module(i, j, matrix_size) or is_control_module(i, j):
                total_modules += 1
                if matrix[i][j] == 1:
                    dark_modules += 1
    
    if total_modules > 0:
        ratio = dark_modules / total_modules
        deviation = abs(ratio - 0.5)
        penalty += int(deviation * 100)
    
    return penalty
