from .config import FINDER_SIZE, SMALL_FINDER_SIZE, MATRIX_SIZE

def generate_finder_pattern(size=FINDER_SIZE):
    pattern = [[0]*size for _ in range(size)]
    border_width = 2
    center_size = 5
    center_start = (size - center_size) // 2
    center_end = center_start + center_size
    
    for i in range(size):
        for j in range(size):
            if i < border_width or i >= size - border_width or j < border_width or j >= size - border_width:
                pattern[i][j] = 1
            elif center_start <= i < center_end and center_start <= j < center_end:
                pattern[i][j] = 1
            else:
                pattern[i][j] = 0
    return pattern

def generate_small_finder_pattern():
    pattern = [[0]*SMALL_FINDER_SIZE for _ in range(SMALL_FINDER_SIZE)]
    border_width = 1
    center_size = 3
    center_start = (SMALL_FINDER_SIZE - center_size) // 2
    center_end = center_start + center_size
    
    for i in range(SMALL_FINDER_SIZE):
        for j in range(SMALL_FINDER_SIZE):
            if i < border_width or i >= SMALL_FINDER_SIZE - border_width or j < border_width or j >= SMALL_FINDER_SIZE - border_width:
                pattern[i][j] = 1
            elif center_start <= i < center_end and center_start <= j < center_end:
                pattern[i][j] = 1
            else:
                pattern[i][j] = 0
    return pattern

def generate_timing_pattern():
    length = MATRIX_SIZE - 22
    timing = [i % 2 for i in range(length)]
    return timing
