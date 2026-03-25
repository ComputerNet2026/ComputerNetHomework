import numpy as np

FINDER_SIZE = 7
MARGIN = 4


def put_finder_patterns(frame):
    finder = _generate_finder()
    size = frame.shape[0]

    _place(frame, finder, MARGIN, MARGIN)
    _place(frame, finder, size - MARGIN - FINDER_SIZE, MARGIN)
    _place(frame, finder, MARGIN, size - MARGIN - FINDER_SIZE)


def _generate_finder():
    pattern = np.zeros((7, 7), dtype=np.uint8)

    for y in range(7):
        for x in range(7):
            if x == 0 or x == 6 or y == 0 or y == 6:
                pattern[y][x] = 1
            elif 2 <= x <= 4 and 2 <= y <= 4:
                pattern[y][x] = 1
            else:
                pattern[y][x] = 0

    return pattern


def _place(frame, pattern, start_x, start_y):
    for y in range(FINDER_SIZE):
        for x in range(FINDER_SIZE):
            frame[start_y + y][start_x + x] = pattern[y][x]