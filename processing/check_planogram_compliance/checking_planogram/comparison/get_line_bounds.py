import numpy as np

# Крайние точки линии полки по X (левый и правый край), даже если она изогнута:
# [(100,50), (300,80), (520,60)] -> (100, 520)
# Возвращает (line_x_min, line_x_max).
def get_line_bounds(line):
    line_np = np.array(line)
    return np.min(line_np[:, 0]), np.max(line_np[:, 0])
