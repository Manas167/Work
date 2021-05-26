def loop_over(x):
    import time
    import numpy as np
    for i in range(100):
        mat = np.random.normal(size=(10,10)) + np.eye(10)
        mat = np.linalg.inv(mat)
    m = x*x*x
    return m

def merge_names(a, b):
    return '{} & {}'.format(a, b)

print("what")