import numpy as np


sobel_x = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

sobel_y = sobel_x.T

def box(size=3):
    return np.ones((size, size)) / (size**2)

def gaussian(size=3):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    sigma = size / 6
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def laplacian(type_=1, add_to_image=False):
    if type_ == 1:
        filter_ = np.array([
            [0,  -1,  0],
            [-1,  4, -1],
            [0,  -1,  0]
        ])
    else:
        filter_ = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
        
    if add_to_image:
        filter_ += np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        
    return filter_