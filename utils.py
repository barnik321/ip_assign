import numpy as np
from scipy import signal


def histogram(image):
    return np.unique(image, return_counts=True)


def scale(image):#, vmin=0, vmax=255):
    return (image - image.min()) / (image.max() - image.min()) * 255  # scale b/w 0-255


def quantize(image):
    return np.round(image).astype('uint8')  # quantize

def truncate(image):
    image = image.copy()
    image[image < 0] = 0
    image[image > 255] = 255
    
    return image

def binary_threshold(image, th=128):
    image = image.copy()
    image[image < th] = 0
    image[image >= th] = 1
    
    return image


def _im2col(image, kernel):
    # each window of image as columns
    kernel_shape = kernel.shape[0]
    rows = []

    for row in range(image.shape[0] - (kernel.shape[0]-1)):
        for col in range(image.shape[1] - (kernel.shape[0]-1)):
            window = image[row: row + kernel_shape, col: col + kernel_shape]
            rows.append(list(
                window.flatten()
            ))
            
    return np.array(rows).T


def convolve(image, kernel):
    # uncomment line below to have a faster implementation but similar output
    return signal.correlate2d(image, kernel, mode='same', boundary='symm')

    # generates image of same size
    # kernel must be of odd size
    # https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf
    # padding
    n_pad = (kernel.shape[0]-1)//2
    image_temp = np.pad(image, (n_pad,), 'reflect')
    
    return (
        kernel.flatten() @ _im2col(image_temp, kernel)  # convolution happens here as a flattened array
    ).reshape(image.shape)