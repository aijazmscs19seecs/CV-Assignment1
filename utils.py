import cv2
import random
import numpy as np

def rgbExclusion(image, channel):
    assert image.shape[2] == 3, "Image doesn't contain 3 channels."
    img = image.copy()
    if channel == "r":
        img[:,:,2] = 0
    elif channel == "g":
        img[:,:,1] = 0
    elif channel == "b":
        img[:,:,0] = 0
    else:
        return 0

    return img

def convolution(image, kernel):

    # Flip the kernel for convolution
    kernel = np.flipud(np.fliplr(kernel))

    x_img, y_img = image.shape[0], image.shape[1]
    x_kernel, y_kernel = kernel.shape[0], kernel.shape[1]

    # convolution output
    output = np.zeros_like(image)
    padding = (y_kernel - 1)//2
    
    # Add zero padding to the input image
    padded_image = np.zeros((image.shape[0] + 2*padding, image.shape[1] + 2*padding))
    padded_image[padding:-padding, padding:-padding] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = (kernel * padded_image[y: y+3, x: x+3]).sum()
                
    return output

def salt_and_pepper(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gaussian_noise(image, mean=0, var=10, sigma=50):
    gaussian = np.random.normal(mean, sigma, (image.shape[0],image.shape[1])) 
    return image+gaussian