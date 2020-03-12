# Kernel Convolution
# Is done by using a n x n Matrix and applying it to an image
# 1. For every pixel we put the Matrix in the center and multiply
#    its values and them sum it up and normalize by total value
#    thereby it is a weighted average, e.g if all values are one it is a mean blur
# 2. Lastly we overwrite that pixel to a different image to remain persistant
import math

image = [
    [17,14,13,20,78],
    [21,64,62,12,98],
    [42,54,61,22,10],
    [56,74,19,50,8],
    [13,91,79,31,23],
]
kernel = [
    [1,1,1],
    [1,1,1],
    [1,1,1],
]

def get_mean(image,kernel, x, y, image_width, image_height):
    adjusted_values = []
    # First we get the kernel_height x kernel_width array from image
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    height_start = y - math.floor(kernel_height / 2)
    width_start = x - math.floor(kernel_width / 2)

    for k_y_iter in range(kernel_height):
        image_y_pos = height_start + k_y_iter
        if image_y_pos < 0 or image_y_pos >= image_height:
            continue
        for k_x_iter in range(kernel_width):
            image_x_pos = width_start + k_x_iter
            if image_x_pos < 0 or image_x_pos >= image_width:
                continue
            pixel_value = image[image_y_pos][image_x_pos]
            kernel_value = kernel[k_y_iter][k_x_iter]
            adjusted_values.append(pixel_value * kernel_value)

    return math.floor(sum(adjusted_values) / len(adjusted_values))



def gaussian_blur(image, kernel):
    height = len(image)
    width = len(image[0])
    new_image = [[0]*width for i in range(height)]
    #we start at the top
    for x in range(height):
        for y in range(width):
            new_image[x][y] = get_mean(image, kernel,x, y, width, height)
            # print("mean", get_mean(image, kernel,x, y, width, height))
            # return
    return new_image



print("res", gaussian_blur(image, kernel))


