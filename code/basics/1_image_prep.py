import cv2
import numpy as np

# For relative imports see https://stackoverflow.com/questions/36476659/how-to-add-a-relative-path-in-python-to-find-image-and-other-file-with-a-short-p
img_liver_path = './images/liver/BZ-1.png'
save_path = './images/liver/liver_cirrhosis_1_edges_2.png'

# Read image as grayscale and convert to array.
# Set second parameter to 1 if rgb is required
# Make sure to convert to .png, as .bmp cannot be read
# online conversion tool: https://convertio.co/de/bmp-png/
img = cv2.imread(img_liver_path)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert gray scale
gray_img = 255 - gray_img

# Now applying Morphological closing to fill the holes in the image
kernel = np.ones((19, 19), np.uint8)
closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)

# Perform the canny edge detector to detect image edges
edges = cv2.Canny(closing, 30, 45)

cv2.imwrite(save_path, edges)