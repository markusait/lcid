# This code is based on https://www.kaggle.com/mburger/freeman-chain-code-second-attempt/data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from math import sqrt
from matplotlib import pyplot as plt
from itertools import chain

from subprocess import check_output
input_dir = "./input"
print(check_output(["ls", input_dir]).decode("utf8"))
train = pd.read_csv(input_dir + "/train.csv", header=0)

# Meaning ???
train[:3]

# Any results you write to the current directory are saved as output.
labels = train['label']
train_images = train.drop('label', axis=1)
train_images.head()

# data_frame = pd.DataFrame(train_images[9:10])
data_frame = train_images[9:10]
target_values = data_frame.values

RESHAPE_ROWS = -1
RESHAPE_COLUMNS = 28
reshaped_array = np.reshape(target_values, (RESHAPE_ROWS, RESHAPE_COLUMNS))
image = reshaped_array.astype(np.uint8)
plt.imshow(image, cmap='Greys')

plt.savefig("./images/1blury.png")

# Getting a sharper image
ret,img = cv2.threshold(image,70,255,0)
plt.imshow(img, cmap='Greys')
plt.savefig("./images/2sharp.png")

# Discover the first (black) point starting top, left
BLACK_COLOR_VALUE = 255
for i, row in enumerate(img):
    print("row", row)
    for j, value in enumerate(row):
        if value == BLACK_COLOR_VALUE:
            start_point = (i, j)
            print("start_point", start_point, value)
            break
    # If no value found yet we continue
    else:
        continue
    break

# Meaning ???
img[3:6, 19:22]

# Direction values for chain-code from center
DIRECTIONS = [ 0,  1,  2,
               7,      3,
               6,  5,  4]
DIR_2_IDX = dict(zip(DIRECTIONS, range(len(DIRECTIONS))))

CHANGE_J =   [-1,  0,  1, # x or columns
              -1,      1,
              -1,  0,  1]

CHANGE_I =   [-1, -1, -1, # y or rows
               0,      0,
               1,  1,  1]

# the border, later drawn around the image
border = []
# saving chain_code as array
chain = []
# iterator for edges
curr_point = start_point
for direction in DIRECTIONS:
    idx = DIR_2_IDX[direction]
    new_point = (start_point[0]+CHANGE_I[idx], start_point[1]+CHANGE_J[idx])
    if img[new_point] != 0: # if is ROI
        border.append(new_point)
        chain.append(direction)
        curr_point = new_point
        break

count = 0
while curr_point != start_point:
    #figure direction to start search
    b_direction = (direction + 5) % 8
    dirs_1 = range(b_direction, 8)
    dirs_2 = range(0, b_direction)
    dirs = []
    dirs.extend(dirs_1)
    dirs.extend(dirs_2)
    for direction in dirs:
        idx = DIR_2_IDX[direction]
        new_point = (curr_point[0]+CHANGE_I[idx], curr_point[1]+CHANGE_J[idx])
        if image[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break
    if count == 1000: break
    count += 1



print("Outer edges:", count)
print("Chain Code:", chain)

plt.imshow(img, cmap='Greys')
plt.plot([i[1] for i in border], [i[0] for i in border])
plt.savefig("./images/3chain_code.png")


