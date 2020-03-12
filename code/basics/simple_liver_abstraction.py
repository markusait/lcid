import cv2
import numpy as np
from matplotlib import pyplot as plt

img_liver_healthy_path = './images/liver_healthy.png'
img_liver_cirrhosis_path = './images/liver_cirrhosis.png'

# Read image as grayscale and convert to array.
# Set second parameter to 1 if rgb is required
img_liver_healthy = cv2.imread(img_liver_healthy_path, 0)
img_liver_cirrhosis = cv2.imread(img_liver_cirrhosis_path, 0)

# This value varies between png and jpg
WHITE_COLOR_VALUE = 255

# Direction values for chain-code from center
DIRECTIONS = [0,  1,  2,
              7,      3,
              6,  5,  4]
DIR_2_IDX = dict(zip(DIRECTIONS, range(len(DIRECTIONS))))

CHANGE_J = [-1,  0,  1,  # x or columns
            -1,      1,
            -1,  0,  1]

CHANGE_I = [-1, -1, -1,  # y or rows
             0,      0,
             1,  1,  1]

# Main driver function to calculate chain code diff
def main():
    liver_healthy_chain_code = get_chain_code_and_safe_image(img_liver_healthy, "./images/liver_healthy_chain_code.png")
    liver_cirrhosis_chain_code = get_chain_code_and_safe_image(img_liver_cirrhosis, "./images/liver_cirrhosis_chain_code.png")

    diff = get_difference_in_chain_codes(liver_healthy_chain_code, liver_cirrhosis_chain_code)
    print(f"Images are different on: {diff} occasions")


# Returns the first non-white starting point
def get_starting_point(img):
    for i, row in enumerate(img):
        for j, value in enumerate(row):
            if value != WHITE_COLOR_VALUE:
                start_point = (i, j)
                print("start_point", start_point, value)
                return start_point
        # If no value found yet we continue
        else:
            continue
        break
    raise Exception("No starting point found")


# Returns the chain code and saves it's visualization
def get_chain_code_and_safe_image(img, save_path):
    # Discover the first non-white point starting top, left
    start_point = get_starting_point(img)
    # the border, later drawn around the image
    border = []
    # saving chain_code as array
    chain = []
    # initialize iterator for edges
    curr_point = start_point
    for direction in DIRECTIONS:
        idx = DIR_2_IDX[direction]
        new_point = (start_point[0]+CHANGE_I[idx], start_point[1]+CHANGE_J[idx])
        if img[new_point] != WHITE_COLOR_VALUE:
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break

    # count = 0
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
            if img[new_point] != WHITE_COLOR_VALUE:
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        # potential stack overflow prevention
        # if count == 5000: break
        # count += 1

    # Reversing gray scale to get original Black and White colors
    plt.imshow(img, cmap='Greys_r')

    # Plotting the chain code
    plt.plot([i[1] for i in border], [i[0] for i in border])

    # Saving file to path
    plt.savefig(save_path)

    return chain

# Gets the difference between two codes and calculates how often it differs
def get_difference_in_chain_codes(code_1, code_2):
    min_len = min(len(code_1), len(code_2))
    diff_array = list(np.absolute(np.array(code_1[:min_len]) - np.array(code_2[:min_len])))
    return len(diff_array) - diff_array.count(0)



main()



