# import the necessary packages
import numpy as np
import cv2
import pdb

def original_lbp(image):
    """origianl local binary pattern"""
    rows = image.shape[0]
    cols = image.shape[1]
    lbp_image = np.zeros((rows - 2, cols - 2), np.uint8)
    pattern = np.array([
        [128, 64, 32],
        [1,   0,  16],
        [2,   4,  8]
    ])

    # 计算每个像素点的lbp值，具体范围如上lbp_image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            wind = image[i - 1: i + 2, j - 1: j + 2]
            # print("wind: \n", wind)
            wind = wind.astype(np.float64) - wind[1, 1]
            # print("wind: \n", wind)
            clip = np.clip(wind, 0, 1)
            # print("clip: \n", clip)
            lbp = clip * pattern
            lbp_image[i - 1, j - 1] = lbp.sum()
            # print(lbp_image[i - 1, j - 1])
            # pdb.set_trace()

    return lbp_image

if __name__ == '__main__':
    image = cv2.imread("./lms.jpg", 0)
    cv2.imshow("image", image)
    org_lbp_image = original_lbp(image)
    cv2.imshow("org_lbp_image", org_lbp_image)
    cv2.waitKey(0)