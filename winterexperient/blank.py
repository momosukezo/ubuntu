import cv2
import numpy as np

height = 600
width = 1060
blank = np.zeros((height, width, 3))
blank += 100

cv2.imwrite('blank.png',blank)
