import numpy as np
import cv2
from my_im_utils import display_image, read_image_cv2, save_image

filename = "images/freedom.png"
im = read_image_cv2(filename=filename, mode=cv2.IMREAD_GRAYSCALE)
START_ROW = 460
START_COL = 170
small_im_before = im[START_ROW:515, START_COL:320]
display_image(small_im_before, title="before")
small_im_after = im[START_ROW:515, START_COL:320]
im_after = im.copy()
# E
small_im_after[15:27, 49] = small_im_after[15:27, 48]
small_im_after[27:46, 48] = small_im_after[27:46, 49]

im_after[START_ROW+15:START_ROW + 27, START_COL+49] = im_after[START_ROW + 15:START_ROW + 27, START_COL + 48]
im_after[START_ROW + 27:START_ROW + 46, START_COL+48] = im_after[START_ROW + 27:START_ROW + 46, START_COL + 49]


display_image(small_im_before, title="small_im_before")
display_image(small_im_after, title="small_im_after")
display_image(im, title="im")
display_image(im_after, title="im_after")


print()