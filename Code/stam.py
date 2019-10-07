import numpy as np
import cv2
from my_im_utils import display_image, read_image_cv2,save_image
filename = "images/freedom.png"
im = read_image_cv2(filename=filename, mode=cv2.IMREAD_UNCHANGED)
im2 = im[465:510, 170:320]
np.save("im2.npy", im2)
# display_image(im2)

# save_image(im=im2, new_file_name="images/reciept2.png")
