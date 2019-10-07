import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import cv2
import sys

# ========================================= CONSTANTS ============================================= #
BINARY_MODE = "1"
GRAYSCALE_MODE = "L"
RGB_MODE = "RGB"
RANGE_0_1 = "RANGE_0_1 "
RANGE_0_255 = "RANGE_0_255"
DEFAULT_IMAGE_FILENAME = "images/ltrs_a.png"
CANNY_LOW_THRESHOLD = 175
CANNY_HIGH_THRESHOLD = 320

# FOR FREEDOM
# CANNY_LOW_THRESHOLD = 100
# CANNY_HIGH_THRESHOLD = 250



def rgb2gray(rgb_image):
    if rgb_image.ndim == 3:
        r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
        grayscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grayscale_image
    else:
        raise ValueError("rgb_image doesn't have 3 color channels!")

def read_image_PIL(file_name, mode=None, to_0_1_range=False):
    im = PIL.Image.open(file_name).convert(mode)
    im = np.array(im)
    if mode != BINARY_MODE and to_0_1_range:
        im = im.astype(dtype=np.float64)
        np.divide(im, 255, out=im)
    return im

def read_image_cv2(filename, mode, to_0_1_range=False):
    im = cv2.imread(filename, mode) # numpy array, dtype = uint8
    if im is None:
        print("could not open {}".format(filename))
        sys.exit()
    if to_0_1_range:
        change_range(im=im, new_min=0, new_max=1, to_int=False)
    return im

def change_range(im, new_min, new_max, to_int=False):
    current_min, current_max = im.min(), im.max()
    new_range = new_max - new_min
    old_range = current_max - current_min
    new_im = (im.astype(np.float64) - current_min) * (new_range / old_range) + new_min
    if to_int:
        np.around(new_im, decimals=0, out=new_im)
        return new_im.astype(np.uint8)
    else:
        return new_im

def angle_between(v1, v2):
    """ returns angle (in radians) between two n-dimensional vectors"""
    v1 = np.array(v1, copy=False, subok=True)
    v2 = np.array(v2, copy=False, subok=True)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.dot(unit_v1, unit_v2))
    return angle

def convert_between_BGR_and_RGB(im):
    if im.ndim == 3:
        return im[:,:,::-1]
    return im

def show_image_cv2(im, title="image"):
    cv2.imshow(title, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_image(im, title=""):
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(im, new_file_name):
    """ takes np array , if needed converts to [0-255], and saves it"""
    if im.dtype==bool:
        im = im.astype(np.uint8)
    # if needed, convert to [0-255]
    if im.dtype != np.uint8 or im.max() < 255 or im.min() > 0:
        im  = change_range(im, new_min=0, new_max=255, to_int=True)
    pil_image = PIL.Image.fromarray(im)
    pil_image.save(new_file_name)
