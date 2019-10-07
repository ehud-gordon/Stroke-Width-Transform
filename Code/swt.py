import math
import os
import sys
import time
import numpy as np
from my_im_utils import (read_image_cv2, display_image, show_image_cv2, change_range, save_image, angle_between)
import argparse
import cv2
import swt_cc
from my_im_utils import CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
import scipy.spatial
import pickle
import itertools

def get_edges(im):
    """ Computes Canny edges, derivative in x (horizontal) and y (vertical) direction.

    Uses the Sobel operator to compute derivative, and uses the paper's recommended threshold values
    for the Canny's hysteresis.
    :param im: NxD numpy array, dtype=uint8,  of grayscale image, with range of [0,255]
    :return sobel_x, sobel_y, canny - all NxD numpy arrays, dtype=float64
    """
    canny = cv2.Canny(image=im, threshold1=CANNY_LOW_THRESHOLD, threshold2=CANNY_HIGH_THRESHOLD)
    blurred_image = cv2.GaussianBlur(src=im, ksize=(3,3), sigmaX=0)
    sobel_x = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
    sobel_y = cv2.Sobel(src=blurred_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=-1)
    sobel_x = cv2.GaussianBlur(src=sobel_x, ksize=(5, 5), sigmaX=0)
    sobel_y = cv2.GaussianBlur(src=sobel_y, ksize=(5, 5), sigmaX=0)
    # normalize gradients by their magnitude
    magnitude = np.hypot(sobel_x, sobel_y)
    sobel_x /= magnitude
    sobel_y /= magnitude
    return sobel_x, sobel_y, canny

def stroke_width_transform(sobel_x, sobel_y, canny):
    """ Performs the Stroke Width Transform using the image gradients and Canny edge maps.

    Computes for every pixel in the image the width of the stroke the pixel is most likely to be in.
    :param gradient_x, gradient_y, canny: all  NxD ndarray, dtype=float64
    :return: swt_image: NxD ndarray, dtype=float64
    """
    num_of_rows, num_of_cols = sobel_x.shape
    swt = np.full_like(sobel_x, fill_value=np.inf)
    rays = []
    # if looking for black letters on white background, switch directions
    black_letters_on_white_background = True
    if black_letters_on_white_background:
        sobel_x = -sobel_x
        sobel_y = -sobel_y
    # get edge pixels
    edge_pixels_indices = np.nonzero(canny > 0)
    # Iterate over edge points, calculate stroke width
    for y, x in zip(*edge_pixels_indices):
        ray = np.array([[y], [x]])
        grad_x = sobel_x[y, x]
        grad_y = sobel_y[y, x]
        previous_x, previous_y, i = x, y, 0
        # get to next edge pixel
        while True:
            i += 1
            cur_x = math.floor(x + grad_x * i)
            cur_y = math.floor(y + grad_y * i)
            if cur_x != previous_x or cur_y != previous_y:  # moved to next pixel
                if cur_x not in range(num_of_cols) or cur_y not in range(num_of_rows):  # if out of bound
                    break
                previous_x = cur_x
                previous_y = cur_y
                ray = np.c_[ray, [cur_y, cur_x]]
                if canny[cur_y, cur_x] > 0:  # reached an edge
                    v1 = [grad_x, grad_y]
                    v2 = [-sobel_x[cur_y, cur_x], -sobel_y[cur_y, cur_x]]
                    if angle_between(v1=v1, v2=v2) < np.pi / 2.0:
                        stroke_width = np.hypot(cur_x - x, cur_y - y)
                        # set ray swt values to minimum between current values and stroke_width
                        swt[ray[0], ray[1]] = np.minimum(swt[ray[0], ray[1]], stroke_width)
                        rays.append(ray)
                    break
    swt[swt==np.inf] = 0
    return swt, rays

def median_swt(rays, swt):
    for ray in rays:
        ray_swt = swt[ray[0], ray[1]]
        ray_median_swt = np.median(ray_swt)
        swt[ray[0], ray[1]] = np.minimum(ray_swt, ray_median_swt)

def morph_plays(swt_gray, basename):
    # TODO cv2.imwrite: must multiply by 100
    # TODO save_image: PIL can't handle inf
    # TODO display_image: plt can't handle inf
    # TODO conclusion: use cv2.imwrite, and multiply by 100
    # TODO binary: display_image works, beware of range
    from scipy.ndimage import morphology

    # convert to binary
    swt_binary = np.where(swt_gray > 0, 255, 0).astype(np.uint8)
    # save_image(im=swt_binary, new_file_name=file_name + "_bin_PIL.png")
    # cv2.imwrite(img=swt_binary, filename=file_name + "_bin_cv2.png")
    display_image(im=swt_binary, title="bin")

    # diag
    diag_structure = morphology.generate_binary_structure(2, 2)

    dilation = morphology.binary_dilation(swt_binary, structure=diag_structure)
    open = morphology.binary_opening(swt_binary, structure=diag_structure)
    close = morphology.binary_closing(swt_binary, structure=diag_structure)
    close_then_open = morphology.binary_opening(close, structure=diag_structure)
    open_then_close = morphology.binary_closing(open, structure=diag_structure)
    dilation_then_open = morphology.binary_opening(dilation, structure=np.full((5, 5), fill_value=True))
    open_then_dilate = morphology.binary_dilation(open, structure=diag_structure)

    display_image(dilation, "dilation")
    display_image(open, "open")
    display_image(close, "close")
    display_image(close_then_open, "close_then_open")
    display_image(open_then_close, "open_then_close")
    display_image(dilation_then_open, "dilation_then_open")
    display_image(open_then_dilate, "open_then_dilate")


    save_image(im=dilation, new_file_name=os.path.join('morph_post_filter', basename+'_dilation.png'))
    save_image(im=open, new_file_name=os.path.join('morph_post_filter', basename+'_open.png'))
    save_image(im=close, new_file_name=os.path.join('morph_post_filter', basename+'_close.png'))
    save_image(im=close_then_open, new_file_name=os.path.join('morph_post_filter', basename+'_close_then_open.png'))
    save_image(im=open_then_close, new_file_name=os.path.join('morph_post_filter', basename+'_open_then_close.png'))
    save_image(im=dilation_then_open, new_file_name=os.path.join('morph_post_filter', basename+'_dilation_then_open.png'))
    save_image(im=open_then_dilate, new_file_name=os.path.join('morph_post_filter', basename+'_open_then_dilate.png'))

def connect_components(swt):
    label_map, label_to_coords = swt_cc.connect_components(swt=swt)
    if diagnostics:
        for label, ind in label_to_coords.items():
            row_ind, col_ind = ind[0], ind[1]
            temp = np.zeros(swt.shape, dtype=np.uint8)
            temp[(row_ind, col_ind)] = 255
            save_image(im=temp, new_file_name=f"cc2_layer{label}.png")
    return label_map, label_to_coords

def find_letter_candidates(swt, label_to_coords):
    """ Filter non-text components"""
    medians, heights, widths, nw_points = [], [], [], []
    components = []
    for comp_ind, coords in label_to_coords.items():
        ind_row, ind_col =  coords[0], coords[1]
        swt_values = swt[(ind_row, ind_col)]

        # compute stats
        mean = np.mean(swt_values)
        var = np.var(swt_values)
        min_x, max_x = np.min(ind_col), np.max(ind_col)
        min_y , max_y = np.min(ind_row), np.max(ind_row)
        height = max_y - min_y
        width = max_x - min_x
        median = np.median(swt_values)

        # filter based on variance
        if var > (mean / 2):
            continue

        # filter based on height and width
        if height < 15 or width < 15:
            continue

        # aspect_ratio = width / height
        # if aspect_ratio < 0.1 or aspect_ratio > 11:
        #     continue
        # filter based on median and diameter
        # comp_diameter = np.hypot(height, width)
        # if comp_diameter / median > 10:
        #     continue

        components.append(coords)
        medians.append(np.log2(median))
        heights.append(np.log2(height))
        widths.append(width)
        nw_points.append(np.array([min_x, min_y]))

    medians = np.array(medians).reshape(-1, 1)
    heights = np.array(heights).reshape(-1, 1)
    return components, medians, heights, np.asarray(widths), nw_points

def find_letter_pairs(components, medians, heights, widths,  nw_points):
    """ Iterates over all possible combinations of letters, and finds those
    that could possibly be a letter pair, i.e. two letters in the same line
    :param components: dict, mapping component index to its coordinates
    :param medians: flat numpy array of log2 of median stroke width of each componnet
    :param heights: flat numpy array of log2 of height of each componnet
    :param widths: flat numpy array of each component width
    :param nw_points: list of north-west points of each component
    :return: list of letter pairs"""
    # filter out pairs whose height ratio is greater than 2
    height_tree = scipy.spatial.KDTree(heights)
    height_pairs = set(height_tree.query_pairs(1))
    # filter out pairs whose median stroke-width ratio is greater than 2
    median_tree = scipy.spatial.KDTree(medians)
    median_pairs = set(median_tree.query_pairs(1))

    pairs = height_pairs & median_pairs

    angles = []
    angle_pairs = []
    # filter out pairs whose distance between is larger than 3 times the width of the widest
    for (letter1, letter2) in pairs:
        letter1_nw = nw_points[letter1]
        letter2_nw = nw_points[letter2]

        dist = np.linalg.norm(letter1_nw - letter2_nw)
        widest = max(widths[letter1], widths[letter2])
        if dist < (widest * 2):
            angle_pairs.append((letter1, letter2))
            angle = angle_between(letter1_nw, letter2_nw)
            angle = angle + np.pi if angle < 0 else angle
            angles.append(angle)

    angles = np.asanyarray(angles).reshape(-1, 1)
    angles_tree = scipy.spatial.KDTree(angles)
    tree_angle_pairs = angles_tree.query_pairs(np.pi / 12)
    chains = []

    for pair_idx in tree_angle_pairs:
        pair_a = angle_pairs[pair_idx[0]]
        pair_b = angle_pairs[pair_idx[1]]
        left_a = pair_a[0]
        right_a = pair_a[1]
        left_b = pair_b[0]
        right_b = pair_b[1]

        # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
        added = False
        for chain in chains:
            if left_a in chain:
                chain.add(right_a)
                added = True
            elif right_a in chain:
                chain.add(left_a)
                added = True
        if not added:
            chains.append({left_a, right_a})
        added = False
        for chain in chains:
            if left_b in chain:
                chain.add(right_b)
                added = True
            elif right_b in chain:
                chain.add(left_b)
                added = True
        if not added:
            chains.append({left_b, right_b})

    word_images = []
    for chain in [c for c in chains if len(c) > 3]:
        for idx in chain:
            word_images.append(components[idx])
            # cv2.imwrite('keeper'+ str(idx) +'.jpg', images[idx] * 255)
            # final += images[idx]

    return word_images

def get_image_filename():
    filename = "Nope"
    # Trying to get filename from args
    parser = argparse.ArgumentParser(description="detects text regions based on SWT algorithm")
    parser.add_argument("--image_filename", help="path of image", type=str, default=None)
    # TODO add hysteresis FUCK THE POLICE
    args = parser.parse_args()
    if args.image_filename:
        filename = args.image_filename
    return filename

def main():
    # Read Image
    image_filenme = "images/Receipt2.png"
    print(f"started {image_filenme}")
    basename = str(os.path.basename(image_filenme).split(".")[0])
    im = read_image_cv2(filename=image_filenme, mode=cv2.IMREAD_GRAYSCALE, to_0_1_range=False)

# ================================== START ALGORITHM ====================================== #
    # Get edges
    sobel_x, sobel_y, canny = get_edges(im=im)
    t0 = time.clock()

    # Perform swt
    swt, rays = stroke_width_transform(sobel_x=sobel_x, sobel_y=sobel_y, canny=canny)
    median_swt(swt=swt, rays=rays)
    time_at_swt_end = time.clock(); print(f"time to swt: {time_at_swt_end - t0}")

    # Connect Components
    label_map, label_to_coords = connect_components(swt=swt)
    time_at_cc_end = time.clock(); print(f"time to cc: {time_at_cc_end - time_at_swt_end}")
    print(f"# of components before filtering =  {len(label_to_coords)}")
    before_letters = np.where(label_map != 0, 255, 0)
    save_image(im=before_letters, new_file_name=f"letter_filtering/{basename}_before_letters.png")

    # Filter non-text components
    components, medians, heights, nw_points = find_letter_candidates(swt=swt, label_to_coords=label_to_coords)
    print(f"# of components after filtering =  {len(components)}")
    time_at_letters_end = time.clock(); print(f"time to letter filtering: {time_at_letters_end - time_at_cc_end}")
    after_letters = np.zeros(swt.shape, dtype=np.uint8)
    for coords in components.values():
        after_letters[coords[0], coords[1]] = 255
    save_image(im=after_letters, new_file_name=f"letter_filtering/{basename}_after_letters.png")
    # Find letter pairs
    pairs = find_letter_pairs(components=components, medians=medians, heights=heights, nw_points=nw_points)
    # Chain letter pairs

# ================================== END ALGORITHM ====================================== #
    # save
    np.save(os.path.join('npy',basename + '.npy'), swt)
    cv2.imwrite(os.path.join('swt_results', basename + '.png'), swt * 200)

def tst():
    basename = 'freedom'
    swt = np.load(os.path.join('npy', basename+".npy"))

    # Connect Components
    label_map, label_to_coords = connect_components(swt=swt)
    # np.save(file=os.path.join('post_cc_npy', basename+'_label_map.npy'), arr=label_map)
    # with open('post_cc_npy/'+basename+'_label_to_coords.pickle', 'wb') as handle:
    #     pickle.dump(label_to_coords, handle, protocol=pickle.HIGHEST_PROTOCOL)
    time_at_cc_end = time.clock(); print(f"time to cc: {time_at_cc_end}")
    print(f"# of components before filtering: =  {len(label_to_coords)}")
    before_letters = np.where(label_map != 0, 255, 0)
    save_image(im=before_letters, new_file_name=f"letter_filtering/{basename}_before_letters.png")

    # Filter non-text components
    components, medians, heights, widths, nw_points = find_letter_candidates(swt=swt, label_to_coords=label_to_coords)
    print(f"of components after filtering =  {len(components)}")
    time_at_letters_end = time.clock();print(f"time to letter filtering: {time_at_letters_end - time_at_cc_end}")
    after_letters = np.zeros(swt.shape, dtype=np.uint8)
    for coords in components:
        after_letters[coords[0], coords[1]] = 255
    save_image(im=after_letters, new_file_name=f"letter_filtering/{basename}_after_letters.png")
    if len(components) < 2:
        print(f"found only {len(components)} letter,"); sys.exit()
    # Find letter pairs
    word_images = find_letter_pairs(components=components, medians=medians, heights=heights, widths=widths, nw_points=nw_points)
    time_at_pairs_end = time.clock();print(f"time to pair and chain: {time_at_pairs_end - time_at_letters_end}")

    final_result = np.zeros(swt.shape, dtype=np.uint8)
    for coords in word_images:
        final_result[coords[0], coords[1]] = 255
    morph_plays(swt_gray=final_result, basename=basename)
    save_image(im=final_result, new_file_name=os.path.join('final_results', basename+'.png'))


diagnostics = False
tst()