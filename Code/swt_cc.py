from collections import defaultdict
# ========================================== IMPORTS ===================================================== #
import numpy as np

class UF(object):
    """ An implementation of Union-Find using path compression and union by rank"""

    def __init__(self):
        self.int_to_label_dict = {}

    class Label(object):
        """ ld: Tree that keeps mapping from ints to Labels"""

        def __init__(self, value):
            self.value = value
            self.parent = self
            self.rank = 0

        def __eq__(self, other):
            if type(other) is type(self):
                return self.value == other.value
            else:
                return False

        def __ne__(self, other):
            return not self.__eq__(other)

    def MakeSet(self, x):
        """ :param x: int, representing a label"""
        try:
            return self.int_to_label_dict[x]
        except KeyError:  # create new Label object
            label = UF.Label(x)
            self.int_to_label_dict[x] = label
            return label

    @staticmethod
    def Find(item):
        """ Find using path compression
        :param item: Label object of type Label"""
        parent = item
        to_be_reparented = []
        while parent != parent.parent:
            to_be_reparented.append(parent)
            parent = parent.parent
        for label in to_be_reparented:
            label.parent = parent
        return parent

    @staticmethod
    def Union(label_x, label_y):
        """ Union by rank
        :param x, label_y: Label objects
            :return: Label object, which is root node of new union tree """
        x_root = UF.Find(label_x)
        y_root = UF.Find(label_y)
        if x_root == y_root:
            return x_root

        if x_root.rank < y_root.rank:
            x_root.parent = y_root
            return y_root
        elif y_root.rank < x_root.rank:
            y_root.parent = x_root
            return x_root
        else:  # equal rank, choose whichever one
            y_root.parent = x_root
            x_root.rank += 1
            return x_root



def get_connected_neighbors(swt, x, y):
    possible_neighbors = [(y, x - 1),  # west
                          (y - 1, x - 1),  # nw
                          (y - 1, x),  # n
                          (y - 1, x + 1)]  # ne
    # check bounds
    num_of_rows, num_of_cols = swt.shape
    neighbors = [(y, x) for (y, x) in possible_neighbors if
                 (y in range(num_of_rows) and x in range(num_of_cols) and swt[y, x] > 0)]
    # check connection
    connected_neighbors = [pixel for pixel in neighbors if 1.0 / 3 < swt[pixel] / swt[y, x] < 3]
    return connected_neighbors

def label_pixels(label_map, pixels_to_label, label_value):
    for pixel in pixels_to_label:
        label_map[pixel] = label_value

def get_neighbors_labels(uf_tree, neighbors_coords, label_map):
    """ Return a list containing non-zero labels of neighbors if they're labeled. Returns empty list otherwise"""
    neighbors_labels = []
    for neighbor_coord in neighbors_coords:
        neighbor_label_value = label_map[neighbor_coord]
        if neighbor_label_value != 0:
            neighbor_label_object = uf_tree.MakeSet(neighbor_label_value)
            neighbors_labels.append(neighbor_label_object)

    return neighbors_labels

def connect_components(swt):
    """ Detect and label legally-connected components"""
    label_map = np.zeros_like(swt, dtype=np.uint16)
    uf_tree = UF()
    next_label = 1
    positive_indices = np.nonzero(swt)
    # iterate over all pixels with positive stroke, find connected neighbors, and label accordingly
    for y, x in zip(*positive_indices):
        connected_neighbors = get_connected_neighbors(swt=swt, x=x, y=y)
        # if this pixel doesn't have connected neighbors, then it isn't part of a component
        if len(connected_neighbors) == 0:
            continue
        neighbors_labels = get_neighbors_labels(uf_tree=uf_tree, neighbors_coords=connected_neighbors, label_map=label_map)
        connected_neighbors.append((y, x))

        # if all neighbors don't have a label, create a new label for them, and assign them this label in label_map
        if len(neighbors_labels) == 0:
            uf_tree.Label(value=next_label)
            label_pixels(label_map=label_map, pixels_to_label=connected_neighbors, label_value=next_label)
            next_label += 1

        # else, if some of the neighbors have labels
        else:
            # Unionize all labels
            for neighbor_label in neighbors_labels:
                uf_tree.Union(neighbors_labels[0], neighbor_label)
            # Label all pixels
            label_value = uf_tree.Find(neighbors_labels[0]).value
            label_pixels(label_map=label_map, pixels_to_label=connected_neighbors, label_value=label_value)

    # second pass, reassign values in label_map based on label's parent
    label_to_coords = defaultdict(lambda: [np.array([], dtype=np.int64), np.array([], dtype=np.int64)])
    for i in range(1, next_label):
        # For each label, get its parent
        cur_label_object = uf_tree.MakeSet(i)
        cur_label_parent = uf_tree.Find(cur_label_object)
        cur_label_parent_value = cur_label_parent.value
        # reassign all indices by their parent's label
        row_ind, col_ind = np.nonzero(label_map==i)
        if row_ind.size == 0:
            continue
        label_to_coords[cur_label_parent_value][0] = np.hstack((label_to_coords[cur_label_parent_value][0], row_ind))
        label_to_coords[cur_label_parent_value][1] = np.hstack((label_to_coords[cur_label_parent_value][1], col_ind))
        label_map[(row_ind, col_ind)] = cur_label_parent_value
    return label_map, label_to_coords