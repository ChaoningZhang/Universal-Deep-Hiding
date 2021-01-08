import torch
import torch.nn as nn
from noise_layers.crop import get_random_rectangle_inside
import numpy as np

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

def get_center_crop(image):
    image_height = image.shape[2]
    image_width = image.shape[3]

    h_start = image_height // 4
    h_end = h_start + image_height // 2

    w_start = image_width // 4
    w_end = w_start + image_width // 2

    return h_start, h_end, w_start, w_end

def get_random_rectangle_inside_fixed(height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = 128
    image_width = 128

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width

class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        super(Cropout, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range
        self.h_start, self.h_end, self.w_start, self.w_end = get_random_rectangle_inside_fixed(self.height_ratio_range, self.width_ratio_range)

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]
        assert noised_image.shape == cover_image.shape

        cropout_mask = torch.zeros_like(noised_image)
        # h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
        #                                                              height_ratio_range=self.height_ratio_range,
        #                                                              width_ratio_range=self.width_ratio_range)
        h_start, h_end, w_start, w_end = get_center_crop(noised_image)

        # h_start, h_end, w_start, w_end = self.h_start, self.h_end, self.w_start, self.w_end

        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        noised_and_cover[0] = noised_image * cropout_mask + cover_image * (1-cropout_mask)
        return  noised_and_cover

    def get_crop_coords(self):
        return self.h_start, self.h_end, self.w_start, self.w_end