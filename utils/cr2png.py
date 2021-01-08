#!/usr/bin/env python

# import what we need
import numpy
import os
import glob
import time
import argparse
from PIL import Image
from rawkit import raw

# # dirs and files
# raw_file_type = ".CR2"
# raw_dir = args.source + '/'
# converted_dir = args.destination + '/'
# raw_images = glob.glob(raw_dir + '*' + raw_file_type)

# converter function which iterates through list of files
def cr2png(src_folder, dst_folder):
    raw_images = os.listdir(src_folder)
    # import pdb; pdb.set_trace()

    for raw_image in raw_images:
        raw_image = src_folder + "/" + raw_image
        print ("Converting the following raw image: " + raw_image + " to PNG")

        # file vars
        file_name = os.path.basename(raw_image)
        file_without_ext = os.path.splitext(file_name)[0]
        file_timestamp = os.path.getmtime(raw_image)

        # parse CR2 image
        raw_image_process = raw.Raw(raw_image)
        buffered_image = numpy.array(raw_image_process.to_buffer())

        # check orientation due to PIL image stretch issue
        if raw_image_process.metadata.orientation == 0:
            png_image_height = raw_image_process.metadata.height
            png_image_width = raw_image_process.metadata.width
        else:
            png_image_height = raw_image_process.metadata.width
            png_image_width = raw_image_process.metadata.height

        # prep PNG details
        png_image_location = dst_folder + "/" + file_without_ext + '.png'
        png_image = Image.frombytes('RGB', (png_image_width, png_image_height), buffered_image)
        png_image.save(png_image_location, format="png")

        # update PNG file timestamp to match CR2
        os.utime(png_image_location, (file_timestamp,file_timestamp))

        # close to prevent too many open files error
        png_image.close()
        raw_image_process.close()