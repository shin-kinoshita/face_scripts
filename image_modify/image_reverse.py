import os
import numpy as np
import itertools
import cv2

def import_image(path):
    image = cv2.imread(path)
    return image
    
def reverse_image(image):
    new_image = cv2.flip(image, 1)
    return new_image

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = '../data/shin'
    output_dir_path = '../data_modified/shin'

    file_name_list = os.listdir(input_dir_path)

    for f_name in file_name_list:
        input_img = import_image(input_dir_path + '/' + f_name)
        output_img = reverse_image(input_img)
        export_image(output_dir_path + '/' + f_name, output_img)

    return

if __name__ == '__main__':
    main()

