import os
import numpy as np
import cv2
import itertools

def import_image(path):
    image = cv2.imread(path)
    return image
    
def gray_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image_gray

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = '../data/shin'
    output_dir_path = '../data_gray/shin'

    file_name_list = os.listdir(input_dir_path)

    for f_name in file_name_list:
        input_img = import_image(input_dir_path + '/' + f_name)
        output_img = gray_image(input_img)
        export_image(output_dir_path + '/' + f_name, output_img)

    return

if __name__ == '__main__':
    main()

