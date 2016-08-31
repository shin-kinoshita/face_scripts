import os
import math
import numpy as np
import cv2

def import_image(path):
    image = cv2.imread(path)
    return image
    
def contrast_image(image, c=10.0):
    under_thresh = 80
    upper_thresh = 145
    maxValue = 255
    th, drop_back = cv2.threshold(image, under_thresh, maxValue, cv2.THRESH_BINARY)
    th, clarify_born = cv2.threshold(image, upper_thresh, maxValue, cv2.THRESH_BINARY_INV)
    image_merged = np.minimum(drop_back, clarify_born)
    return image_merged

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = '../data_gray/shin'
    output_dir_path = '../data_test/shin'

    file_name_list = os.listdir(input_dir_path)

    for f_name in file_name_list:
        input_img = import_image(input_dir_path + '/' + f_name)
        output_img = contrast_image(input_img, 10)
        export_image(output_dir_path + '/' + f_name, output_img)

    return

if __name__ == '__main__':
    main()

