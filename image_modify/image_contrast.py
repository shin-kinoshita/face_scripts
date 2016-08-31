import os
import math
import numpy as np
import cv2

def import_image(path):
    image = cv2.imread(path)
    return image
    
def contrast_image(image, c=10.0):
    lut = [np.uint8(255.0 / (1 + math.exp(-c * (i - 128.) / 255.))) for i in range(256)] 
    image_contrast = np.array([lut[value] for value in image.flat], dtype=np.uint8)
    image_contrast = image_contrast.reshape(image.shape)
    return image_contrast

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = '../data/shin'
    output_dir_path = '../data_test/shin'

    file_name_list = os.listdir(input_dir_path)

    for f_name in file_name_list:
        input_img = import_image(input_dir_path + '/' + f_name)
        output_img = contrast_image(input_img)
        export_image(output_dir_path + '/' + f_name, output_img)

    return

if __name__ == '__main__':
    main()

