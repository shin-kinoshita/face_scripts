import os
import numpy as np
import cv2

def import_image(path):
    image = cv2.imread(path)
    return image
    
def gamma_image(image, gamma=2.0):
    look_up_table = np.ones((256, 1), dtype = 'uint8') * 0
    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    
    look_up_table = np.array(look_up_table)
    image_gamma = cv2.LUT(image, look_up_table)
    return image_gamma

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = '../data/shin'
    output_dir_path = '../data_modified/shin'

    file_name_list = os.listdir(input_dir_path)

    for f_name in file_name_list:
        input_img = import_image(input_dir_path + '/' + f_name)
        output_img = gamma_image(input_img, 2.0)
        export_image(output_dir_path + '/' + f_name, output_img)

    return

if __name__ == '__main__':
    main()

