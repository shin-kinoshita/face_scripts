import os
import numpy as np
import cv2

def import_image(path):
    image = cv2.imread(path)
    return image
    
def gauss_image(image, gamma=2.0):
    row, col, ch= image.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    image_gauss = image + gauss
    return image_gauss

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = '../data/shin'
    output_dir_path = '../data_test/shin'

    file_name_list = os.listdir(input_dir_path)

    for f_name in file_name_list:
        input_img = import_image(input_dir_path + '/' + f_name)
        print f_name
        output_img = gauss_image(input_img, 2.0)
        export_image(output_dir_path + '/' + f_name, output_img)

    return

if __name__ == '__main__':
    main()

