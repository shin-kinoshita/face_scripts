import os
import numpy as np
import cv2
import sys
sys.path.append('./image_modify')
import image_gamma
import image_gray 
import image_contrast
import image_reverse
import image_average 
import image_gauss
import image_salt_pepper

def import_image(path):
    image = cv2.imread(path)
    return image

def modify_image(path_list, image_list, mode):
    func_list = [image_gamma.gamma_image,
                 #image_gray.gray_image,
                 image_contrast.contrast_image,
                 image_reverse.reverse_image,
                 image_average.average_image,
                 image_gauss.gauss_image,
                 image_salt_pepper.salt_pepper_image]

    for (path, image) in zip(path_list, image_list):
        ext = path.rsplit('.', 1)[1]
        new_image = func_list[mode](image)
        new_path = path.replace('.' + ext, '') + '_' + str(mode) + '.' + ext

        path_list.append(new_path)
        image_list.append(new_image)
        export_image(new_path, new_image)
    if mode == len(func_list) - 1:
        return
    modify_image(path_list, image_list, mode+1)

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = './data/'
    output_dir_path = './data_modified/'

    people_list = os.listdir(input_dir_path)

    for person in people_list:
        file_name_list = os.listdir(input_dir_path + '/' + person)
        for f_name in file_name_list:
            input_img = import_image(input_dir_path + '/' + person + '/' + f_name)
            output_path = output_dir_path + '/' + person + '/' + f_name
            export_image(output_path, input_img)
            modify_image([output_path], [input_img], 0)

    return

if __name__ == '__main__':
    main()

