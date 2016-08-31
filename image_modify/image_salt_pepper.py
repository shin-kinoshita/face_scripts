import os
import numpy as np
import cv2

def import_image(path):
    image = cv2.imread(path)
    return image
    
def salt_pepper_image(image, gamma=2.0):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    image_sp = image.copy()

    # 塩モード
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt)) for i in image.shape]
    image_sp[coords[:-1]] = (255,255,255)

    # 胡椒モード
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in image.shape]
    image_sp[coords[:-1]] = (0,0,0)

    return image_sp

def export_image(path, image):
    cv2.imwrite(path, image)
    return

def main():
    input_dir_path = '../data/shin'
    output_dir_path = '../data_test/shin'

    file_name_list = os.listdir(input_dir_path)

    for f_name in file_name_list:
        input_img = import_image(input_dir_path + '/' + f_name)
        output_img = salt_pepper_image(input_img, 2.0)
        export_image(output_dir_path + '/' + f_name, output_img)

    return

if __name__ == '__main__':
    main()

