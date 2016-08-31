import cv2
import numpy as np
def capture_camera(mirror=True, size=None):
    """Capture video from camera"""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if mirror is True:
            frame = frame[:,::-1]
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)
            frame = cv2.putText(frame,"Hello!!!",(1, 1),cv2.FONT_HERSHEY_PLAIN, 3 ,(255,255,0))

        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


#capture_camera(size = (1280, 720))
#img = cv2.imread('lena.png')
img = cv2.imread('anitha0.png') #load rgb image
gamma = 1.5
look_up_table = np.ones((256, 1), dtype = 'uint8') * 0
for i in range(256):
    look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
img_gamma = cv2.LUT(img, look_up_table)
img_gamma = cv2.resize(img_gamma, (56,56))
cv2.imwrite("image_processed.jpg", img_gamma)
