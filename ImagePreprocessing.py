import cv2
import numpy as np

def process_and_load_image(image_path):
    img = cv2.imread(image_path)
    if(img is None):
        print("Image object could not load")
        return
    img = img/255
    img = cv2.resize(img,(150,150))
    img_tensor = np.reshape(img,[1, 150,150,3])
    return img_tensor