from keras.models import load_model
import numpy as np
import cv2
import math
import copy
import pylab
import seaborn as sns
import matplotlib.pyplot as plt
from ImagePreprocessing import process_and_load_image
import argparse

def Occlusion_exp(image_path, model_path, occluding_size, occluding_pixel, occluding_stride):
    model = load_model(model_path)

    img_tensor = process_and_load_image(image_path)
    out = model.predict(img_tensor)
    out = out[0]
    # Getting the index of the winning class:
    m = max(out)
    index_object = [i for i, j in enumerate(out) if j == m]
    height, width, _ = img.shape
    output_height = int(math.ceil((height-occluding_size) / occluding_stride + 1))
    output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
    heatmap = np.zeros((output_height, output_width))
    
    for h in range(output_height):
        for w in range(output_width):
            # Occluder region:
            h_start = h * occluding_stride
            w_start = w * occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)
            # Getting the image copy, applying the occluding window and classifying it again:
            input_image = copy.copy(img)
            input_image[h_start:h_end, w_start:w_end,:] =  occluding_pixel            
            heatmap_img = input_image/255
            heatmap_img = cv2.resize(heatmap_img,(150,150))
            img_tensor = np.reshape(heatmap_img,[1, 150,150,3])
            out = model.predict(img_tensor)
            out = out[0]
            print('scanning position (%s, %s)'%(h,w))

            prob = (out[index_object]) 
            heatmap[h,w] = prob
    
    f = pylab.figure()
    f.add_subplot(1, 2, 2)  # this line outputs images side-by-side    
    ax = sns.heatmap(heatmap, xticklabels=False, yticklabels=False)
    f.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.show()
    print ( 'Object index is %s'%index_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="creates heatmap of image based on models sensitivity to occlusion")

    parser.add_argument('model_path',
                        help='path to model.h5 file')
    parser.add_argument('image_path',
                        help='path to image file')

    #Optional
    parser.add_argument('--occluding_size',
                        type=int,
                        default = 10,
                        help='set occluding size')

    parser.add_argument('--occluding_stride',
                        type=int,
                        default = 5,
                        help='set occluding stride')

    parser.add_argument('--occluding_pixel',
                        type=int,
                        default = 0,
                        help='set occluding pixel')
    
    args = parser.parse_args()

    model_path       = args.model_path
    image_path       = args.image_path
    occluding_size   = args.occluding_size
    occluding_stride = args.occluding_stride
    occluding_pixel  = args.occluding_pixel

    Occlusion_exp(image_path, model_path, occluding_size, occluding_pixel, occluding_stride)








model = load_model("/content/model2.h5")
for i in range(102, 120):
  image_path = '/content/Damaged-or-not-v1/Test/'+str(i)+'.jpg'
  # Enter here the parameters of the occluding window:
  occluding_size = 10
  occluding_pixel = 0
  occluding_stride = 5
  Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride)