'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''
import os
import cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    # read the first training image
    folder = 'data/training/Non_Bin_Blue'
    i = 0
    for filename in os.listdir(folder):

        # X[i] = img[0, 0].astype(np.float64)/255
        # i += 1

        # filename = '0001.jpg'
        img = cv2.imread(os.path.join(folder, filename))
        img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img_1)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')

    # get the image mask
        mask = my_roi.get_mask(img_1)

    # display the labeled region and the image mask
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img_1[mask, :].shape[0])

        ax1.imshow(img_1)
        ax1.add_line(plt.Line2D(
            my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)

        plt.show(block=True)

        rgb_buffer = img_1[mask == 1].astype(np.float64)/255
        yuv_buffer = img_2[mask == 1].astype(np.float64)/255

        
        if i == 0:
            rgb_m = rgb_buffer
            yuv_m = yuv_buffer
            i = i+1
        else:
            rgb_m = np.append(rgb_m, rgb_buffer, axis=0)
            yuv_m = np.append(yuv_m, yuv_buffer, axis=0)
            
    

    np.savetxt('non_bin_blue_rgb.txt', rgb_m, fmt="%s")
    np.savetxt('non_bin_blue_yuv.txt', yuv_m, fmt="%s")
