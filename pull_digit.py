import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

from skimage import measure
from skimage import morphology

def labeled_user_image(image,k = 0):

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(imgray, 90, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #im_bw= cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    im_bw = np.array(255-im_bw, dtype=bool)
    cleaned = morphology.remove_small_objects(im_bw, min_size=20, connectivity=2)
    cleaned = np.array(cleaned, dtype=int)
    cleaned = 255+cleaned
    label, n = measure.label(cleaned, connectivity=2, background=255, return_num=True)
    print("numbers of numpers on image : ",n)
    x = []
    y = []
    numbers = []
    ph=[]
    rect=[]
    for i in range(1, n + 1):
        for r in range(label.shape[0]):
            for c in range(label.shape[1]):
                if label[r, c] == i:
                    x.append(r)
                    y.append(c)

        digit = im_bw[min(x): max(x), min(y): max(y)]

        rect.append([(min(y), min(x)), (max(y) - min(y)), (max(x) - min(x))])
        padd_y = 0
        padding = np.zeros([digit.shape[0]+padd_y, digit.shape[1] + k], dtype='float64')
        padding[padd_y//2:padding.shape[0]-padd_y//2, k//2:padding.shape[1] - k//2] = digit
        ph.append(padding)

        if not padding.size:
            # Handle empty image, for example, print a warning and skip processing
            print("Warning: Empty image")
        else:
            re_digit = cv2.resize(np.array(padding, dtype='float64'), (28, 28), interpolation=cv2.INTER_AREA)

            # Calculate the HOG features
            roi_hog_fd = hog(re_digit, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))

            numbers.append( np.array([roi_hog_fd], 'float64'))
        x = []
        y = []

    return numbers,ph,rect

#im = cv2.imread(r'test_images/numbers.png')
# nums =labeled_user_image(im)
# plt.show()
