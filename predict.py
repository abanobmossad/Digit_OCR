import cv2
from train_model import new,view

# digit.jpg , numbers.png , [nums.png] , im.jpg , index.jpg ,photo_2.jpg , 2.png
image = cv2.imread('test_images/digit.jpg')
padd = 10


if __name__ == '__main__':
    new(image, padd)
    view(image, padd)


