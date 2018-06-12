from matplotlib import pyplot as plt
from skimage.feature import hog
from matplotlib import patches
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from pull_digit import labeled_user_image
import cv2

mnist = fetch_mldata('mnist-original', data_home="dataset")

x = np.array(mnist.data, 'int16')
y = np.array(mnist.target, 'int')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)


def train_model(x_train, y_train):
    list_hog_fd = []
    for feature in x_train:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False)
        list_hog_fd.append(fd)

    x_train = np.array(list_hog_fd, 'float64')

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    # save the model to disk
    filename = 'digit_knn_model.sav'
    pickle.dump(knn, open(filename, 'wb'))


def knn_score(x_test, y_test):
    filename = 'digit_knn_model.sav'
    # load the model from disk
    knn = pickle.load(open(filename, 'rb'))
    list_hog_fd = []
    for feature in x_test:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False)
        list_hog_fd.append(fd)

    x_test = np.array(list_hog_fd, 'float64')

    score = knn.score(x_test, y_test)
    print("score is : ")
    print(np.round(score * 100, 2), "%")

    return x_test, y_test


def knn_predict(image):
    filename = 'digit_knn_model.sav'
    # load the model from disk
    knn = pickle.load(open(filename, 'rb'))
    expected = knn.predict(image)
    return expected[0]


def read_image(rects, nums):
    xx = []
    yy = []
    for n in range(len(nums)):
        x = rects[n][0][0]
        y = rects[n][0][1]
        xx.append(x)
        yy.append(y)

    max_x = np.max(xx)
    max_y = np.max(yy)

    digits = np.zeros((max_x + 5, max_y + 5), np.object)

    return digits


def get_string(arr):
    dd = []
    for x in range(arr.shape[0] - 1):
        for y in range(arr.shape[1] - 1):
            if type(arr[x][y]) == list:
                dd.append(arr[x][y][0])

    print("the numers in the image are :")
    print(', '.join(str(x) for x in dd))


def new(image, padd):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    nums, ph, rects = labeled_user_image(image, padd)
    digits = read_image(rects, nums)
    for n in range(len(nums)):
        rect = patches.Rectangle(rects[n][0], rects[n][1], rects[n][2], linewidth=1, edgecolor='g',
                                 facecolor='none')

        ax.add_patch(rect)
        ex = knn_predict(nums[n])

        digits[rects[n][0][0]][rects[n][0][1]] = [ex]
        ax.text(rects[n][0][0] + 3, rects[n][0]
                [1] - 3, str(int(ex)), style='italic')

        plt.axis("off")

    get_string(digits)
    plt.show()


def view(image, pad):
    nums, ph, rects = labeled_user_image(image, pad)
    plt.figure()
    for n in range(len(ph)):
        plt.subplot(8, 11, n + 1)
        plt.imshow(ph[n], "gray")
        ex = knn_predict(nums[n])
        title_obj = plt.title(str(ex))
        plt.setp(title_obj, color='r')
        plt.axis("off")
    plt.show()


def main():
   # train the model and save it to "digit_knn_model"
    train_model(x_train, y_train)
    # get the socer of the model
    im, t = knn_score(x_test, y_test)

if __name__ == '__main__':
    main()
