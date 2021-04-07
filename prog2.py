import numpy as np
import matplotlib.pyplot as plt

# We will use os to iterate through different directories and join paths, etc.
import os

# cv2 will be used to carry out image operations
import cv2


# Specifying a data directory. Be sure to change this to the directory
#   where you have your datasets
DATADIR = "/Users/homeresidence/Desktop/Development/DeepLearning/PetImages"
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)      # provide the complete path to cats or dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)       # read the images and convert to grayscale
                                                                                    # converting to grayscale reduces the datasize
                                                                                    # to almost 1/3 of original.
        plt.imshow(img_array, cmap="gray")
        # plt.show()
        break
    break

# We can check the shape of image array
print(img_array.shape)      

# The problem we face now is that all the images are not of the same shape in the dataset. So now we have to reduce all of them to
# same shape. But keep in mind that reducing the images too much will distort them in a manner, that they can't be read.

# Now we are ready to create training data
training_data = []