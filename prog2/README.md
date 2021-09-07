    import numpy as np
    import matplotlib.pyplot as plt

***
💢 We will use os to iterate through different directories and join paths, etc.
    
    import os

***
💢 Library cv2 will be used to carry out image operations.

    import cv2

***
💢 Specifying a data directory. Be sure to change this to the directory where you have your datasets.

    DATADIR = "/Users/homeresidence/Desktop/Development/DeepLearning/PetImages"
    CATEGORIES = ["Dog", "Cat"]

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)      # provide the complete path to cats or dogs dir
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)       # read the images and convert to grayscale 
                                                                                    # converting to grayscale reduces the datasize 
                                                                                    # to almost 1/3 of original. 
        #     plt.imshow(img_array, cmap="gray")
        #     plt.show()
        #     break
        # break

***
💢 We can check the shape of image array.

    print(img_array.shape)      

***
💢 The problem we face now is that all the images are not of the same shape in the dataset. So now we have to reduce all of them to 
same shape. But keep in mind that reducing the images too much will distort them in a manner, that they can't be read. So, we 
carefully decide a size.


    IMG_SIZE = 80
    # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     # this changes size to IMG_SIZE x IMG_SIZE
    #                                                             # so we get a square shape image always
    # plt.imshow(new_array, cmap='gray')
    # plt.show()

***
💢 After trying different sizes, 80 x 80 seems to be the best fit to recognize the image.

Now we are ready to create training data.

    training_data = []
    def create_training_data():
        # same code loop as above 
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            # Now, we got the features as numbers but the classification(label) is still not a number. 
            # We have to map things to a numerical value. So we decide that 0 is dog, and 1 is cat. 
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:                            # we are using try and except because some of the images are broken in this dataset 
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass

    create_training_data()
    # print(len(training_data))

***
💢 Now we need to focus on balancing our training data. We always need to make sure that all of our classification 
categories, have equal probability distribution. Since in our current dataset we have 2 classifications, i.e, 
cats and dogs, we need to make sure that our training data has 50% cats and 50% dogs. 

Sometimes it may happen that we don't have equal data in all of our categories. In that case, class weights are 
assigned. The mechanism behind it is that the loss is the handled differently in order to make up for imabalabce 
in the dataset.

Moving on, we need to shuffle the data. Otherwise the neural network will go through a whole category at once 
and learn everything to be category 1 only. Then when it will reach category 2, it will be always wrong and 
so it will assign everything to be of category 2 only. All of this will make our neural network completely 
useless. Thus we have to shuffle the data. 


    import random

    random.shuffle(training_data)

***
💢 To check whether the data is shuffled.
    
    # for sample in training_data:
    #     print(sample[1])

***
💢 Since the data is now shuffled, we need to packet it into the variables 
that we will use just before feeding them into the neural network.
'''
X = []      # feature set
y = []      # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

# We can't pass a list to neural network so we need to convert X to a numpy array
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # the first argument of reshape defines how many features we have 
                                                    # -1 represents "any number", thus there can be any number of features 
                                                    # and we are not concerned about it. The last argument defines the number 
                                                    # of colours, so, '1' stands for grayscale. If it were rgb, we would have 
                                                    # written '3'
'''
We don't wan't to rebuild the dataset multiple times. In this program we are working with a 
very simple and straightforward dataset, whose features and nodes are all simple. In general, 
when we work with datasets, we have to tweak it. Thus we need to save our dataset after tweaking 
in order to avoid building it again and again from scratch. The following commands can be used 
to save dataset
'''

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


'In order to read the datasets in future'

# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)
