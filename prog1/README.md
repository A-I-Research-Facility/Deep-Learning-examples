    import tensorflow as tf
    
***
28x28 images of hand written digits from 0 to 9 :-

    mnist = tf.keras.datasets.mnist

Loading training and testing data :-

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

***
💢 Normalizing training data to get tensor values between 0 and 1
We dont have to do this always but it makes it easier for neural net to learn
Normalization is a good measure to avoid overflow. Axis refers to 'column' or 'row'.

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

***
💢 We are creating a sequential type model (it is a feed forward, most common type of model) :-

    model = tf.keras.models.Sequential()

***
💢 Flattening the data (we currently have a multi dimensional array and we dont want that)
We can use numpy's reshape function to flatten too, but here we do it directly.

    model.add(tf.keras.layers.Flatten()) 

***
💢 We need to have this in layer type so that if we have a convolusional neural network,
at the end of the network there will be a densely connected layer, so we need to flatten 
it before that layer.

Hidden layer 1, 128 => neurons, rectilinear activation function

    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    
Hidden layer 2, similar to hidden layer 1   
    
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))        # 
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))        # output layer, 10 => number of classifications(0-9), softmax activation
                                                                   # function(because we have a probability distribution)
'''
Defining the parameters for training of the model.
Optimizer is the most complex part of the neural network. If we are familiar with gradient-descent,
we can pass something like stochastic_gradient_descent, otherwise, 'adam' is the goto optimizer.
Loss is the degree of error. NN doesn't try to maximize the accuracy, it always tries to minimize loss.
Thus, the way we calculate loss makes a huge impact on training.
Finally, the metrics that we want to track = 'accuracy' only.
'''

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Now we are ready to train the model.
model.fit(x_train, y_train, epochs = 3)

'''
Calculating the validation loss (the model shouldn't overfit, instead of learning the images of number,
it needs to learn what makes up that particular number)
'''
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)      
'''
the loss is expected to be slightly higher and accuracy is expected to be slightly lower.
What we don't want to see is too little or too much delta, beacuse that means we have
overfit the model.
'''

# Now we can save the model
model.save('number_reader.model')

# To reload the model
new_model = tf.keras.models.load_model('number_reader.model')

# To make a prediction
predictions = new_model.predict([x_test])       
'''
Always remember that predict() always take list as argument
The result of this prediction is tensor format. In order to read
it in easy language, we can use numpy
'''

# Example prediction and verification with our eyes
import numpy as np
import matplotlib.pyplot as plt
print(np.argmax(predictions[0]))        # Example prediction of x_test[0]
plt.imshow(x_test[0])
plt.show()
