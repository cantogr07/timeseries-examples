import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# read ucr data
def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t") # data is tab separated
    y = data[:, 0] # extract the first column aka labels
    x = data[:, 1:] # extract the rest of the columns
    return x, y.astype(int) # convert y to integers


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")

x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

classes = np.unique(np.concatenate((y_train, y_test), axis=0)) # concatenate the two arrays and remove duplicates of labels


# plot the data
plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()






# ---------------------------------------------------------------------------------------------
# make data multivariate
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1)) # reshape to (n_samples, n_timesteps, n_variables)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1)) # also reshape the test data so that it has the same shape as the training data


# loss function -> what is the difference between the predicted and the actual value (less the better)
# count classes to use loss function sparse_categorical_crossentropy
num_classes = len(np.unique(y_train))

# shuffle the training set to use validation_split afterwards 
# it is basically saying how much of the training data should be used for validation
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


# convert labels to positive integers (expected labels: 0,1)
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0






# --------------------------------------BUILD THE MODEL---------------------------------------
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape) # input layer


    # 1D convolutional layers
    # kernel_size = 3 -> 3 time steps (region of input that the filter considers at a time)
    # padding = same -> output has the same length as the original input
    # BatchNormalization -> normalize the activations of the previous layer at each batch
    # ReLU -> activation function
        # ReLU(x) = max(0, x). It simply outputs the input value if it is positive, and zero otherwise.
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    # reduces the spatial dimensions
    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    # output layer
    # Dense -> fully connected layer
    # softmax -> activation function
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    # return a new model instance
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
tf.keras.utils.plot_model(model, show_shapes=True)




# --------------------------------------TRAIN THE MODEL---------------------------------------
epochs = 500 # epoch is one complete pass through the training data
batch_size = 32 # num of training examples that are processed together

# define a list of callback functions
callbacks = [

    # save the best model
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),

    # reduce learning rate when a validation loss has stopped improving
    # patience -> number of epochs with no improvement 
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001 
    ),

    # refrain from overfitting
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

# optimizer -> how weights are updated
# loss -> error between true and predicted values
# metrics -> used to monitor the training and testing steps
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

# train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2, # set aside 20% of the training data for validation
    verbose=1, # progress bar
)

model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)




