# /usr/bin/python3.6
# coding:utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
import binascii
from sklearn import preprocessing

PNG_SIZE = 28
evaluate_file_path = r'E:\PycharmProjects\pythonProject\gas_final_evaluate.csv'
train_file_path = r'E:\PycharmProjects\pythonProject\gas_final_train.csv'
number_classes = 8
epochs = 10


def create_DNN_model():
    model = tf.keras.Sequential()
    model.add(keras.layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(keras.layers.Dense(8, activation='relu', name='dense1'))
    model.add(keras.layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(keras.layers.Dense(26, activation='relu', name='dense2'))
    model.add(keras.layers.Dense(8, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def create_cnn_model():
    model = tf.keras.Sequential()
    # Conv layer 1 output shape (32, 28, 28)
    model.add(
        keras.layers.Convolution2D(
            batch_input_shape=(None, 28, 28, 1),
            filters=32,
            kernel_size=(5, 5),
            strides=1,
            padding='same',
            data_format='channels_last',
        )
    )
    model.add(Activation('relu'))
    # Pooling layer 1 (max pooling) output shape (32, 14, 14)
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_last',
    ))

    # Conv layer 2 output shape (64, 14, 14)
    model.add(Convolution2D(filters=64,
                            kernel_size=(2, 2),
                            strides=(2, 2),
                            padding='same',
                            data_format='channels_last',
                            ))
    model.add(Activation('relu'))
    # Pooling layer 2 (max pooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(pool_size=(2, 2),
                           data_format='channels_last'

                           ))
    # Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024) 3
    model.add(Flatten())
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    # Fully connected layer 2 to shape (10) for 10 classes 2
    model.add(Dense(8))
    model.add(Activation('softmax'))

    adam = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def load_data(csv_file, num_classes):
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    features = data[:, :-1]
    attack_one_hot = utils.to_categorical(data[:, -1], num_classes=num_classes)
    features_arr = []
    for i, item in enumerate(features):
        hexst = binascii.hexlify(item)
        fh = np.array([int(hexst[j:j + 2], 16)
                       for j in range(0, len(hexst), 2)])
        fh = np.pad(
            fh, (0, ((PNG_SIZE * PNG_SIZE) - len(fh))),
            'constant',
            constant_values=(0, 0))
        fh = np.uint8(fh)
        features_arr.append(fh)
    return np.array(features_arr), attack_one_hot


def draw_plot_by_history(_history):
    plt.style.use("ggplot")  # matplotlib的美化样式
    plt.figure()
    plt.plot(np.arange(0, epochs), _history.history["loss"],
             label="train_loss")
    plt.plot(np.arange(0, epochs), _history.history["categorical_accuracy"], label="train_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("../result.png")
    plt.show()


if __name__ == '__main__':
    evalulate_features, evalulate_labels = load_data(evaluate_file_path, number_classes)
    train_features, train_labels = load_data(train_file_path, number_classes)
    x_train = train_features.reshape(train_features.shape[0], PNG_SIZE, PNG_SIZE, 1) / 255.0
    x_evalulate = evalulate_features.reshape(evalulate_features.shape[0], PNG_SIZE, PNG_SIZE, 1) / 255.0
    model = create_cnn_model()
    model.summary()
    _history = model.fit(x=x_train, y=train_labels, epochs=epochs, batch_size=64, )
    loss, accuracy = model.evaluate(x=x_evalulate, y=evalulate_labels)
    draw_plot_by_history(_history)
    print('\ntest loss: ', loss)
    print('\ntest accuracy: ', accuracy)
