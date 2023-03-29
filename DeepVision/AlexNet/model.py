import tensorflow as tf

class AlexNet:

    """
    AlexNet model
    
    AlexNet is a convolutional neural network architecture named after Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton. It was the first large-scale deep neural network to be trained on the ImageNet dataset.

    In this implementation, the model is trained on the CIFAR-10 dataset.

    input_shape: 
        X: (None, 224, 224, 3)
        y: (None,  classes)

    output_shape:
        y: (None, classes)

    """

    def __init__(self):
        pass

    def alexnet(self, input_shape, classes):

        input_img = tf.keras.Input(shape=input_shape, name="input")

        c1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), activation="relu", strides=(4, 4), padding="valid", name="Conv1")(input_img)
        n1 = tf.keras.layers.BatchNormalization(name="BatchNorm1")(c1)
        p1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="MaxPool1")(n1)

        c2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), activation="relu", strides=(1, 1), padding="same", name="Conv2")(p1)
        n2 = tf.keras.layers.BatchNormalization(name="BatchNorm2")(c2)
        p2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="MaxPool2")(n2)

        c3 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="same", name="Conv3")(p2)
        c4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="same", name="Conv4")(c3)
        c5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding="same", name="Conv5")(c4)

        p3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="MaxPool3")(c5)

        f1 = tf.keras.layers.Flatten(name="Flatten")(p3)
        d1 = tf.keras.layers.Dense(units=4096, activation="relu", name="FC1")(f1)
        do1 = tf.keras.layers.Dropout(rate=0.5, name="Dropout1")(d1)
        d2 = tf.keras.layers.Dense(units=4096, activation="relu", name="FC2")(do1)
        do2 = tf.keras.layers.Dropout(rate=0.5, name="Dropout2")(d2)
        d3 = tf.keras.layers.Dense(units=classes, name="FC3")(do2)


        model = tf.keras.Model(inputs=input_img, outputs=d3, name="AlexNet")

        return model
