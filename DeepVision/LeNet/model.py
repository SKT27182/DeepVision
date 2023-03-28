import tensorflow as tf

class LeNet:

    """
    LeNet-5 model

    LeNet-5 is a convolutional neural network architecture that was proposed by Yann LeCun in 1998. It was the first successful attempt to apply convolutional neural networks to handwritten and machine-printed character recognition.

    In this implementation, the model is trained on the MNIST dataset.

    input_shape: 
        X: (None, 32, 32, channels)
        y: (None, 1, 1, classes)

    output_shape: 
        y: (None, 1, 1, classes)

    """

    def __init__(self, A=1.7159):
        self.A = A

    def scaled_tanh(self, x):
        return self.A * tf.nn.tanh(x)

    def lenet_5_exact(self, input_shape, classes):

        input_img = tf.keras.Input(shape=input_shape, name="input")

        c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="valid", input_shape=input_shape, name="C1")(input_img)
        c1 = tf.keras.layers.Activation(self.scaled_tanh)(c1)

        s2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S2")(c1)
        s2 = tf.keras.layers.Activation(tf.nn.sigmoid)(s2)

        c3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid", name="C3")(s2)
        c3 = tf.keras.layers.Activation(self.scaled_tanh)(c3)

        s4 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S4")(c3)
        s4 = tf.keras.layers.Activation(tf.nn.sigmoid)(s4)

        c5 = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), padding="valid", name="C5")(s4)
        c5 = tf.keras.layers.Activation(self.scaled_tanh)(c5)

        f6 = tf.keras.layers.Dense(units=84, name="F6")(c5)
        f6 = tf.keras.layers.Activation(self.scaled_tanh)(f6)

        f7 = tf.keras.layers.Dense(units=classes, name="F7")(f6)
        f7 = tf.keras.layers.Activation(tf.nn.softmax)(f7)

        model = tf.keras.Model(inputs=input_img, outputs=f7, name="LeNet-5")

        return model
    

    def lenet_5_mod_1(self, input_shape, classes):

        input_img = tf.keras.Input(shape=input_shape)

        c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="valid", input_shape=input_shape, name="C1")(input_img)
        c1 = tf.keras.layers.Activation(tf.nn.relu)(c1)

        s2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S2")(c1)

        c3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid", name="C3")(s2)
        c3 = tf.keras.layers.Activation(tf.nn.relu)(c3)

        s4 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="S4")(c3)

        c5 = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), padding="valid", name="C5")(s4)
        c5 = tf.keras.layers.Activation(tf.nn.relu)(c5)

        # modified
        f6 = tf.keras.layers.Flatten()(c5)

        f6 = tf.keras.layers.Dense(units=84, name="F6")(f6)
        f6 = tf.keras.layers.Activation(tf.nn.relu)(f6)

        f7 = tf.keras.layers.Dense(units=classes, name="F7")(f6)
        f7 = tf.keras.layers.Activation(tf.nn.softmax)(f7)

        model = tf.keras.Model(inputs=input_img, outputs=f7)

        return model
    

    def lenet_5_mod_2(self, input_shape, classes):


        input_img = tf.keras.Input(shape=input_shape)

        c1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same", input_shape=input_shape, name="C1")(input_img)
        c1 = tf.keras.layers.Activation(tf.nn.relu)(c1)

        s2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="S2")(c1)

        c3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid", name="C3")(s2)
        c3 = tf.keras.layers.Activation(tf.nn.relu)(c3)

        s4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="S4")(c3)

        c5 = tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), padding="valid", name="C5")(s4)
        c5 = tf.keras.layers.Activation(tf.nn.relu)(c5)

        f6 = tf.keras.layers.Flatten()(c5)

        f6 = tf.keras.layers.Dense(units=84, name="F6")(f6)
        f6 = tf.keras.layers.Activation(tf.nn.relu)(f6)

        f7 = tf.keras.layers.Dense(units=classes, name="F7")(f6)
        f7 = tf.keras.layers.Activation(tf.nn.softmax)(f7)

        model = tf.keras.Model(inputs=input_img, outputs=f7)

        return model
