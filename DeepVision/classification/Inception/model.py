import tensorflow as tf

class Blocks:
    def __init__(self, filters, name=None):
        self.name = name
        self.filters = filters

    def __call__(self, x):
        return self.call(x=x)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return self.name

    def call(self, x):
        pass

class InceptionNaiveBlock(Blocks):
    def __init__(self, filters, name=None):

        self.name = name

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters[0], kernel_size=1, padding="same", activation='relu'
        )

        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters[1], kernel_size=3, padding="same", activation='relu'
        )

        self.conv5 = tf.keras.layers.Conv2D(
            filters=filters[2], kernel_size=5, padding="same", activation='relu'
        )

        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=1, padding="same"
        )

    def call(self, inputs):
        x = inputs
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x4 = self.maxpool(x)
        x = tf.concat([x1, x2, x3, x4], axis=-1)
        model = tf.keras.models.Model(inputs=inputs, outputs=x, name=self.name)
        return x, model
    
    def __call__(self, inputs):
        return self.call(inputs=inputs)
    

class InceptionBlock(Blocks):
    def __init__(self, filters, name=None):
        
        self.name = name
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters[0], kernel_size=1, padding="same", activation='relu'
        )

        self.conv3_1 = tf.keras.layers.Conv2D(
            filters=filters[1], kernel_size=1, padding="same", activation='relu'
        )
        self.conv3_2 = tf.keras.layers.Conv2D(
            filters=filters[2], kernel_size=3, padding="same", activation='relu'
        )

        self.conv5_1 = tf.keras.layers.Conv2D(
            filters=filters[3], kernel_size=1, padding="same", activation='relu'
        )
        self.conv5_2 = tf.keras.layers.Conv2D(
            filters=filters[4], kernel_size=5, padding="same", activation='relu'
        )

        self.maxpool_1 = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=1, padding="same"
        )
        self.maxpool_2 = tf.keras.layers.Conv2D(
            filters=filters[5], kernel_size=1, padding="same", activation='relu'
        )

    def call(self, inputs):
        x = inputs

        x1 = self.conv1(x)

        x2 = self.conv3_1(x)
        x2 = self.conv3_2(x2)

        x3 = self.conv5_1(x)
        x3 = self.conv5_2(x3)

        x4 = self.maxpool_1(x)
        x4 = self.maxpool_2(x4)

        x = tf.concat([x1, x2, x3, x4], axis=-1)
        model = tf.keras.models.Model(inputs=inputs, outputs=x, name=self.name)
        return x, model
    
    def __call__(self, inputs):
        return self.call(inputs=inputs)
    

class GoogLeNet:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __preprocess(self, inputs):
        x = inputs

        # if input height and width is smaller than 224, then resize it to 224
        if (x.shape[1] < 224 or x.shape[2] < 224):
            x = tf.keras.layers.experimental.preprocessing.Resizing(
                height=224, width=224
            )(x)

        # if height and width are not equal then make a square with smaller dimension
        if (x.shape[1] != x.shape[2]):
            x = tf.keras.layers.experimental.preprocessing.CenterCrop(
                height=min(x.shape[1], x.shape[2]), width=min(x.shape[1], x.shape[2])
            )(x)

        x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(x)

        return x
    
    def __mid_output(self, input):
        
        x = input

        x = tf.keras.layers.AveragePooling2D(
            pool_size=5, strides=3
        )(x)

        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=1, padding="same", activation='relu'
        )(x)

        x = tf.keras.layers.Dense(
            units=1024, activation='relu'
        )(x)

        x = tf.keras.layers.Dropout(
            rate=0.7
        )(x)

        x = tf.keras.layers.Dense(
            units=self.output_shape, activation='softmax'
        )(x)

        x = tf.keras.layers.Flatten()(x)

        return x


    def build(self, pre_process = True, naive = False, mid_output = False):

        input = tf.keras.layers.Input(shape=self.input_shape)

        if pre_process:
            x = self.__preprocess(input)

        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=7, strides=2, padding="same", activation='relu'
        )(x)

        x = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding="same"
        )(x)

        x = tf.keras.layers.Conv2D(
            filters=192, kernel_size=3, strides=1, padding="same", activation='relu'
        )(x)

        x = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding="same"
        )(x)

        if naive:
            x, model = InceptionNaiveBlock(filters=[64, 128, 32], name="inception_3a")(x)
            x, model = InceptionNaiveBlock(filters=[128, 192, 96], name="inception_3b")(x)
        else:
            x, model = InceptionBlock(filters=[64, 96, 128, 16, 32, 32], name="inception_3a")(x)
            x, model = InceptionBlock(filters=[128, 128, 192, 32, 96, 64], name="inception_3b")(x)

        x = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding="same"
        )(x)

        if naive:
            x_4a, model = InceptionNaiveBlock(filters=[192, 208, 48], name="inception_4a")(x)
            x, model = InceptionNaiveBlock(filters=[160, 224, 64], name="inception_4b")(x_4a)
            x, model = InceptionNaiveBlock(filters=[128, 256, 64], name="inception_4c")(x)
            x_4e, model = InceptionNaiveBlock(filters=[112, 288, 64], name="inception_4d")(x)
            x, model = InceptionNaiveBlock(filters=[256, 320, 128], name="inception_4e")(x_4e)
        else:
            x_4a, model = InceptionBlock(filters=[192, 96, 208, 16, 48, 64], name="inception_4a")(x)
            x, model = InceptionBlock(filters=[160, 112, 224, 24, 64, 64], name="inception_4b")(x_4a)
            x, model = InceptionBlock(filters=[128, 128, 256, 24, 64, 64], name="inception_4c")(x)
            x_4e, model = InceptionBlock(filters=[112, 144, 288, 32, 64, 64], name="inception_4d")(x)
            x, model = InceptionBlock(filters=[256, 160, 320, 32, 128, 128], name="inception_4e")(x_4e)

        x = tf.keras.layers.MaxPool2D(
            pool_size=3, strides=2, padding="same"
        )(x)

        if naive:
            x, model = InceptionNaiveBlock(filters=[256, 320, 128], name="inception_5a")(x)
            x, model = InceptionNaiveBlock(filters=[384, 384, 128], name="inception_5b")(x)
        else:
            x, model = InceptionBlock(filters=[256, 160, 320, 32, 128, 128], name="inception_5a")(x)
            x, model = InceptionBlock(filters=[384, 192, 384, 48, 128, 128], name="inception_5b")(x)

        x = tf.keras.layers.AveragePooling2D(
            pool_size=7, strides=1, padding="valid"
        )(x)

        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(
            units=self.output_shape
        )(x)

        x = tf.keras.layers.Activation('softmax')(x)

        x = tf.keras.layers.Flatten()(x)

        if mid_output:
            x_4a = self.__mid_output(x_4a)
            x_4e = self.__mid_output(x_4e)

            model = tf.keras.models.Model(inputs=input, outputs=[x, x_4a, x_4e], name="googlenet")
        else:
            model = tf.keras.models.Model(inputs=input, outputs=x, name="googlenet")

        return model