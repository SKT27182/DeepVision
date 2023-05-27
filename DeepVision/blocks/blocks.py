import tensorflow as tf

class Blocks:
    def __init__(self,  name=None):
        self.name = name

    def __call__(self, x):

        return self.call(x=x)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return self.name

    def call(self, x):
        pass

class DenseBlock(Blocks):
    def __init__(self, n_layers, k=12, drop_rate =0.2, name=None):
        """
        DenseBlock
        ----------

        Parameters
        ----------

        n_layers : int
            number of dense layers in the block

        k : int
            growth rate of the block

        drop_rate : float
            dropout rate of the block

        name : str
            name of the block

        """
        self.n_layers = n_layers
        self.k = k
        self.drop_rate = drop_rate
        self.name = name

        super().__init__(name=name)

    def _dense_layer(self, inputs):
        x = inputs  # this is already the concatenated tensor from all previous layers
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=4 * self.k, kernel_size=1, strides=1, padding="same"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=self.k, kernel_size=3, strides=1, padding="same"
        )(x)

        # Add dropout layer

        x = tf.keras.layers.Dropout(self.drop_rate)(x)

        x = tf.keras.layers.Concatenate()([inputs , x])

        return x

    def call(self, inputs):

        x = inputs
        name = self.name

        for i in range(self.n_layers):
            x = self._dense_layer(x)

        block = tf.keras.Model(inputs=inputs, outputs=x, name="DenseBlock_" + name)

        return x, block
    
    def __call__(self, inputs):
        return self.call(inputs=inputs)



class InceptionNaiveBlock(Blocks):

    """
    InceptionNaiveBlock

    Parameters
    ----------

    filters : list
        list of filters for each convolutional layer

    name : str
        name of the block
    """
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

    """
    InceptionBlock

    Parameters
    ----------

    filters : list
        list of filters for each convolutional layer

    name : str
        name of the block
    """
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
    


class ResidualIdenticalBlock(Blocks):
    def __init__(self, filters, kernel_size=3, strides=1, name=None):

        self.name = name

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=1, padding="same"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        if strides != 1:
            self.downsample = tf.keras.layers.Conv2D(filters, 1, strides)
        else:
            self.downsample = lambda x: x

    def call(self, inputs):
        x = inputs
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # concat x and residual along the channel axis
        x = tf.concat([x, residual], axis=-1)

        x = tf.nn.relu(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name=self.name)
        return x, model
    
    def __call__(self, inputs):
        return self.call(inputs=inputs)
    

class ResidualBottleNeckBlock(Blocks):
    def __init__(self, filters, kernel_size=3, strides=1, name=None):
        self.name = name

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), strides=1, padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters * 4, kernel_size=(1, 1), strides=1, padding="same"
        )
        self.bn3 = tf.keras.layers.BatchNormalization()

        if strides != 1:
            self.downsample = tf.keras.layers.Conv2D(
                filters=filters * 4, kernel_size=(3, 3), strides=strides, padding="same"
            )
        else:
            self.downsample = lambda x: x

    def call(self, inputs):
        x = inputs
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # concat x and residual along the channel axis
        x = tf.concat([x, residual], axis=-1)

        x = tf.nn.relu(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name=self.name)
        return x, model
    
    def __call__(self, inputs):
        return self.call(inputs=inputs)



class ResPlainBlock(Blocks):
    def __init__(self, filters, kernel_size=3, strides=1, name=None):
        self.name = name

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=1, padding="same"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = tf.nn.relu(x)
        model = tf.keras.Model(inputs=inputs, outputs=x, name=self.name)
        return x, model
    
    def __call__(self, inputs):
        return self.call(inputs=inputs)


class MobileNetV1(Blocks):

    def __init__(self, filters, kernel_size=3, strides=1, activation="relu", alpha=1, name=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.alpha = alpha
        self.name = name

        super().__init__(name=name)


    def call(self, inputs):

        x = inputs

        filters = int(self.filters * self.alpha)

        # Depthwise Convolution
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size, strides=self.strides, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        # Pointwise Convolution
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        block = tf.keras.Model(inputs=inputs, outputs=x, name="DSC"+self.name)

        return x, block
    
    def __call__(self, inputs):

        return self.call(inputs=inputs)