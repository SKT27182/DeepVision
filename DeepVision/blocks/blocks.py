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
